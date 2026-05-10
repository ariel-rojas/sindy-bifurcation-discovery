#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script CLOUD-NATIVE de integración masiva de trayectorias (Takens-Bogdanov).
============================================================================

Genera trayectorias del sistema dinámico y las sube directamente a Google Drive
bajo una jerarquía basada en el nombre del sistema y la región de exploración.

Estructura en Drive:
    Carpeta Base
      └── <nombre_del_sistema>          (ej. "takens_bogdanov")
            └── <region>                (ej. "base", "far_z1", "far_z5")
                └── trajectories
                    ├── <key>.npz           (una por punto del barrido)
                    ├── t_eval.npz
                    └── grid_metadata_<sistema>_<region>.json

Compatibilidad
--------------
El nuevo `core/drive_auth.py` devuelve un ``AuthorizedSession`` (REST puro) en
lugar del clásico Drive service de ``googleapiclient``. Este script usa un
puente (`_build_drive_service`) que detecta el tipo del objeto.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import threading
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build

# =============================================================================
# CONFIGURACIÓN DE RUTAS E INICIALIZACIÓN DE REGIÓN
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System

from core.integrators import rk4_general
from core.io import make_param_key
from core.utils import generate_param_grid
from core.drive_auth import get_drive_service, get_target_folder_id

# --- SELECCIÓN DE REGIÓN DE EXPLORACIÓN (Fase 2) ---
TARGET_REGION = "far_z5" # Opciones: "base", "far_z1", "far_z2", "far_z5"
System.set_region(TARGET_REGION)

# =============================================================================
# CONFIGURACIÓN GENERAL Y JERARQUÍA DE CARPETAS
# =============================================================================
# Densidad de la grilla de parámetros (mu_1, mu_2)
GRID_DENSITY = 10

# Trayectorias: grilla de condiciones iniciales (N x N) por punto del barrido
N_TRAJ_PER_AXIS = 5

# Tamaño de lote para no ahogar la RAM ni el disco
BATCH_SIZE = 50

# Jerarquía Dinámica en Drive
SYSTEM_FOLDER_NAME = System.name.lower().replace("-", "_").replace(" ", "_")
TRAJECTORIES_FOLDER_NAME = "trajectories"

# Archivos locales (solo metadata + temporales)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
JSON_FILE = os.path.join(
    OUTPUT_DIR,
    f"grid_metadata_{SYSTEM_FOLDER_NAME}_{TARGET_REGION}.json",
)
TMP_DIR = os.path.join(tempfile.gettempdir(), f"sindy_sim_tmp_{SYSTEM_FOLDER_NAME}_{TARGET_REGION}")

# Tiempo de integración
T_SPAN = [0.0, 10]
N_STEPS = 1000
DT = np.float32((T_SPAN[1] - T_SPAN[0]) / (N_STEPS - 1))
T_EVAL = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS, dtype=np.float32)

# Resolución del campo vectorial y workers
VF_RESOLUTION = 100
MAX_WORKERS_CPU = min(4, os.cpu_count() or 1)
MAX_WORKERS_IO = 5
MAX_RETRIES = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


# =============================================================================
# UTILIDADES GENERALES
# =============================================================================
def clean_tmp_directory(tmp_dir):
    """Borra completamente el directorio temporal y lo recrea vacío."""
    if os.path.exists(tmp_dir):
        print(f"Limpiando directorio temporal: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)


# =============================================================================
# PUENTE DE COMPATIBILIDAD CON EL NUEVO drive_auth.py
# =============================================================================
def _build_drive_service(raw):
    if hasattr(raw, "files"):
        return raw  # ya es un Drive service de googleapiclient
    creds = getattr(raw, "credentials", raw)
    return build("drive", "v3", credentials=creds)


# =============================================================================
# FUNCIONES DE GOOGLE DRIVE (THREAD-SAFE VÍA THREAD-LOCAL)
# =============================================================================
thread_local = threading.local()

def get_thread_local_service():
    if not hasattr(thread_local, "service"):
        thread_local.service = _build_drive_service(get_drive_service())
    return thread_local.service

def create_or_get_subfolder(service, parent_id, folder_name):
    """Busca una subcarpeta por nombre; si no existe, la crea."""
    query = (
        f"name='{folder_name}' and '{parent_id}' in parents and "
        f"mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if not items:
        print(f"Creando la subcarpeta '{folder_name}' en Google Drive...")
        metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id],
        }
        return service.files().create(body=metadata, fields='id').execute().get('id')
    return items[0]['id']

def get_existing_files(service, folder_id):
    """Lista todos los nombres de archivos ya presentes en una carpeta (paginado)."""
    print("Sincronizando inventario de la carpeta en Drive...")
    existing = set()
    page_token = None
    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='nextPageToken, files(name)',
            pageToken=page_token,
            pageSize=1000,
        ).execute()
        for item in results.get('files', []):
            existing.add(item['name'])
        page_token = results.get('nextPageToken', None)
        if not page_token:
            break
    return existing

def upload_to_drive(file_path, file_name, folder_id):
    """Sube un archivo local a Drive con reintentos exponenciales."""
    service = get_thread_local_service()
    metadata = {'name': file_name, 'parents': [folder_id]}
    for attempt in range(3):
        try:
            media = MediaFileUpload(
                file_path,
                mimetype='application/octet-stream',
                resumable=True,
            )
            request = service.files().create(media_body=media, body=metadata)
            response = None
            while response is None:
                _, response = request.next_chunk()
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            return True, file_name
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return False, str(e)


# =============================================================================
# WORKER: CÓMPUTO (CPU)
# =============================================================================
def run_simulation_job(param_arr):
    """Integra todas las trayectorias para un punto (mu_1, mu_2) y guarda un .npz local."""
    ode_func = System.get_ode_jit()
    vf_func = System.get_vector_field_jit()

    fixed_points = None
    if hasattr(System, 'calculate_fixed_points'):
        fixed_points = System.calculate_fixed_points(param_arr)

    n_fp = fixed_points.shape[0] if fixed_points is not None else 0

    x_lim, y_lim = list(System.state_limits[0]), list(System.state_limits[1])
    if fixed_points is not None and fixed_points.size > 0:
        fp_x = fixed_points[:, 0]
        x_lim[0] = min(x_lim[0], np.min(fp_x) - 1.0)
        x_lim[1] = max(x_lim[1], np.max(fp_x) + 1.0)
        y_lim[0] = min(y_lim[0], x_lim[0])
        y_lim[1] = max(y_lim[1], x_lim[1])

    x_min, x_max = np.float32(x_lim[0]), np.float32(x_lim[1])
    y_min, y_max = np.float32(y_lim[0]), np.float32(y_lim[1])

    x_vals = np.linspace(x_min, x_max, VF_RESOLUTION, dtype=np.float32)
    y_vals = np.linspace(y_min, y_max, VF_RESOLUTION, dtype=np.float32)
    U, V = vf_func(x_vals, y_vals, param_arr)

    tx = np.linspace(x_min, x_max, N_TRAJ_PER_AXIS, dtype=np.float32)
    ty = np.linspace(y_min, y_max, N_TRAJ_PER_AXIS, dtype=np.float32)

    trajectories = []
    t_start, t_end = np.float32(T_SPAN[0]), np.float32(T_SPAN[1])

    for y0_val in ty:
        for x0_val in tx:
            y0 = np.array([x0_val, y0_val], dtype=np.float32)
            sol = rk4_general(ode_func, y0, t_start, t_end, DT, param_arr)
            trajectories.append(sol)

    traj_array = np.stack(trajectories)
    key = make_param_key(param_arr)
    out_path = os.path.join(TMP_DIR, f"{key}.npz")

    data_to_save = {
        "trajectories": traj_array,
        "fixed_points": fixed_points,
        "U": U,
        "V": V,
        "x_vals": x_vals,
        "y_vals": y_vals,
    }
    np.savez_compressed(out_path, **data_to_save)

    return key, out_path, n_fp


# =============================================================================
# WORKER: I/O A LA NUBE (RED)
# =============================================================================
def run_cloud_io_job(key, npz_path, folder_id, n_fp):
    """Sube un .npz pre-computado a la carpeta destino en Drive."""
    success, msg = upload_to_drive(npz_path, f"{key}.npz", folder_id)
    return success, key, n_fp, msg


# =============================================================================
# METADATA: CABECERA "_system_info"
# =============================================================================
def build_system_info():
    """Resumen de configuración del barrido para que el viewer pueda autoajustarse."""
    return {
        "system_name": System.name,
        "region": TARGET_REGION,
        "state_names": list(System.state_names),
        "param_names": list(System.param_names),
        "sweep_ranges": [list(r) for r in System.param_ranges],
        "grid_density": GRID_DENSITY,
        "n_traj_per_axis": N_TRAJ_PER_AXIS,
        "vf_resolution": VF_RESOLUTION,
        "t_span": list(T_SPAN),
        "n_steps": N_STEPS,
    }


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def main():
    clean_tmp_directory(TMP_DIR)

    print("--- Iniciando Cómputo Directo a la Nube ---")
    print(f"Sistema: {System.name} | Región: {TARGET_REGION} | Grilla: {GRID_DENSITY}^2 puntos")

    # 1. Conexión a Drive (con puente al googleapiclient clásico)
    try:
        raw_service = get_drive_service()
        main_service = _build_drive_service(raw_service)

        base_folder_id = get_target_folder_id()

        # Carpeta del sistema (ej. "takens_bogdanov")
        system_folder_id = create_or_get_subfolder(
            main_service, base_folder_id, SYSTEM_FOLDER_NAME
        )
        
        # Subcarpeta de la Región (Novedad Fase 2)
        region_folder_id = create_or_get_subfolder(
            main_service, system_folder_id, TARGET_REGION
        )

        # Subcarpeta de trayectorias
        target_folder_id = create_or_get_subfolder(
            main_service, region_folder_id, TRAJECTORIES_FOLDER_NAME
        )

        existing_files = get_existing_files(main_service, target_folder_id)
        print(f"Archivos ya asegurados en la nube: {len(existing_files)}")
    except Exception as e:
        print(f"Error fatal de conexión con Drive: {e}")
        return

    # 2. Subir vector de tiempo si hace falta
    if "t_eval.npz" not in existing_files:
        print("Subiendo t_eval.npz base...")
        t_path = os.path.join(TMP_DIR, "t_eval.npz")
        np.savez_compressed(t_path, t_eval=T_EVAL)
        upload_to_drive(t_path, "t_eval.npz", target_folder_id)

    # 3. Cargar metadata histórica (por si estamos reanudando)
    metadata = {}
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ JSON local corrupto. Reiniciando metadata.")
                metadata = {}

    # 3.b Refrescar la cabecera _system_info (lo lee el viewer interactivo)
    metadata["_system_info"] = build_system_info()

    # 4. Auto-reparación de metadata
    print("Generando combinaciones y verificando integridad de la metadata...")
    all_params = generate_param_grid(System.param_ranges, GRID_DENSITY)

    jobs_to_run = []
    for p in all_params:
        key_str = make_param_key(p)
        if f"{key_str}.npz" not in existing_files:
            jobs_to_run.append(p)
        else:
            if key_str not in metadata:
                fixed_points = System.calculate_fixed_points(p)
                n_fp = fixed_points.shape[0] if fixed_points is not None else 0
                metadata[key_str] = {"num_fixed_points": int(n_fp)}

    with open(JSON_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    if not jobs_to_run:
        print("✅ Todas las simulaciones ya están en Google Drive para esta región.")
        _upload_metadata_with_dedup(main_service, target_folder_id)
        return

    print(f"Pendientes de cálculo: {len(jobs_to_run)}")

    # 5. Bucle Principal con Reintentos
    for attempt in range(MAX_RETRIES):
        if not jobs_to_run:
            break
        print(f"\n--- Lote de Ejecución: Intento {attempt + 1} de {MAX_RETRIES} ---")

        failed_params = []

        with ProcessPoolExecutor(max_workers=MAX_WORKERS_CPU) as cpu_pool, \
             ThreadPoolExecutor(max_workers=MAX_WORKERS_IO) as io_pool, \
             tqdm(total=len(jobs_to_run), desc="Procesando") as pbar:

            for i in range(0, len(jobs_to_run), BATCH_SIZE):
                batch_params = jobs_to_run[i:i + BATCH_SIZE]

                cpu_futures = {cpu_pool.submit(run_simulation_job, p): p
                               for p in batch_params}
                io_futures = {}

                for c_future in as_completed(cpu_futures):
                    original_param = cpu_futures[c_future]
                    try:
                        key_str, npz_path, n_fp = c_future.result()
                        i_future = io_pool.submit(
                            run_cloud_io_job, key_str, npz_path,
                            target_folder_id, n_fp,
                        )
                        io_futures[i_future] = original_param
                    except Exception as e:
                        print(f"\n[Fallo CPU] {make_param_key(original_param)}: {e}")
                        failed_params.append(original_param)
                        pbar.update(1)

                for i_future in as_completed(io_futures):
                    original_param = io_futures[i_future]
                    success, key_str, n_fp, msg = i_future.result()

                    if success:
                        metadata[key_str] = {"num_fixed_points": int(n_fp)}
                    else:
                        print(f"\n[Fallo Red] {key_str}: {msg}")
                        failed_params.append(original_param)

                    pbar.update(1)

                with open(JSON_FILE, "w") as f:
                    json.dump(metadata, f, indent=2)

        jobs_to_run = failed_params

        if jobs_to_run:
            print(f"⚠️ {len(jobs_to_run)} trabajos fallaron. Reintentando...")

    # 6. Subida final de la metadata
    print("\nRespaldando Metadata en Google Drive...")
    _upload_metadata_with_dedup(main_service, target_folder_id)

    print(f"\n✅ ¡Proceso 100% Finalizado! Datos de la región '{TARGET_REGION}' seguros en la nube.")


def _upload_metadata_with_dedup(service, target_folder_id):
    json_name = os.path.basename(JSON_FILE)
    try:
        old_json = service.files().list(
            q=f"name='{json_name}' and '{target_folder_id}' in parents and trashed=false",
            fields='files(id)',
        ).execute()
        for item in old_json.get('files', []):
            try:
                service.files().delete(fileId=item['id']).execute()
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ No se pudo limpiar JSONs previos: {e}")

    upload_to_drive(JSON_FILE, json_name, target_folder_id)


# =============================================================================
# WARMUP DEL JIT + ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("Pre-compilando JIT (Warm-up)...")
    try:
        ode = System.get_ode_jit()
        ode(
            np.float32(0.0),
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.1, 0.1], dtype=np.float32),
        )
        print("JIT listo.")
    except Exception:
        pass

    main()