#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script CLOUD-NATIVE de Validación del Modelo SINDy (Versión Dimensional).

FUNCIONALIDAD:
- Conecta a Google Drive y descarga un modelo SINDy específico.
- Extrae la configuración temporal (t_start, t_end, dt) de t_eval.npz manejando paginación.
- Muestreo Estratificado Geométrico: Evalúa curvas teóricas, clasifica el espacio 
  en Zonas Dinámicas y selecciona 'N' configuraciones vírgenes.
- Meta-Programación Numba: Genera el código del sistema de ecuaciones al vuelo 
  basado en las características (features) aprendidas por SINDy.
- Pipeline de 2 Fases: Cálculo pesado en CPU (RK4) -> Subida asíncrona a Drive.
"""

import os
import sys
import json
import random
import tempfile
import numpy as np
import joblib
import io
import time
import shutil
import threading
from tqdm import tqdm
from numba import jit
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.discovery import build
# =============================================================================
# CONFIGURACIÓN DE RUTAS E IMPORTACIONES
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

from experiments.syrinx.config import EXPERIMENT
from systems.syrinx import SyrinxModel as System
from core.integrators import rk4_general
from core.io import parse_param_key
from core.drive_auth import get_drive_service, get_target_folder_id

import pysindy as ps

# Fix retrocompatibilidad de Joblib
if 'pysindy.pysindy' not in sys.modules: sys.modules['pysindy.pysindy'] = ps
if 'pysindy.utils.axes' not in sys.modules: sys.modules['pysindy.utils.axes'] = ps.utils

# =============================================================================
# PARÁMETROS DE EJECUCIÓN
# =============================================================================
VERSION_TO_LOAD = 22         # La versión del modelo a validar (v1, v2, etc.)

N_SIMS_PER_ZONE = 5         # Cuántas configuraciones NUEVAS simular por cada zona
N_TRAJ_PER_AXIS = 10        # Grilla de condiciones iniciales por configuración (ej. 10x10=100)
GRID_DENSITY_VF = 100       # Resolución de la grilla para el campo vectorial

# Directorios
SYSTEM_FOLDER_NAME = System.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
TRAJECTORIES_FOLDER_NAME = "trajectories"
MODELS_FOLDER_NAME = "sindy_models"
SIMS_OUT_FOLDER_NAME = "simulations"
METADATA_FILE_NAME = f"grid_metadata_{SYSTEM_FOLDER_NAME}.json"

# Recursos (Ajustados para Windows/Cloud)
MAX_WORKERS_CPU = min(6, os.cpu_count() or 1)
MAX_WORKERS_IO = 4

# Caché Local Temporal
TMP_DIR = os.path.join(tempfile.gettempdir(), f"sindy_val_cache_{SYSTEM_FOLDER_NAME}")
os.makedirs(TMP_DIR, exist_ok=True)

# =============================================================================
# UTILIDADES DE GOOGLE DRIVE ROBUSTAS (THREAD-SAFE & PAGINATION)
# =============================================================================
thread_local = threading.local()

def get_thread_local_service():
    if not hasattr(thread_local, "service"):
        raw_service = get_drive_service()
        # FIX: Si get_drive_service() retorna un AuthorizedSession, construimos el cliente API v3
        if not hasattr(raw_service, 'files'):
            creds = getattr(raw_service, 'credentials', raw_service)
            thread_local.service = build('drive', 'v3', credentials=creds)
        else:
            thread_local.service = raw_service
    return thread_local.service

def create_or_get_subfolder(service, parent_id, folder_name):
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if not items:
        meta = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_id]}
        return service.files().create(body=meta, fields='id').execute().get('id')
    return items[0]['id']

def get_drive_index_dict(service, folder_id):
    """Obtiene TODOS los archivos de una carpeta manejando la paginación de la API."""
    idx = {}
    page_token = None
    while True:
        res = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive', fields='nextPageToken, files(id, name)',
            pageToken=page_token, pageSize=1000
        ).execute()
        for item in res.get('files', []):
            idx[item['name']] = item['id']
        page_token = res.get('nextPageToken', None)
        if not page_token: break
    return idx

def download_file_to_cache(service, file_id, file_name):
    local_path = os.path.join(TMP_DIR, file_name)
    if os.path.exists(local_path): return local_path 
    for attempt in range(4): 
        try:
            request = service.files().get_media(fileId=file_id)
            fh = io.FileIO(local_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done: _, done = downloader.next_chunk()
            return local_path
        except Exception as e:
            try: fh.close()
            except: pass
            if attempt < 3: time.sleep(2 ** attempt)
            else: raise ConnectionError(f"Fallo al descargar {file_name}: {e}")

def upload_file_to_drive(local_path, file_name, folder_id):
    service = get_thread_local_service()
    for attempt in range(4):
        try:
            with open(local_path, "rb") as fd:
                media = MediaIoBaseUpload(fd, mimetype='application/octet-stream', resumable=True)
                meta = {'name': file_name, 'parents': [folder_id]}
                service.files().create(media_body=media, body=meta).execute()
            if os.path.exists(local_path): os.remove(local_path)
            return True, file_name
        except Exception as e:
            if attempt < 3: time.sleep(2 ** attempt)
            else: return False, str(e)

# =============================================================================
# LÓGICA DE CLASIFICACIÓN GEOMÉTRICA DE ZONAS
# =============================================================================
def build_boundary_interpolator(curve_data, x_key, y_key):
    x_v, y_v = curve_data[x_key], curve_data[y_key]
    valid = np.isfinite(x_v) & np.isfinite(y_v)
    x_v, y_v = x_v[valid], y_v[valid]
    if len(y_v) == 0: return lambda y: np.nan
    
    sort_idx = np.argsort(y_v)
    y_sorted, x_sorted = y_v[sort_idx], x_v[sort_idx]
    y_uniq, unq_idx = np.unique(y_sorted, return_index=True)
    x_uniq = x_sorted[unq_idx]
    
    return lambda y_input: float(np.interp(y_input, y_uniq, x_uniq, left=np.nan, right=np.nan))

def classify_grid_zones(metadata_keys, phys_base):
    print("\nCalculando fronteras geométricas para estratificación...")
    curves = System.get_physical_bifurcation_curves(
        physical_params=phys_base, x_min=0.001, x_max=0.5, n_points=50000
    )
    
    hopf_func = build_boundary_interpolator(curves.get("Hopf", {}), "psub", "kappa1")
    sn_func = build_boundary_interpolator(curves.get("Saddle-Node", {}), "psub", "kappa1")
    
    zone_map = {
        "Zona 1 (Izquierda de Hopf)": [],
        "Zona 2 (Entre Hopf y SN)": [],
        "Zona 3 (Derecha de SN)": [],
        "Zona 0 (Sin fronteras teóricas)": []
    }
    
    for key in metadata_keys:
        vals = parse_param_key(key)
        if len(vals) < 2: continue
        
        kappa1_val, psub_val = float(vals[0]), float(vals[1])
        p_hopf = hopf_func(kappa1_val)
        p_sn = sn_func(kappa1_val)
        
        if np.isnan(p_hopf) and np.isnan(p_sn): zona = "Zona 0 (Sin fronteras teóricas)"
        elif np.isnan(p_hopf): zona = "Zona 1 (Izquierda de Hopf)" if psub_val <= p_sn else "Zona 3 (Derecha de SN)"
        elif np.isnan(p_sn): zona = "Zona 1 (Izquierda de Hopf)" if psub_val <= p_hopf else "Zona 3 (Derecha de SN)"
        else:
            left_b, right_b = min(p_hopf, p_sn), max(p_hopf, p_sn)
            if psub_val <= left_b: zona = "Zona 1 (Izquierda de Hopf)"
            elif psub_val <= right_b: zona = "Zona 2 (Entre Hopf y SN)"
            else: zona = "Zona 3 (Derecha de SN)"
            
        zone_map[zona].append(key)
        
    return zone_map

# =============================================================================
# FÁBRICAS DE FUNCIONES JIT PARA RK4 (CON META-PROGRAMACIÓN)
# =============================================================================

def generate_sindy_ode_jit(coeffs, feature_names):
    """
    Genera y compila dinámicamente la ecuación ODE en Numba leyendo
    los nombres de las características directamente del modelo SINDy.
    """
    state_dim = len(System.state_names)
    coeffs_states = coeffs[:state_dim, :].astype(np.float32)
    coeffs_T = np.ascontiguousarray(coeffs_states.T)
    
    # El param_arr que pasamos aquí viene de parse_param_key, por lo que es estrictamente [kappa1, psub]
    lines = [
        "@jit(nopython=True)",
        "def sindy_ode_wrapper(t, state_arr, param_arr):",
        "    x = state_arr[0]",
        "    y = state_arr[1]",
        "    kappa1 = param_arr[0]",
        "    psub = param_arr[1]",
        f"    theta = np.empty({len(feature_names)}, dtype=np.float32)"
    ]

    for i, feat in enumerate(feature_names):
        if feat == "1":
            val = "1.0"
        else:
            val = feat.replace(" ", " * ").replace("^", "**")
        lines.append(f"    theta[{i}] = {val}")

    lines.append("    d_state = np.dot(theta, coeffs_T)")
    lines.append("    return d_state.astype(np.float32)")

    code = "\n".join(lines)
    
    namespace = {"np": np, "jit": jit, "coeffs_T": coeffs_T}
    exec(code, namespace)
    return namespace["sindy_ode_wrapper"]

def test_sindy_stability(model, X_list, U_list, t_list):
    """Simula una muestra de condiciones iniciales para descartar modelos que divergen."""
    feature_names = model.get_feature_names()
    ode_func = generate_sindy_ode_jit(model.optimizer.coef_, feature_names)

    n_tests = min(3, len(X_list))
    test_indices = random.sample(range(len(X_list)), n_tests)
    max_safe_val = 1e5 

    for idx in test_indices:
        x0 = X_list[idx][0].astype(np.float32)
        t_array = t_list[idx]
        dt = np.float32(t_array[1] - t_array[0])
        t_start, t_end = np.float32(t_array[0]), np.float32(t_array[-1])

        # U_list[idx] tiene shape (N_steps, 2), extraemos la dupla (kappa1, psub)
        param_arr = U_list[idx][0].astype(np.float32)

        sol = rk4_general(ode_func, x0, t_start, t_end, dt, param_arr)

        expected_steps = int(np.round((t_end - t_start) / dt)) + 1
        if sol.shape[1] < expected_steps - 1 or np.any(np.abs(sol) > max_safe_val) or np.any(np.isnan(sol)):
            return False # Diverge

    return True 
def make_sindy_vf_jit(sindy_ode_func):
    @jit(nopython=True, parallel=True) 
    def vf_func(x_vals, y_vals, param_arr):
        nx, ny = len(x_vals), len(y_vals)
        U, V = np.zeros((ny, nx), dtype=np.float32), np.zeros((ny, nx), dtype=np.float32)
        t = np.float32(0.0)
        
        for i in range(ny):
            for j in range(nx):
                state = np.array([x_vals[j], y_vals[i]], dtype=np.float32)
                d_state = sindy_ode_func(t, state, param_arr)
                U[i, j], V[i, j] = d_state[0], d_state[1]
        return U, V
    return vf_func

# =============================================================================
# WORKERS DE PROCESAMIENTO CPU
# =============================================================================
_W_ODE = None
_W_VF = None
_W_TPARAMS = None

def worker_init(coeffs, time_params, feature_names):
    global _W_ODE, _W_VF, _W_TPARAMS
    
    # Se genera la función de forma dinámica en cada subproceso
    _W_ODE = generate_sindy_ode_jit(coeffs, feature_names)
    _W_VF = make_sindy_vf_jit(_W_ODE)
    _W_TPARAMS = time_params
    
    # Warm-up (Compilación temprana)
    _W_ODE(0.0, np.zeros(len(System.state_names), dtype=np.float32), np.zeros(2, dtype=np.float32))

def run_validation_job(param_key_str):
    # Aseguramos el casteo explícito a array float32 para que Numba no falle
    param_arr = np.array(parse_param_key(param_key_str), dtype=np.float32)
    t_start, t_end, dt = _W_TPARAMS
    
    x_min, x_max = np.float32(EXPERIMENT.phase_space.x_lim_physical[0]), np.float32(EXPERIMENT.phase_space.x_lim_physical[1])
    y_min, y_max = np.float32(EXPERIMENT.phase_space.y_lim_physical[0]), np.float32(EXPERIMENT.phase_space.y_lim_physical[1])

    x_vals = np.linspace(x_min, x_max, GRID_DENSITY_VF, dtype=np.float32)
    y_vals = np.linspace(y_min, y_max, GRID_DENSITY_VF, dtype=np.float32)
    U, V = _W_VF(x_vals, y_vals, param_arr)

    tx = np.linspace(x_min, x_max, N_TRAJ_PER_AXIS, dtype=np.float32)
    ty = np.linspace(y_min, y_max, N_TRAJ_PER_AXIS, dtype=np.float32)
    
    trajectories = []
    for y0_val in ty:
        for x0_val in tx:
            y0 = np.array([x0_val, y0_val], dtype=np.float32)
            sol = rk4_general(_W_ODE, y0, t_start, t_end, dt, param_arr)
            trajectories.append(sol)
    
    # FIX: Crear un ragged array de tipo object para permitir trayectorias de distinto tamaño
    traj_array = np.empty(len(trajectories), dtype=object)
    for i, tr in enumerate(trajectories):
        traj_array[i] = tr
        
    out_path = os.path.join(TMP_DIR, f"{param_key_str}.npz")
    np.savez_compressed(out_path, trajectories=traj_array, U=U, V=V, x_vals=x_vals, y_vals=y_vals) 
    
    return param_key_str, out_path

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def main():
    print(f"--- Validación Cloud SINDy (Modelo: {System.name} | v{VERSION_TO_LOAD}) ---")
    
    if os.path.exists(TMP_DIR): shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    
    try:
        raw_service = get_drive_service()
        if not hasattr(raw_service, 'files'):
            creds = getattr(raw_service, 'credentials', raw_service)
            service = build('drive', 'v3', credentials=creds)
        else:
            service = raw_service

        base_id = get_target_folder_id()
        
        sys_id = create_or_get_subfolder(service, base_id, SYSTEM_FOLDER_NAME)
        traj_id = create_or_get_subfolder(service, sys_id, TRAJECTORIES_FOLDER_NAME)
        models_root_id = create_or_get_subfolder(service, sys_id, MODELS_FOLDER_NAME)
        
        # CORRECCIÓN AQUÍ: Apuntar directamente a "rs_batch" tal como lo genera el Script 1
        batch_folder_id = create_or_get_subfolder(service, models_root_id, "rs_batch")
        
        # 1. Verificar existencia de la versión del modelo
        v_results = service.files().list(q=f"name='v{VERSION_TO_LOAD}' and '{batch_folder_id}' in parents and trashed=false", fields='files(id)').execute()
        if not v_results.get('files'):
            raise FileNotFoundError(f"La versión v{VERSION_TO_LOAD} no existe en la carpeta rs_batch.")
        
        version_folder_id = v_results.get('files')[0]['id']
        sims_folder_id = create_or_get_subfolder(service, version_folder_id, SIMS_OUT_FOLDER_NAME)
        
        # 2. Descargar Metadatos de la Grilla Original
        print("Sincronizando índice de trayectorias...")
        
        drive_index_traj = get_drive_index_dict(service, traj_id)
        
        if METADATA_FILE_NAME not in drive_index_traj:
            raise FileNotFoundError(f"Falta el metadata JSON en la carpeta trajectories de Drive.")
            
        meta_path = download_file_to_cache(service, drive_index_traj[METADATA_FILE_NAME], METADATA_FILE_NAME)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        sys_info = metadata.get("_system_info", {})
        phys_base = sys_info.get("physical_base_params", System.default_params)
        
        # 3. Extraer Setup Temporal Directamente del Entrenamiento
        if "t_eval.npz" not in drive_index_traj:
            raise FileNotFoundError("No se encontró 't_eval.npz' en Drive.")
            
        teval_path = download_file_to_cache(service, drive_index_traj["t_eval.npz"], "t_eval.npz")
        with np.load(teval_path) as data: t_full = data["t_eval"]
        
        t_start = np.float32(t_full[0])
        t_end = np.float32(t_full[-1])
        dt = np.float32(t_full[1] - t_full[0])
        time_params = (t_start, t_end, dt)
        print(f"Setup Temporal Extraído: T_max = {t_end}s | dt = {dt}s")

        # 4. Descargar Modelo SINDy
        print("Buscando y descargando modelo SINDy a testear...")
        files_in_version = get_drive_index_dict(service, version_folder_id)
        
        model_path = download_file_to_cache(service, files_in_version["sindy_model.joblib"], "sindy_model.joblib")
        params_path = download_file_to_cache(service, files_in_version["sindy_training_params.json"], "sindy_training_params.json")
        
    except Exception as e:
        print(f"Error Crítico accediendo a la Nube: {e}")
        return

    # 5. Cargar parámetros de entrenamiento y Características
    model = joblib.load(model_path)
    coeffs = model.optimizer.coef_ 
    feature_names = model.get_feature_names()  # EXTRAEMOS LAS FEATURES PARA NUMBA
    
    with open(params_path, "r") as f: training_params = json.load(f)
    train_keys = set(training_params.get("keys_used", []))

    existing_sims = set(get_drive_index_dict(service, sims_folder_id).keys())
    if "t_eval.npz" not in existing_sims:
        out_teval = os.path.join(TMP_DIR, "t_eval.npz")
        np.savez_compressed(out_teval, t_eval=t_full)
        with open(out_teval, "rb") as fd:
            media = MediaIoBaseUpload(fd, mimetype='application/octet-stream', resumable=True)
            service.files().create(media_body=media, body={'name': 't_eval.npz', 'parents': [sims_folder_id]}).execute()

    # 6. Muestreo Estratificado Basado en Zonas Dinámicas
    metadata_keys = [k.replace(".npz", "") for k in metadata.keys() if not k.startswith("_")]
    zone_map = classify_grid_zones(metadata_keys, phys_base)
    
    jobs_keys = []
    print("\nAsignación de Simulaciones Virgenes (No usadas en entrenamiento):")
    for zone_id, candidates_in_zone in zone_map.items():
        virgen_candidates = [k for k in candidates_in_zone if f"{k}.npz" not in train_keys]
        actual_sample_size = min(N_SIMS_PER_ZONE, len(virgen_candidates))
        
        if actual_sample_size > 0:
            selected = random.sample(virgen_candidates, actual_sample_size)
            jobs_keys.extend(selected)
            print(f"  > {zone_id}: Seleccionadas {actual_sample_size} configuraciones.")
        else:
            print(f"  > {zone_id}: No hay suficientes configuraciones vírgenes.")

    jobs_keys = [k for k in jobs_keys if f"{k}.npz" not in existing_sims]

    if not jobs_keys:
        print("\n✅ Todas las simulaciones requeridas ya se encuentran en Google Drive.")
        return

    # 7. Pipeline Fase 1 (Matemática Pura CPU)
    print(f"\nSimulando {len(jobs_keys)} configuraciones (CPU Pool)...")
    archivos_generados = []
    
    # Pasamos 'feature_names' al inicializador de los workers
    with ProcessPoolExecutor(max_workers=MAX_WORKERS_CPU, initializer=worker_init, initargs=(coeffs, time_params, feature_names)) as cpu_pool:
        futures = {cpu_pool.submit(run_validation_job, k): k for k in jobs_keys}
        for f in tqdm(as_completed(futures), total=len(jobs_keys), desc="Fase 1/2: Integración RK4 (CPU)"):
            try:
                key_str, npz_path = f.result()
                archivos_generados.append((key_str, npz_path))
            except Exception as e:
                print(f"\n[Error CPU] Falla en configuración {futures[f]}: {e}")

    # 8. Pipeline Fase 2 (I/O Red)
    if archivos_generados:
        print("\nIniciando respaldo en la nube (IO Pool)...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_IO) as io_pool:
            upload_futures = {
                io_pool.submit(upload_file_to_drive, path, f"{key}.npz", sims_folder_id): key 
                for key, path in archivos_generados
            }
            for f in tqdm(as_completed(upload_futures), total=len(archivos_generados), desc="Fase 2/2: Subida a Drive"):
                try:
                    success, msg = f.result()
                    if not success: print(f"\n[Error Red] No se pudo subir {upload_futures[f]}: {msg}")
                except Exception: pass

    print("\nLimpiando caché temporal...")
    if os.path.exists(TMP_DIR): shutil.rmtree(TMP_DIR, ignore_errors=True)

    print(f"✅ Validación de {len(jobs_keys)} configuraciones completada exitosamente en v{VERSION_TO_LOAD}/simulations/.")

if __name__ == "__main__":
    main()