#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script CLOUD-NATIVE de Validación Cruzada SINDy (Ensamble).

FUNCIONALIDAD MÁXIMA (Fase 5):
- Define TRAIN_REGION y TEST_REGION.
- Descarga modelos generados en TRAIN_REGION.
- Modifica los límites espaciales y la matemática evaluando sobre TEST_REGION.
- Guarda las carpetas de simulación dinámicamente como "simulations_[TEST_REGION]" 
  para no pisar resultados si se valida el mismo modelo en zonas distintas.
"""

import os
import sys
import json
import random
import tempfile
import numpy as np
import pysindy as ps
import io
import time
import shutil
import threading
from tqdm import tqdm
from numba import jit
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# =============================================================================
# CONFIGURACIÓN DE RUTAS E IMPORTACIONES
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System
from experiments.takens_bogdanov.data_zone_manager import DataZoneManager
from core.integrators import rk4_general
from core.io import parse_param_key
from core.drive_auth import get_drive_service, get_target_folder_id

# --- CONFIGURACIÓN DE VALIDACIÓN CRUZADA (Fase 5) ---
TRAIN_REGION = "far_z5"  # De dónde salió el modelo entrenado
TEST_REGION  = "base"    # Dónde lo vamos a hacer integrar ahora

# Forzamos la clase a adaptar los límites a la zona de Test
System.set_region(TEST_REGION)

# =============================================================================
# ENUMS Y PARÁMETROS DE EJECUCIÓN
# =============================================================================
class SamplingMode(Enum):
    ALL_ZONES = "all_zones"
    ONE_ZONE_VAR_PARAMS = "one_zone_var_params"
    ONE_ZONE_FIXED_PARAMS = "one_zone_fixed_params"

N_SIMS_PER_ZONE = 10      # Cuántas configuraciones NUEVAS probar por CADA zona (Estratificado)
N_TRAJ_PER_AXIS = 10     # Cuántas condiciones iniciales por configuración simular (10x10 = 100)
GRID_DENSITY_VF = 100    # Resolución para el cálculo del campo vectorial predicho

# --- Control Temporal de Simulación ---
SIM_T_MAX = 10         
SIM_DT = 0.01            

# Nombres de Carpetas
SYSTEM_FOLDER_NAME = System.name.lower().replace("-", "_").replace(" ", "_")
TRAJECTORIES_FOLDER_NAME = "trajectories"
MODELS_FOLDER_NAME = "sindy_models"
SIMS_OUT_FOLDER_NAME = f"simulations_{TEST_REGION}" # DINÁMICO POR REGIÓN

# Recursos
MAX_WORKERS_CPU = min(8, os.cpu_count() or 1)
MAX_WORKERS_IO = 5

# Caché Temporal local
TMP_DIR = os.path.join(tempfile.gettempdir(), f"sindy_val_cache_{SYSTEM_FOLDER_NAME}_{TEST_REGION}")
os.makedirs(TMP_DIR, exist_ok=True)

# =============================================================================
# FUNCIONES DE GOOGLE DRIVE ROBUSTAS (THREAD-SAFE)
# =============================================================================
thread_local = threading.local()

def get_thread_local_service():
    if not hasattr(thread_local, "service"):
        from experiments.takens_bogdanov.data_zone_manager import _build_drive_service
        raw_service = get_drive_service()
        thread_local.service = _build_drive_service(raw_service)
    return thread_local.service

def create_or_get_subfolder(service, parent_id, folder_name):
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if not items:
        metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_id]}
        return service.files().create(body=metadata, fields='id').execute().get('id')
    return items[0]['id']

def download_json_in_memory(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return json.load(fh)

def upload_file_to_drive(local_path, file_name, folder_id):
    service = get_thread_local_service()
    for attempt in range(4):
        try:
            with open(local_path, "rb") as fd:
                media = MediaIoBaseUpload(fd, mimetype='application/octet-stream', resumable=True)
                metadata = {'name': file_name, 'parents': [folder_id]}
                service.files().create(media_body=media, body=metadata).execute()
            
            if os.path.exists(local_path):
                os.remove(local_path)
            return True, file_name
        except Exception as e:
            if attempt < 3: time.sleep(2 ** attempt)
            else: return False, str(e)

def get_existing_files(service, folder_id):
    existing = set()
    page_token = None
    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive', fields='nextPageToken, files(name)',
            pageToken=page_token, pageSize=1000
        ).execute()
        for item in results.get('files', []):
            existing.add(item['name'])
        page_token = results.get('nextPageToken', None)
        if not page_token: break
    return existing

# =============================================================================
# FÁBRICAS DE FUNCIONES JIT 
# =============================================================================
def make_sindy_ode_jit(coeffs, feature_builder_func):
    state_dim = len(System.state_names)
    coeffs_states = coeffs[:state_dim, :].astype(np.float32)
    coeffs_T = np.ascontiguousarray(coeffs_states.T)

    @jit(nopython=True)
    def sindy_ode_wrapper(t, state_arr, param_arr):
        theta = feature_builder_func(state_arr[0], state_arr[1], param_arr)
        d_state = np.dot(theta, coeffs_T)
        return d_state.astype(np.float32)

    return sindy_ode_wrapper

def make_sindy_vf_jit(sindy_ode_func):
    @jit(nopython=True, parallel=True) 
    def vf_func(x_vals, y_vals, param_arr):
        nx = len(x_vals)
        ny = len(y_vals)
        U = np.zeros((ny, nx), dtype=np.float32)
        V = np.zeros((ny, nx), dtype=np.float32)
        t = np.float32(0.0)
        
        for i in range(ny):
            for j in range(nx):
                state = np.array([x_vals[j], y_vals[i]], dtype=np.float32)
                d_state = sindy_ode_func(t, state, param_arr)
                U[i, j] = d_state[0]
                V[i, j] = d_state[1]
        return U, V
    return vf_func

# =============================================================================
# WORKERS DE PROCESAMIENTO CPU
# =============================================================================
_W_ODE = None
_W_VF = None
_W_TPARAMS = None

def worker_init(coeffs, time_params, poly_degree):
    global _W_ODE, _W_VF, _W_TPARAMS
    
    feat_builder = System.get_numba_features_func(degree=poly_degree)
    _W_ODE = make_sindy_ode_jit(coeffs, feat_builder)
    _W_VF = make_sindy_vf_jit(_W_ODE)
    _W_TPARAMS = time_params
    
    _W_ODE(0.0, np.zeros(len(System.state_names), dtype=np.float32), np.zeros(len(System.param_names), dtype=np.float32))

def run_validation_job(param_key_str):
    param_arr = parse_param_key(param_key_str)
    t_start, t_end, dt, expected_len = _W_TPARAMS
    
    x_lim = System.state_limits[0]
    y_lim = System.state_limits[1]
    x_min, x_max = np.float32(x_lim[0]), np.float32(x_lim[1])
    y_min, y_max = np.float32(y_lim[0]), np.float32(y_lim[1])

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
            
            # --- PROTECCIÓN ANTI-BLOWUP ---
            if sol.shape[1] < expected_len:
                pad_width = expected_len - sol.shape[1]
                sol = np.pad(sol, ((0, 0), (0, pad_width)), mode='constant', constant_values=np.nan)
            elif sol.shape[1] > expected_len:
                sol = sol[:, :expected_len]
                
            trajectories.append(sol)
    
    traj_array = np.stack(trajectories)

    out_path = os.path.join(TMP_DIR, f"{param_key_str}.npz")
    np.savez_compressed(out_path, trajectories=traj_array, U=U, V=V, x_vals=x_vals, y_vals=y_vals) 
    
    return param_key_str, out_path

# =============================================================================
# RECONSTRUCCIÓN DEL MODELO DESDE METADATA
# =============================================================================
def build_coefficients_matrix(metadata):
    poly_degree = metadata.get("config", {}).get("sindy_hyperparams", {}).get("poly_degree", 3)
    feature_names = System.state_names + System.param_names
    
    dummy_x = np.zeros((2, len(feature_names)))
    lib = ps.PolynomialLibrary(degree=poly_degree)
    lib.fit(dummy_x)
    raw_f_names = lib.get_feature_names(feature_names)
    
    coeffs = np.zeros((len(System.state_names), len(raw_f_names)), dtype=np.float32)
    disc_coefs = metadata.get("discovered_coefficients", {})
    
    def norm_name(n): return " ".join(sorted(n.split(" ")))
    
    for i, eq_name in enumerate(["x'", "y'"]):
        if eq_name in disc_coefs:
            for j, raw_name in enumerate(raw_f_names):
                normalized = norm_name(raw_name)
                if normalized in disc_coefs[eq_name]:
                    coeffs[i, j] = disc_coefs[eq_name][normalized]["predictions_stats"]["mean"]
                    
    return coeffs, poly_degree

def get_train_keys(metadata):
    train_keys = set()
    for log in metadata.get("iteration_logs", []):
        for key in log.get("sampling_record", {}).keys():
            train_keys.add(key)
    return train_keys

# =============================================================================
# LÓGICA DE ORQUESTACIÓN
# =============================================================================
def get_latest_run_metadata(service, models_root_id, sampling_mode, target_zone):
    batch_name = f"batch_{sampling_mode.value}" if sampling_mode == SamplingMode.ALL_ZONES else f"batch_{sampling_mode.value}_zone_{target_zone}"
    
    q_batch = f"name='{batch_name}' and '{models_root_id}' in parents and trashed=false"
    res_batch = service.files().list(q=q_batch, fields='files(id)').execute().get('files', [])
    if not res_batch: return None, None, None
    batch_id = res_batch[0]['id']
    
    q_runs = f"'{batch_id}' in parents and name contains 'run_' and trashed=false"
    runs = service.files().list(q=q_runs, fields='files(id, name)').execute().get('files', [])
    if not runs: return None, None, None
    
    latest_run = sorted(runs, key=lambda x: x['name'], reverse=True)[0]
    run_id = latest_run['id']
    
    q_json = f"name='ensemble_metadata.json' and '{run_id}' in parents and trashed=false"
    files_json = service.files().list(q=q_json, fields='files(id)').execute().get('files', [])
    if not files_json: return None, None, None
    
    metadata = download_json_in_memory(service, files_json[0]['id'])
    return metadata, run_id, latest_run['name']

def process_configuration(service, manager, models_root_id, traj_id, sampling_mode, target_zone):
    print(f"\n{'='*60}")
    print(f"Validando Configuración | Modo: {sampling_mode.value} | Zona: {target_zone}")
    print(f"{'='*60}")
    
    metadata, run_id, run_name = get_latest_run_metadata(service, models_root_id, sampling_mode, target_zone)
    
    if metadata is None:
        print("⏭️ No se encontraron entrenamientos para esta configuración. Saltando...")
        return
        
    print(f"Modelo encontrado: {run_name}")
    
    coeffs, poly_degree = build_coefficients_matrix(metadata)
    train_keys = get_train_keys(metadata)
    
    # Creamos la subcarpeta dinámica para la validación cruzada
    sims_folder_id = create_or_get_subfolder(service, run_id, SIMS_OUT_FOLDER_NAME)
    
    t_eval = np.arange(0.0, SIM_T_MAX + SIM_DT, SIM_DT, dtype=np.float32)
    # Empaquetamos la longitud esperada para la matriz de trayectorias
    time_params = (np.float32(0.0), np.float32(SIM_T_MAX), np.float32(SIM_DT), int(len(t_eval)))

    existing_sims = get_existing_files(service, sims_folder_id)
    if "t_eval.npz" not in existing_sims:
        t_eval_path = os.path.join(TMP_DIR, "t_eval.npz")
        np.savez_compressed(t_eval_path, t_eval=t_eval)
        with open(t_eval_path, "rb") as fd:
            media = MediaIoBaseUpload(fd, mimetype='application/octet-stream', resumable=True)
            service.files().create(media_body=media, body={'name': 't_eval.npz', 'parents': [sims_folder_id]}).execute()
        os.remove(t_eval_path)

    print(f"\nCalculando Muestreo Estratificado en Región '{TEST_REGION}'...")
    jobs_keys = []
    for zone_id, zone_desc in System.zone_names.items():
        candidates = [k for k in manager.zone_map.get(zone_id, []) if k not in train_keys]
        actual_sample_size = min(N_SIMS_PER_ZONE, len(candidates))
        if actual_sample_size > 0:
            selected = random.sample(candidates, actual_sample_size)
            jobs_keys.extend(selected)
            print(f"  > Zona {zone_id}: Seleccionadas {actual_sample_size} configs de prueba.")

    jobs_keys = [k for k in jobs_keys if f"{k}.npz" not in existing_sims]

    if not jobs_keys:
        print(f"✅ Todas las simulaciones requeridas ya están listas en '{SIMS_OUT_FOLDER_NAME}'.")
        return

    print(f"\nSimulando {len(jobs_keys)} trayectorias de prueba (T={SIM_T_MAX}s, dt={SIM_DT}s)...")
    
    archivos_generados = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS_CPU, initializer=worker_init, initargs=(coeffs, time_params, poly_degree)) as cpu_pool:
        futures = {cpu_pool.submit(run_validation_job, k): k for k in jobs_keys}
        for f in tqdm(as_completed(futures), total=len(jobs_keys), desc="Fase 1/2: Integración Matemática (CPU)"):
            try:
                key_str, npz_path = f.result()
                archivos_generados.append((key_str, npz_path))
            except Exception as e:
                print(f"\n[Error CPU] Clave {futures[f]}: {e}")

    if archivos_generados:
        print("\nIniciando respaldo en la nube...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_IO) as io_pool:
            upload_futures = {
                io_pool.submit(upload_file_to_drive, path, f"{key}.npz", sims_folder_id): key 
                for key, path in archivos_generados
            }
            for f in tqdm(as_completed(upload_futures), total=len(archivos_generados), desc="Fase 2/2: Subida a Drive (Red)"):
                try:
                    success, msg = f.result()
                    if not success: print(f"\n[Error Red] No se pudo subir {upload_futures[f]}: {msg}")
                except Exception as e:
                    pass

    print(f"✅ Validación completada para el modelo {run_name}!")

def main():
    print(f"--- Validación Cruzada Cloud SINDy ---")
    print(f"📥 Train Region: {TRAIN_REGION} | 📈 Test Region: {TEST_REGION}")
    
    RUN_ALL_CONFIGS = True 
    
    try:
        from experiments.takens_bogdanov.data_zone_manager import _build_drive_service
        raw_service = get_drive_service()
        service = _build_drive_service(raw_service)
        base_id = get_target_folder_id()
        
        sys_id = create_or_get_subfolder(service, base_id, SYSTEM_FOLDER_NAME)
        
        # --- División de Rutas: Entrenamos en una, Validamos en otra ---
        train_region_id = create_or_get_subfolder(service, sys_id, TRAIN_REGION)
        test_region_id = create_or_get_subfolder(service, sys_id, TEST_REGION)
        
        models_root_id = create_or_get_subfolder(service, train_region_id, MODELS_FOLDER_NAME)
        traj_test_id = create_or_get_subfolder(service, test_region_id, TRAJECTORIES_FOLDER_NAME)
        
        print(f"Inicializando Gestor de Zonas Dinámicas para la región de TEST '{TEST_REGION}'...")
        manager = DataZoneManager(system_class=System, service=service, target_folder_id=traj_test_id, region=TEST_REGION)
        
    except Exception as e:
        print(f"Error crítico accediendo a Drive o Metadata: {e}")
        return

    if RUN_ALL_CONFIGS:
        print("\n" + "="*60)
        print("MODO AUTOMÁTICO: Validando las 11 configuraciones")
        print("="*60)
        
        configs_to_run = [(SamplingMode.ALL_ZONES, "Todas")]
        for z in range(1, 6):
            configs_to_run.append((SamplingMode.ONE_ZONE_VAR_PARAMS, z))
        for z in range(1, 6):
            configs_to_run.append((SamplingMode.ONE_ZONE_FIXED_PARAMS, z))
            
        for mode, zone in configs_to_run:
            process_configuration(service, manager, models_root_id, traj_test_id, mode, zone)
            
    else:
        target_mode = SamplingMode.ALL_ZONES
        target_zone = "Todas"
        process_configuration(service, manager, models_root_id, traj_test_id, target_mode, target_zone)

    print("\nLimpiando caché temporal global...")
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()