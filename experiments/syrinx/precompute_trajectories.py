#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Syrinx Precompute Engine
---------------------------------------------
Motor de simulación masiva optimizado para Windows y Nube.
- Resiliencia: Manejo de bloqueos de archivo (WinError 32) y Timeouts (10060).
- Paralelismo: Capa de cálculo (CPU) + Capa de subida (Threads).
- Integridad: Auto-recorte de trayectorias inestables.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import threading
import numpy as np
from dataclasses import asdict
from tqdm import tqdm
from numba import jit
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from googleapiclient.http import MediaFileUpload

# =============================================================================
# RUTAS E IMPORTACIONES LOCALES
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from experiments.syrinx.config import EXPERIMENT
from systems.syrinx import SyrinxModel as System
from core.io import make_param_key
from core.utils import generate_param_grid
from core.drive_auth import get_drive_service, get_target_folder_id

# =============================================================================
# CONFIGURACIÓN TÉCNICA
# =============================================================================
SYSTEM_FOLDER_NAME = System.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
TRAJECTORIES_FOLDER = "trajectories"
TMP_DIR = os.path.join(tempfile.gettempdir(), f"syrinx_bastion_{SYSTEM_FOLDER_NAME}")

T_SPAN = EXPERIMENT.numeric.t_span_physical
N_STEPS = EXPERIMENT.numeric.n_steps
T_EVAL = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS, dtype=np.float32)
DT = np.float32((T_SPAN[1] - T_SPAN[0]) / (N_STEPS - 1))

# Seguridad Biomecánica
SAFE_DENOM_MARGIN = np.float32(0.001) 
MAX_VEL_JUMP = np.float32(2.5) 
MAX_RAD = np.float32(EXPERIMENT.numeric.max_radius_physical)
MAX_VEL = np.float32(EXPERIMENT.numeric.max_velocity_physical)

# Multiprocesamiento (Ajustado para no saturar I/O en Windows)
MAX_WORKERS_CPU = min(4, os.cpu_count() or 1)
MAX_WORKERS_IO = 4  
os.makedirs(TMP_DIR, exist_ok=True)

# =============================================================================
# NÚCLEO DE CÁLCULO JIT
# =============================================================================

@jit(nopython=True, cache=True)
def simulate_grid_rk4_clean_jit(ode_func, ics, t_start, n_steps, dt, param_arr, 
                                 max_rad, max_vel, safe_margin, max_jump):
    n_traj = ics.shape[0]
    trajs = np.zeros((n_traj, 2, n_steps), dtype=np.float32)
    valid_steps = np.zeros(n_traj, dtype=np.int32)
    
    f05, f20, f60 = np.float32(0.5), np.float32(2.0), np.float32(6.0)
    a01, tau = np.float32(param_arr[9]), np.float32(param_arr[11])

    for idx in range(n_traj):
        y = ics[idx].astype(np.float32).copy()
        trajs[idx, :, 0] = y
        t = np.float32(t_start)
        count = 1
        for step in range(1, n_steps):
            if (a01 + y[0] + tau * y[1]) < safe_margin: break
            k1 = ode_func(t, y, param_arr)
            k2 = ode_func(t + f05*dt, y + f05*dt*k1, param_arr)
            k3 = ode_func(t + f05*dt, y + f05*dt*k2, param_arr)
            k4 = ode_func(t + dt, y + dt*k3, param_arr)
            y_next = y + (dt / f60) * (k1 + f20*k2 + f20*k3 + k4)
            
            if abs(y_next[1] - y[1]) > max_jump: break
            if abs(y_next[1]) > max_vel or not np.isfinite(y_next[0]): break
            if (y_next[0]**2 + y_next[1]**2) > max_rad**2: break

            y, trajs[idx, :, step] = y_next, y_next
            t, count = t + dt, count + 1
            
        valid_steps[idx] = count
    return trajs, valid_steps

# =============================================================================
# WORKER DE SIMULACIÓN
# =============================================================================

def run_simulation_job(task_data):
    swept_vals, param_arr = task_data
    ode_f, vf_f = System.get_ode_jit(), System.get_vector_field_jit()
    
    # Calculate fixed points for this specific parameter set
    fixed_points = System.calculate_fixed_points(param_arr)
    
    x_lim, y_lim = EXPERIMENT.phase_space.x_lim_physical, EXPERIMENT.phase_space.y_lim_physical
    res = EXPERIMENT.numeric.vf_resolution
    x_grid = np.linspace(x_lim[0], x_lim[1], res, dtype=np.float32)
    y_grid = np.linspace(y_lim[0], y_lim[1], res, dtype=np.float32)
    U, V = vf_f(x_grid, y_grid, param_arr)

    # 1. Base Global Grid
    n_ax = EXPERIMENT.phase_space.n_traj_per_axis
    gx, gy = np.meshgrid(np.linspace(x_lim[0], x_lim[1], n_ax), 
                         np.linspace(y_lim[0], y_lim[1], n_ax))
    ics_list = [np.column_stack((gx.ravel(), gy.ravel()))]

    # 2. Localized Grids & Lines around Fixed Points
    local_nx = EXPERIMENT.phase_space.local_ic_points_per_axis
    local_dx = EXPERIMENT.phase_space.local_ic_delta_x
    local_dy = EXPERIMENT.phase_space.local_ic_delta_y
    line_n = EXPERIMENT.phase_space.line_ic_points

    for fp in fixed_points:
        x_star, y_star = fp[0], fp[1]

        # A. Local Rectangular Grid
        if local_nx > 1:
            lx = np.linspace(x_star - local_dx, x_star + local_dx, local_nx)
            ly = np.linspace(y_star - local_dy, y_star + local_dy, local_nx)
            lgx, lgy = np.meshgrid(lx, ly)
            ics_list.append(np.column_stack((lgx.ravel(), lgy.ravel())))

        # B. Local Crossed Lines (to intersect potential diagonal manifolds)
        if line_n > 1:
            diag_x = np.linspace(x_star - local_dx, x_star + local_dx, line_n)
            diag_y1 = np.linspace(y_star - local_dy, y_star + local_dy, line_n)
            diag_y2 = np.linspace(y_star + local_dy, y_star - local_dy, line_n)
            ics_list.append(np.column_stack((diag_x, diag_y1)))
            ics_list.append(np.column_stack((diag_x, diag_y2)))

    # Consolidate all Initial Conditions into a single array
    ics = np.vstack(ics_list).astype(np.float32)

    # Execute JIT compiled integration
    raw_trajs, valids = simulate_grid_rk4_clean_jit(
        ode_f, ics, T_SPAN[0], N_STEPS, DT, param_arr, 
        MAX_RAD, MAX_VEL, SAFE_DENOM_MARGIN, MAX_VEL_JUMP
    )

    # Filter and format successful trajectories
    processed = [raw_trajs[k, :, :valids[k]] for k in range(ics.shape[0]) if valids[k] > 10]
    trajectories_obj = np.empty(len(processed), dtype=object)
    for i, t in enumerate(processed): 
        trajectories_obj[i] = t
    
    # Save to disk
    key = make_param_key(swept_vals)
    path = os.path.join(TMP_DIR, f"{key}.npz")
    np.savez_compressed(
        path, 
        trajectories=trajectories_obj, 
        fixed_points=fixed_points, 
        U=U, V=V, 
        x_vals=x_grid, y_vals=y_grid, 
        swept_vals=swept_vals, 
        param_arr_full=param_arr
    )
    return key, path, len(fixed_points)

# =============================================================================
# GESTIÓN DE DRIVE CON REINTENTOS (Bastion Logic)
# =============================================================================

thread_local = threading.local()

def get_service():
    if not hasattr(thread_local, "service"): thread_local.service = get_drive_service()
    return thread_local.service

def upload_to_drive_robust(file_path, name, folder_id):
    session = get_service()

    url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"

    metadata = {
        "name": name,
        "parents": [folder_id]
    }

    for attempt in range(5):
        try:
            with open(file_path, "rb") as f:
                files = {
                    "metadata": ("metadata", json.dumps(metadata), "application/json"),
                    "file": (name, f, "application/octet-stream")
                }

                r = session.post(url, files=files, timeout=60)

            if r.status_code in [200, 201]:
                # borrar archivo
                for _ in range(5):
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        break
                    except PermissionError:
                        time.sleep(0.5)

                return True, name, ""

            time.sleep(1.5 * (attempt + 1))

        except Exception as e:
            time.sleep(1.5 * (attempt + 1))
            if attempt == 4:
                return False, name, str(e)

    return False, name, "Max retries reached"

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"--- Simulando trayectorias---")
    
    try:
        service = get_drive_service()
        base_id = get_target_folder_id()
        sys_id = create_folder_drive(service, base_id, SYSTEM_FOLDER_NAME)
        traj_id = create_folder_drive(service, sys_id, TRAJECTORIES_FOLDER)
        existing_drive = get_drive_files_set(service, traj_id)
        print(f"Drive: {len(existing_drive)} archivos detectados.")
    except Exception as e:
        print(f"Error de conexión inicial: {e}"); return

    # Maestros
    t_eval_path = os.path.join(TMP_DIR, "t_eval.npz")
    np.savez_compressed(t_eval_path, t_eval=T_EVAL)
    upload_to_drive_robust(t_eval_path, "t_eval.npz", traj_id)

    # Tareas
    grid = generate_param_grid(EXPERIMENT.sweep.physical_ranges, EXPERIMENT.sweep.grid_density)
    tasks = []
    base_phys = asdict(EXPERIMENT.physical_params)
    
    for p in grid:
        kv, pv = float(p[0]), float(p[1])
        key = make_param_key([kv, pv])
        if f"{key}.npz" not in existing_drive:
            p_m = base_phys.copy(); p_m.update({"kappa1": kv, "psub": pv})
            tasks.append(([kv, pv], System.build_full_param_arr(**p_m)))

    if not tasks:
        print("✅ Todo actualizado."); return

    metadata = {}
    print(f"Procesando {len(tasks)} puntos en {MAX_WORKERS_CPU} núcleos...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS_CPU) as cpu_pool:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_IO) as io_pool:
            with tqdm(total=len(tasks), desc="Avance") as pbar:
                
                # Procesamos por lotes para evitar saturar el disco de archivos temporales
                batch_size = EXPERIMENT.batch_size
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i : i + batch_size]
                    futures_cpu = {cpu_pool.submit(run_simulation_job, t): t for t in batch}
                    futures_io = []
                    
                    # Recolectar resultados CPU y lanzarlos a I/O
                    for f_cpu in as_completed(futures_cpu):
                        try:
                            key, path, n_fp = f_cpu.result()
                            # El delay de 0.1s ayuda a Windows a soltar el handle del archivo
                            time.sleep(0.1) 
                            f_io = io_pool.submit(upload_to_drive_robust, path, f"{key}.npz", traj_id)
                            futures_io.append((f_io, key, n_fp, futures_cpu[f_cpu][0]))
                        except Exception as e:
                            print(f"\n[Error CPU] {e}")
                            pbar.update(1)
                    
                    # Esperar subidas del lote
                    for f_io, key, n_fp, swept in futures_io:
                        success, name, err = f_io.result()
                        if success:
                            metadata[key] = {"num_fixed_points": int(n_fp), "swept_vals": [float(v) for v in swept]}
                        else:
                            print(f"\n[Error Drive] {name}: {err}")
                        pbar.update(1)

    # Metadata Final
    meta_name = f"grid_metadata_{SYSTEM_FOLDER_NAME}.json"
    meta_local = os.path.join(TMP_DIR, meta_name)
    
    # Extraer los rangos configurados para el barrido
    sweep_ranges = EXPERIMENT.sweep.physical_ranges
    
    final_meta = {
        "_system_info": {
            "system_name": System.name, 
            "physical_base_params": base_phys,
            "sweep_ranges": sweep_ranges  
        }
    }
    final_meta.update(metadata)
    
    with open(meta_local, "w") as f: json.dump(final_meta, f, indent=2)
    upload_to_drive_robust(meta_local, meta_name, traj_id)
    print("\n--- Bastion: Proceso Finalizado con Éxito ---")

# --- Helpers ---
def create_folder_drive(session, parent_id, name):
    print(f"Creando carpeta '{name}' en Drive...")
    url = "https://www.googleapis.com/drive/v3/files"

    query = f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"

    r = session.get(url, params={"q": query})
    files = r.json().get("files", [])

    if files:
        return files[0]["id"]

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id]
    }

    r = session.post(url, json=metadata)
    return r.json()["id"]

def get_drive_files_set(session, folder_id):
    print("Obteniendo archivos de Drive...")
    files = set()
    page_token = None

    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken, files(name)",
            "pageSize": 1000
        }

        if page_token:
            params["pageToken"] = page_token

        r = session.get("https://www.googleapis.com/drive/v3/files", params=params)
        data = r.json()

        for f in data.get("files", []):
            files.add(f["name"])

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return files

if __name__ == "__main__":
    main()