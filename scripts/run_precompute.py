#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script GENÉRICO de integración masiva de trayectorias.
Utiliza la arquitectura modular (System + Core).

Flujo:
1. Carga la definición del Sistema (TakensBogdanov).
2. Genera una grilla N-dimensional de parámetros.
3. Calcula en paralelo (ProcessPool) usando el integrador genérico RK4.
4. Guarda en HDF5 en segundo plano (ThreadPool).
"""

import os
import sys
import json
import numpy as np
import h5py
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
# =============================================================================
# CONFIGURACIÓN DE RUTAS (FIX PARA CARPETA SCRIPTS)
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)
# --- IMPORTACIONES DEL NÚCLEO MODULAR ---
from systems.takens_bogdanov import TakensBogdanov as System  # <--- AQUÍ CAMBIAS EL SISTEMA
from core.integrators import rk4_general
from core.io import make_param_key, save_results_to_hdf5, load_npz_to_dict
from core.utils import generate_param_grid

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
HDF5_FILE = os.path.join(OUTPUT_DIR, "trajectory_data.hdf5")
JSON_FILE = os.path.join(OUTPUT_DIR, "grid_metadata.json")
TMP_DIR = os.path.join(OUTPUT_DIR, "tmp_npz")

# --- Configuración de Simulación ---
# Densidad de la grilla de parámetros (puntos por dimensión)
GRID_DENSITY = 20 

# Trayectorias: grilla de condiciones iniciales (N x N)
N_TRAJ_PER_AXIS = 10

# Tiempo
T_SPAN = [0.0, 5.0]
N_STEPS = 500
DT = np.float32((T_SPAN[1] - T_SPAN[0]) / (N_STEPS - 1))
T_EVAL = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS, dtype=np.float32)

# Resolución del Campo Vectorial (para visualización)
VF_RESOLUTION = 100

# Recursos
MAX_WORKERS_CPU = min(4, os.cpu_count() or 1)
MAX_WORKERS_IO = 4
MAX_RETRIES = 3

# Crear carpetas necesarias
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# =============================================================================
# WORKER: TAREA DE CÓMPUTO (CPU)
# =============================================================================
def run_simulation_job(args):
    """
    Ejecuta la simulación para UN conjunto de parámetros (param_arr).
    """
    param_arr = args
    
    # 1. Obtener funciones JIT
    ode_func = System.get_ode_jit()
    vf_func = System.get_vector_field_jit()
    
    # 2. Calcular Puntos Fijos
    fixed_points = None
    if hasattr(System, 'calculate_fixed_points'):
        fixed_points = System.calculate_fixed_points(param_arr)
    
    # 3. Definir límites
    x_lim = list(System.state_limits[0])
    y_lim = list(System.state_limits[1])
    
    if fixed_points is not None and fixed_points.size > 0:
        fp_x = fixed_points[:, 0] 
        x_lim[0] = min(x_lim[0], np.min(fp_x) - 1.0)
        x_lim[1] = max(x_lim[1], np.max(fp_x) + 1.0)
        y_lim[0] = min(y_lim[0], x_lim[0])
        y_lim[1] = max(y_lim[1], x_lim[1])

    x_min, x_max = np.float32(x_lim[0]), np.float32(x_lim[1])
    y_min, y_max = np.float32(y_lim[0]), np.float32(y_lim[1])

    # 4. Calcular Campo Vectorial
    x_vals = np.linspace(x_min, x_max, VF_RESOLUTION, dtype=np.float32)
    y_vals = np.linspace(y_min, y_max, VF_RESOLUTION, dtype=np.float32)
    U, V = vf_func(x_vals, y_vals, param_arr)
    
    # 5. Integrar Trayectorias
    tx = np.linspace(x_min, x_max, N_TRAJ_PER_AXIS, dtype=np.float32)
    ty = np.linspace(y_min, y_max, N_TRAJ_PER_AXIS, dtype=np.float32)
    
    trajectories = []
    t_start = np.float32(T_SPAN[0])
    t_end = np.float32(T_SPAN[1])
    
    for y0_val in ty:
        for x0_val in tx:
            y0 = np.array([x0_val, y0_val], dtype=np.float32)
            sol = rk4_general(ode_func, y0, t_start, t_end, DT, param_arr)
            trajectories.append(sol)
            
    traj_array = np.stack(trajectories)

    # 6. Guardar temporal
    key = make_param_key(param_arr)
    out_path = os.path.join(TMP_DIR, f"{key}.npz")
    
    data_to_save = {
        "trajectories": traj_array,
        "fixed_points": fixed_points,
        "U": U, "V": V, "x_vals": x_vals, "y_vals": y_vals
    }
    np.savez_compressed(out_path, **data_to_save)
    
    return key, out_path

# =============================================================================
# WORKER: TAREA DE I/O (DISCO)
# =============================================================================
def run_io_job(hf_handle, key, npz_path, pbar):
    try:
        data_dict = load_npz_to_dict(npz_path)
        save_results_to_hdf5(hf_handle, key, data_dict)
    except Exception as e:
        print(f"[Error I/O] Clave {key}: {e}")
    finally:
        if os.path.exists(npz_path):
            os.remove(npz_path)
        pbar.update(1)

# =============================================================================
# MAIN CON LÓGICA DE REINTENTOS
# =============================================================================
def main():
    print(f"--- Iniciando Precompute Modular ---")
    # Corrección del print: Usamos len() en lugar de acceder a la propiedad
    print(f"Sistema: {System.name}")
    print(f"Dimensiones: Estado={len(System.state_names)}, Parámetros={len(System.param_names)}")
    
    # 1. Generar Grilla
    print("Generando grilla de parámetros...")
    all_params = generate_param_grid(System.param_ranges, GRID_DENSITY)
    print(f"Total de configuraciones posibles: {len(all_params)}")

    # 2. Preparar HDF5
    with h5py.File(HDF5_FILE, "a") as hf:
        if "t_eval" not in hf:
            hf.create_dataset("t_eval", data=T_EVAL, dtype="f4")
        
        existing_keys = set(hf.keys()) - {"t_eval"}
        
        # Filtrar trabajos iniciales
        jobs_to_run = []
        for p in all_params:
            key = make_param_key(p)
            if key not in existing_keys:
                jobs_to_run.append(p)
        
        print(f"Ya procesados: {len(existing_keys)}")
        print(f"Pendientes iniciales: {len(jobs_to_run)}")
        
        # Limpieza inicial
        pending_npz = [f for f in os.listdir(TMP_DIR) if f.endswith(".npz")]
        if pending_npz:
            print(f"Limpiando {len(pending_npz)} archivos temporales previos...")
            for f in pending_npz:
                try: os.remove(os.path.join(TMP_DIR, f))
                except: pass

        if not jobs_to_run:
            print("Todos los trabajos están completos.")
            return

        # 3. BUCLE DE EJECUCIÓN Y REINTENTOS
        # Iteramos 'MAX_RETRIES + 1' veces (Intento 0, Intento 1, ...)
        for attempt in range(MAX_RETRIES + 1):
            if not jobs_to_run:
                print("\n¡Todos los trabajos se completaron exitosamente!")
                break
            
            print(f"\n--- Ejecución: Intento {attempt + 1} de {MAX_RETRIES + 1} ---")
            print(f"Procesando {len(jobs_to_run)} trabajos...")
            
            failed_params = [] # Aquí guardaremos los que fallen para la próxima vuelta

            pbar = tqdm(total=len(jobs_to_run), desc=f"Intento {attempt + 1}")
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS_IO) as io_pool:
                with ProcessPoolExecutor(max_workers=MAX_WORKERS_CPU) as cpu_pool:
                    
                    # Mapeamos el Future -> Parametros Originales (para poder reintentar)
                    future_to_params = {
                        cpu_pool.submit(run_simulation_job, p): p 
                        for p in jobs_to_run
                    }
                    
                    for future in as_completed(future_to_params):
                        # Recuperamos los parámetros asociados a este futuro
                        params_of_job = future_to_params[future]
                        key_str = make_param_key(params_of_job)
                        
                        try:
                            # Intentamos obtener el resultado
                            _, npz_path = future.result()
                            
                            # Si éxito: Enviar a I/O
                            io_pool.submit(run_io_job, hf, key_str, npz_path, pbar)
                            
                        except Exception as e:
                            # Si falla (ej: worker vanished), guardamos para reintentar
                            error_msg = str(e)
                            print(f"[Fallo] Clave {key_str}: {error_msg[:50]}...") # Print corto
                            failed_params.append(params_of_job)
                            pbar.update(1) # Contamos como "procesado" (aunque fallido)
            
            pbar.close()
            
            # Actualizamos la lista de trabajos para la siguiente iteración
            jobs_to_run = failed_params
            
            if jobs_to_run:
                print(f"⚠️ {len(jobs_to_run)} trabajos fallaron en este intento.")
            else:
                print("✅ Lote completado sin errores.")

    # 4. Generar Metadatos
    print("\nActualizando metadatos JSON...")
    metadata = {}
    with h5py.File(HDF5_FILE, "r") as hf:
        for key in hf.keys():
            if key == "t_eval": continue
            n_fp = 0
            if "fixed_points" in hf[key]:
                n_fp = hf[key]["fixed_points"].shape[0]
            metadata[key] = {"num_fixed_points": int(n_fp)}
            
    with open(JSON_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n¡Proceso Finalizado! Datos en: {OUTPUT_DIR}")

if __name__ == "__main__":
    # Warm-up
    print("Pre-compilando JIT...")
    try:
        ode = System.get_ode_jit()
        dummy_state = np.array([0.0, 0.0], dtype=np.float32)
        dummy_param = np.array([0.1, 0.1], dtype=np.float32)
        ode(0.0, dummy_state, dummy_param)
        print("JIT listo.")
    except Exception as e:
        print(f"Advertencia en Warm-up JIT: {e}")
        
    main()