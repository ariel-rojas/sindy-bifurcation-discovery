#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script GENÉRICO de validación del modelo SINDy.
Simula trayectorias nuevas usando las ecuaciones DESCUBIERTAS.

ARQUITECTURA ROBUSTA:
- Usa un 'worker_init' para compilar la ODE de SINDy dentro de cada proceso hijo.
- Evita errores de serialización (Pickle) al no pasar funciones JIT por la cola.
"""

import os
import sys
import h5py
import numpy as np
import joblib
import json
import random
from tqdm import tqdm
from numba import jit
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
# =============================================================================
# CONFIGURACIÓN DE RUTAS (FIX PARA CARPETA SCRIPTS)
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)
# --- IMPORTACIONES DEL NÚCLEO MODULAR ---
from systems.takens_bogdanov import TakensBogdanov as System
from core.integrators import rk4_general
from core.io import parse_param_key, save_results_to_hdf5, load_npz_to_dict

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
CARPETA_FILES = os.path.join(PROJECT_ROOT, "output/5/20")
MODEL_PATH = os.path.join(CARPETA_FILES, "sindy_model.joblib")
TRAINING_PARAMS_PATH = os.path.join(CARPETA_FILES, "sindy_training_params.json")
HDF5_GT = os.path.join(os.path.join(PROJECT_ROOT, "output"), "trajectory_data.hdf5")

# Salida
HDF5_SINDY = os.path.join(CARPETA_FILES, "sindy_simulations.hdf5")
TMP_DIR = os.path.join(CARPETA_FILES, "tmp_sindy_sims")

# Configuración de Validación
N_RANDOM_SIMS = 50   # Cuántas configuraciones probar
GRID_DENSITY_VF = 100 # Resolución para campo vectorial
N_TRAJ_PER_AXIS = 10  # Trayectorias por eje

# Recursos
MAX_WORKERS_CPU = min(8, os.cpu_count() or 1)
MAX_WORKERS_IO = 4

os.makedirs(TMP_DIR, exist_ok=True)

# =============================================================================
# FÁBRICAS DE FUNCIONES JIT (DINÁMICAS)
# =============================================================================
def make_sindy_ode_jit(coeffs, feature_builder_func):
    """
    Crea una función ODE compilada que usa los coeficientes aprendidos.
    """
    # Obtenemos dimensiones estáticas del sistema
    state_dim = len(System.state_names)
    
    # Recortamos coeficientes: SINDy devuelve (N+M) filas, solo nos importan las N primeras (estados)
    # coeffs shape original: (state_dim + param_dim, n_features)
    coeffs_states = coeffs[:state_dim, :].astype(np.float32)
    
    # Transpuesta para producto punto rápido: (n_features, state_dim)
    coeffs_T = np.ascontiguousarray(coeffs_states.T)

    @jit(nopython=True, cache=True)
    def sindy_ode_wrapper(t, state_arr, param_arr):
        # 1. Construir vector theta (feature library)
        # Asumimos que el builder del sistema acepta (x, y, params)
        # Nota: Si el sistema fuera N-dimensional puro, habría que adaptar feature_builder_func
        theta = feature_builder_func(state_arr[0], state_arr[1], param_arr)
        
        # 2. Calcular derivadas: dX = theta . Coeffs^T
        # theta: (n_feat,), Coeffs^T: (n_feat, n_states) -> (n_states,)
        d_state = np.dot(theta, coeffs_T)
        
        return d_state.astype(np.float32)

    return sindy_ode_wrapper

def make_sindy_vf_jit(sindy_ode_func):
    """Crea el calculador de campo vectorial usando la ODE de SINDy."""
    
    @jit(nopython=True, parallel=True, cache=True)
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
# VARIABLES GLOBALES DEL WORKER
# (Se inicializan dentro de cada proceso hijo)
# =============================================================================
_W_ODE = None
_W_VF = None
_W_TPARAMS = None # (t_start, t_end, dt)

def worker_init(coeffs, time_params):
    """
    Inicializador del proceso Worker.
    Compila las funciones JIT localmente para evitar errores de serialización.
    """
    global _W_ODE, _W_VF, _W_TPARAMS
    
    # 1. Obtener el constructor de features del sistema
    feat_builder = System.get_numba_features_func(degree=3)
    
    # 2. Fabricar funciones locales con los coeficientes recibidos
    _W_ODE = make_sindy_ode_jit(coeffs, feat_builder)
    _W_VF = make_sindy_vf_jit(_W_ODE)
    _W_TPARAMS = time_params
    
    # 3. Warm-up (Compilación JIT inicial)
    dummy_s = np.zeros(len(System.state_names), dtype=np.float32)
    dummy_p = np.zeros(len(System.param_names), dtype=np.float32)
    _W_ODE(0.0, dummy_s, dummy_p)

# =============================================================================
# TAREAS DE SIMULACIÓN E I/O
# =============================================================================

def run_validation_job(param_key_str):
    """
    Simula trayectorias usando las variables globales (_W_ODE, _W_VF).
    """
    param_arr = parse_param_key(param_key_str)
    t_start, t_end, dt = _W_TPARAMS
    
    # 1. Límites (Usamos los del sistema base)
    x_lim = System.state_limits[0]
    y_lim = System.state_limits[1]
    x_min, x_max = np.float32(x_lim[0]), np.float32(x_lim[1])
    y_min, y_max = np.float32(y_lim[0]), np.float32(y_lim[1])

    # 2. Campo Vectorial (SINDy)
    x_vals = np.linspace(x_min, x_max, GRID_DENSITY_VF, dtype=np.float32)
    y_vals = np.linspace(y_min, y_max, GRID_DENSITY_VF, dtype=np.float32)
    U, V = _W_VF(x_vals, y_vals, param_arr)

    # 3. Integrar Trayectorias
    tx = np.linspace(x_min, x_max, N_TRAJ_PER_AXIS, dtype=np.float32)
    ty = np.linspace(y_min, y_max, N_TRAJ_PER_AXIS, dtype=np.float32)
    
    trajectories = []
    for y0_val in ty:
        for x0_val in tx:
            y0 = np.array([x0_val, y0_val], dtype=np.float32)
            # Usamos el integrador genérico con la ODE local SINDy
            sol = rk4_general(_W_ODE, y0, t_start, t_end, dt, param_arr)
            trajectories.append(sol)
    
    traj_array = np.stack(trajectories)

    # 4. Guardar temporal
    out_path = os.path.join(TMP_DIR, f"{param_key_str}.npz")
    
    # Corrección: Guardar "plano" para que core/io.py lo detecte
    np.savez_compressed(out_path, 
                        trajectories=traj_array, 
                        U=U, V=V, x_vals=x_vals, y_vals=y_vals) 
    
    return param_key_str, out_path

def run_io_job(hf_handle, key, npz_path, pbar):
    try:
        data = load_npz_to_dict(npz_path)
        save_results_to_hdf5(hf_handle, key, data)
    except Exception as e:
        print(f"[IO Error] Clave {key}: {e}")
    finally:
        if os.path.exists(npz_path): os.remove(npz_path)
        pbar.update(1)

# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"--- Validación SINDy Modular ({System.name}) ---")
    
    # 1. Cargar Modelo y Coeficientes
    if not os.path.exists(MODEL_PATH):
        print("Error: No existe el modelo .joblib.")
        return
    
    print("Cargando modelo SINDy...")
    model = joblib.load(MODEL_PATH)
    coeffs = model.optimizer.coef_ # Shape (n_vars, n_features)
    
    # Validación rápida de forma
    n_states = len(System.state_names)
    if coeffs.shape[0] < n_states:
        raise ValueError(f"El modelo tiene menos filas ({coeffs.shape[0]}) que estados ({n_states}).")

    # 2. Obtener parámetros de tiempo del Ground Truth
    with h5py.File(HDF5_GT, "r") as hf:
        t_eval = hf["t_eval"][:]
        all_gt_keys = set(hf.keys()) - {"t_eval"}
        
    t_start = np.float32(t_eval[0])
    t_end = np.float32(t_eval[-1])
    dt = np.float32(t_eval[1] - t_eval[0])
    time_params = (t_start, t_end, dt)

    # 3. Seleccionar Claves de Test (Excluir entrenamiento)
    train_keys = set()
    if os.path.exists(TRAINING_PARAMS_PATH):
        with open(TRAINING_PARAMS_PATH, "r") as f:
            d = json.load(f)
            train_keys = set(d.get("keys_used", []))
    
    test_candidates = list(all_gt_keys - train_keys)
    if not test_candidates:
        print("No quedan claves para validar (todas se usaron en training).")
        return

    n_sample = min(N_RANDOM_SIMS, len(test_candidates))
    print(f"Total GT: {len(all_gt_keys)}. Training: {len(train_keys)}.")
    print(f"Simulando {n_sample} configuraciones de prueba...")
    
    jobs_keys = random.sample(test_candidates, n_sample)

    # 4. Ejecución con Inicializador de Workers
    # Usamos 'w' para limpiar validaciones anteriores y empezar limpio
    with h5py.File(HDF5_SINDY, "w") as hf_out:
        hf_out.create_dataset("t_eval", data=t_eval)
        
        pbar = tqdm(total=len(jobs_keys), desc="Simulando")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_IO) as io_pool:
            # INICIALIZADOR CLAVE: Pasa coeffs a cada worker al nacer
            with ProcessPoolExecutor(max_workers=MAX_WORKERS_CPU, 
                                     initializer=worker_init, 
                                     initargs=(coeffs, time_params)) as cpu_pool:
                
                future_to_key = {}
                for k in jobs_keys:
                    # Ya no pasamos coeffs ni time_params, el worker ya los tiene
                    f = cpu_pool.submit(run_validation_job, k)
                    future_to_key[f] = k
                
                for future in as_completed(future_to_key):
                    k = future_to_key[future]
                    try:
                        _, npz_path = future.result()
                        io_pool.submit(run_io_job, hf_out, k, npz_path, pbar)
                    except Exception as e:
                        print(f"[CPU Error] Clave {k}: {e}")
                        pbar.update(1)
        pbar.close()

    print(f"Validación completada. Resultados en {HDF5_SINDY}")

if __name__ == "__main__":
    main()