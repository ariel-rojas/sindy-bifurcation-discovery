#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script GENÉRICO de descubrimiento de ecuaciones (SINDy).
Utiliza la arquitectura modular (System + Core).

MEJORAS:
- Gestión automática de carpetas de salida: output/{BATCH_SIZE}/{VERSION_ID}/
- Log centralizado en output/{BATCH_SIZE}/versions_log.json
- Soporte para recorte temporal (TRAIN_T_MAX).
"""

import os
import h5py
import numpy as np
import pysindy as ps
from tqdm import tqdm
import random
import joblib
import json
import sys
from datetime import datetime

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

# --- IMPORTACIONES MODULARES ---
from systems.takens_bogdanov import TakensBogdanov as System
from core.io import parse_param_key

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
# La carpeta de salida se calcula dinámicamente en main()
BASE_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
HDF5_FILE = os.path.join(BASE_OUTPUT_ROOT, "trajectory_data.hdf5")

# --- Configuración de Entrenamiento ---
TRAIN_T_MAX = 20.0  # None para usar todo, float para recortar

# --- Configuración de SINDy ---
POLY_DEGREE = 3
THRESHOLD = 0.05

# --- Selección de Datos ---
USE_MANUAL_SELECTION = False

MANUAL_PARAM_KEYS = [
    "0.2146_0.1525", 
    "0.2005_0.2283", 
    "0.2182_0.2636", 
    "0.2288_0.1222", 
]

N_RANDOM_CONFIGS = 5

# =============================================================================
# UTILIDADES
# =============================================================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_next_folder_id(base_dir):
    """Busca la siguiente carpeta numérica disponible."""
    if not os.path.exists(base_dir):
        return 1
    subdirs = [d for d in os.listdir(base_dir) if d.isdigit() and os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return 1
    return max(int(d) for d in subdirs) + 1

# =============================================================================
# LÓGICA DE CARGA GENÉRICA
# =============================================================================
def load_data_generic(hdf5_path, t_max=None):
    X_list = []
    sys_state_dim = len(System.state_names)
    
    print(f"Cargando datos desde {hdf5_path}...")
    print(f"Sistema detectado: {System.name}")
    if t_max:
        print(f" *** RECORTE TEMPORAL ACTIVADO: Usando primeros {t_max} segundos ***")
    else:
        print(" *** Usando trayectorias completas ***")

    with h5py.File(hdf5_path, "r") as hf:
        if "t_eval" not in hf: raise ValueError("HDF5 sin t_eval.")
        
        t_full = hf["t_eval"][:]
        
        # Recorte temporal
        if t_max is not None:
            valid_indices = np.where(t_full <= t_max)[0]
            if valid_indices.size == 0: raise ValueError("t_max muy pequeño.")
            limit_idx = valid_indices[-1] + 1
            t_train = t_full[:limit_idx]
        else:
            limit_idx = len(t_full)
            t_train = t_full

        param_keys_all = [k for k in hf.keys() if k != "t_eval"]
        if not param_keys_all: raise ValueError("HDF5 vacío.")

        # Selección de claves
        if USE_MANUAL_SELECTION:
            print(f"\n*** MODO MANUAL: Usando {len(MANUAL_PARAM_KEYS)} configuraciones ***")
            param_keys = [k for k in MANUAL_PARAM_KEYS if k in param_keys_all]
            if not param_keys: raise ValueError("Ninguna clave manual válida.")
        else:
            print(f"\n*** MODO ALEATORIO: Usando {N_RANDOM_CONFIGS} configuraciones ***")
            n_sample = min(N_RANDOM_CONFIGS, len(param_keys_all))
            param_keys = random.sample(param_keys_all, n_sample)

        # Carga
        for key in tqdm(param_keys, desc="Procesando grupos"):
            try:
                param_vals = parse_param_key(key)
                trajs_raw = hf[key]["trajectories"]["all_trajectories"][:, :, :limit_idx]
                n_trajs, state_dim, n_points = trajs_raw.shape
                
                if state_dim != sys_state_dim: continue

                param_block = np.tile(param_vals, (n_points, 1))
                for i in range(n_trajs):
                    traj_state_T = trajs_raw[i].T 
                    X_augmented = np.hstack((traj_state_T, param_block))
                    X_list.append(X_augmented)
            except Exception as e:
                print(f"Error procesando clave {key}: {e}")

    print(f"Carga finalizada. {len(X_list)} trayectorias de {len(t_train)} puntos.")
    return X_list, t_train, param_keys

def calculate_derivatives(X_list, t):
    X_dot_list = []
    print("Calculando derivadas (np.gradient)...")
    for X in tqdm(X_list, desc="Derivando"):
        X_dot = np.gradient(X, t, axis=0)
        X_dot_list.append(X_dot)
    return X_dot_list

# =============================================================================
# MAIN
# =============================================================================
def main():
    # 1. Preparar Directorios de Salida
    # Determinar Batch Size
    if USE_MANUAL_SELECTION:
        batch_size = len(MANUAL_PARAM_KEYS)
    else:
        batch_size = N_RANDOM_CONFIGS
        
    batch_dir = os.path.join(BASE_OUTPUT_ROOT, str(batch_size))
    os.makedirs(batch_dir, exist_ok=True)
    
    # Determinar ID de Versión
    version_id = get_next_folder_id(batch_dir)
    output_dir = os.path.join(batch_dir, str(version_id))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Iniciando Discovery V{version_id} (Batch {batch_size}) ===")
    print(f"Salida en: {output_dir}")

    # 2. Cargar Datos
    try:
        X_list, t, used_keys = load_data_generic(HDF5_FILE, t_max=TRAIN_T_MAX)
    except Exception as e:
        print(f"Error fatal cargando datos: {e}")
        return

    if not X_list:
        print("Lista de datos vacía.")
        return

    # 3. Entrenar
    X_dot_list = calculate_derivatives(X_list, t)
    feature_names = System.state_names + System.param_names
    
    optimizer = ps.STLSQ(threshold=THRESHOLD)
    feature_library = ps.PolynomialLibrary(degree=POLY_DEGREE)

    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=feature_library,
        differentiation_method=None 
    )

    print(f"\nEntrenando SINDy sobre {len(feature_names)} variables...")
    model.fit(
        X_list, 
        x_dot=X_dot_list, 
        t=t, 
        feature_names=feature_names 
    )

    print("\n--- ¡Modelo Descubierto! ---")
    model.print()

    # 4. Guardar Archivos del Modelo
    model_path = os.path.join(output_dir, "sindy_model.joblib")
    params_path = os.path.join(output_dir, "sindy_training_params.json")
    
    print(f"\nGuardando modelo en {model_path}...")
    joblib.dump(model, model_path)
    
    # Metadata específica de este run
    run_metadata = {
        "id": version_id,
        "system_name": System.name,
        "timestamp": datetime.now().isoformat(),
        "selection_mode": "MANUAL" if USE_MANUAL_SELECTION else "RANDOM",
        "keys_used": used_keys, # Compatible con run_comparison legacy
        "training_keys_mu1_mu2": used_keys, # Compatible con nuevo standard
        "poly_degree": POLY_DEGREE,
        "threshold": THRESHOLD,
        "train_t_max": TRAIN_T_MAX,
        "model_path_abs": os.path.abspath(model_path)
    }
    
    with open(params_path, "w") as f:
        json.dump(run_metadata, f, indent=2, cls=NpEncoder)

    # 5. Actualizar Log Centralizado (versions_log.json)
    central_log_path = os.path.join(batch_dir, "versions_log.json")
    full_log = []
    
    if os.path.exists(central_log_path):
        try:
            with open(central_log_path, "r") as f:
                full_log = json.load(f)
        except json.JSONDecodeError:
            pass
    
    # Info resumen para el log
    log_entry = {
        "id": version_id,
        "path": f"{version_id}/",
        "timestamp": run_metadata["timestamp"],
        "train_t_max": TRAIN_T_MAX,
        "threshold": THRESHOLD,
        "keys_count": len(used_keys)
    }
    
    full_log.append(log_entry)
    
    with open(central_log_path, "w") as f:
        json.dump(full_log, f, indent=2, cls=NpEncoder)
        
    print(f"Log actualizado en: {central_log_path}")
    print("Proceso completado exitosamente.")

if __name__ == "__main__":
    main()