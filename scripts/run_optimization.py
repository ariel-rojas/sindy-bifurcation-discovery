#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de OPTIMIZACIÓN de Hiperparámetros y Datos.

Objetivo: Encontrar los MEJORES modelos (Top 5) variando datos e hiperparámetros.

CAMBIOS ESTRUCTURALES:
- Guarda modelos en carpetas jerárquicas: output/{BATCH_SIZE}/{ID_ITERACION}/
- Mantiene un log histórico acumulativo en output/{BATCH_SIZE}/optimization_log.json
- Soporta recorte temporal (TRAIN_T_MAX).
"""

import os
import h5py
import numpy as np
import pysindy as ps
import joblib
import json
import random
import itertools
import shutil
from tqdm import tqdm
import sys
import copy
from datetime import datetime

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System
from core.io import parse_param_key

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
CARPETA_BASE = os.path.join(PROJECT_ROOT, "output")
HDF5_FILE = os.path.join(CARPETA_BASE, "trajectory_data.hdf5")

# --- Parámetros del Experimento ---
NUM_ITERATIONS_DATA = 50
BATCH_SIZE_DATA = 5
TOP_N_KEEP = 3  # Guardar los mejores 5

# --- Configuración Temporal ---
# None = Todo el tiempo. Valor numérico (ej 20.0) = Recorte en segundos
TRAIN_T_MAX = 40
FIXED_POLY_DEGREE = 3

# --- Grilla ---
PARAM_GRID = {
    "threshold": [],      
    "optimizer_alpha": [],
}

# Rutas Dinámicas (se definen en main pero las preparamos aquí)
BATCH_DIR = os.path.join(CARPETA_BASE, str(BATCH_SIZE_DATA))
LOG_FILE = os.path.join(BATCH_DIR, "versions_log.json")

# =============================================================================
# UTILIDADES
# =============================================================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def generate_hyperparam_combinations(grid):
    active_keys = {k: v for k, v in grid.items() if v}
    if not active_keys: return [{}]
    keys = list(active_keys.keys())
    values = list(active_keys.values())
    product = list(itertools.product(*values))
    return [dict(zip(keys, p)) for p in product]

def calculate_model_error(model, true_coeffs_list):
    learned_coeffs = model.coefficients()
    feature_names = model.get_feature_names()
    
    total_error = 0.0
    term_count = 0
    details = {}
    
    for eq_idx, true_terms in enumerate(true_coeffs_list):
        try:
            eq_name = System.state_names[eq_idx]
        except IndexError:
            eq_name = f"eq_{eq_idx}"
        
        eq_error = 0.0
        details[eq_name] = {}
        
        for term, true_val in true_terms.items():
            if term in feature_names:
                feat_idx = feature_names.index(term)
                learned_val = learned_coeffs[eq_idx, feat_idx]
                
                # Error Relativo
                abs_diff = abs(learned_val - true_val)
                denominator = abs(true_val) if abs(true_val) > 1e-9 else 1.0
                rel_err = abs_diff / denominator
                
                eq_error += rel_err
                details[eq_name][term] = {
                    "true": float(true_val),
                    "pred": float(learned_val),
                    "diff": float(rel_err)
                }
            else:
                eq_error += 1.0
                details[eq_name][term] = {"true": float(true_val), "pred": 0.0, "status": "MISSING"}
            term_count += 1

        for feat_idx, feat_name in enumerate(feature_names):
            val = learned_coeffs[eq_idx, feat_idx]
            if abs(val) > 1e-6 and feat_name not in true_terms:
                details[eq_name][f"GHOST_{feat_name}"] = {"pred": float(val), "status": "GHOST"}
        
        total_error += eq_error

    avg_error = total_error / max(1, term_count)
    return avg_error, details

def load_batch_data(hf, keys, limit_idx=None):
    X_list = []
    sys_state_dim = len(System.state_names)
    for key in keys:
        try:
            param_vals = parse_param_key(key)
            if limit_idx is not None:
                trajs_raw = hf[key]["trajectories"]["all_trajectories"][:, :, :limit_idx]
            else:
                trajs_raw = hf[key]["trajectories"]["all_trajectories"][:]
                
            n_trajs, state_dim, n_points = trajs_raw.shape
            if state_dim != sys_state_dim: continue
            param_block = np.tile(param_vals, (n_points, 1))
            for i in range(n_trajs):
                traj_state_T = trajs_raw[i].T
                X_augmented = np.hstack((traj_state_T, param_block))
                X_list.append(X_augmented)
        except Exception:
            pass
    return X_list

def get_next_folder_id(base_dir):
    """
    Busca la carpeta numérica más alta dentro de base_dir y devuelve la siguiente.
    Si no hay carpetas, devuelve 1.
    """
    if not os.path.exists(base_dir):
        return 1
    
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    max_id = 0
    for d in subdirs:
        try:
            num = int(d)
            if num > max_id:
                max_id = num
        except ValueError:
            continue # Ignorar carpetas no numéricas
            
    return max_id + 1

# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"--- Iniciando Optimización Top-{TOP_N_KEEP} ---")
    print(f"Batch Size: {BATCH_SIZE_DATA} | T_MAX: {TRAIN_T_MAX}")
    
    # Crear carpeta base del batch si no existe
    os.makedirs(BATCH_DIR, exist_ok=True)
    
    try:
        with h5py.File(HDF5_FILE, "r") as hf:
            if "t_eval" not in hf: raise ValueError("HDF5 corrupto")
            t_eval_full = hf["t_eval"][:]
            all_keys = [k for k in hf.keys() if k != "t_eval"]
    except FileNotFoundError:
        print(f"Error: No se encuentra {HDF5_FILE}.")
        return
    
    # Preparar Recorte Temporal
    if TRAIN_T_MAX is not None:
        valid_indices = np.where(t_eval_full <= TRAIN_T_MAX)[0]
        limit_idx = valid_indices[-1] + 1
        t_eval_train = t_eval_full[:limit_idx]
    else:
        limit_idx = None
        t_eval_train = t_eval_full

    print(f"Total configuraciones disponibles: {len(all_keys)}")
    
    hyperparam_list = generate_hyperparam_combinations(PARAM_GRID)
    
    # Cache de mejores modelos
    top_runs_cache = [] 
    
    total_runs = len(hyperparam_list) * NUM_ITERATIONS_DATA
    pbar = tqdm(total=total_runs, desc="Optimizando")
    
    feature_names_list = System.state_names + System.param_names

    for params in hyperparam_list:
        opt_kwargs = params.copy()
        opt = ps.STLSQ(**opt_kwargs)
        lib = ps.PolynomialLibrary(degree=FIXED_POLY_DEGREE)
        
        for i_iter in range(NUM_ITERATIONS_DATA):
            current_keys = random.sample(all_keys, min(BATCH_SIZE_DATA, len(all_keys)))
            
            with h5py.File(HDF5_FILE, "r") as hf:
                X_train = load_batch_data(hf, current_keys, limit_idx=limit_idx)
            
            if not X_train:
                pbar.update(1)
                continue
                
            X_dot_train = [np.gradient(x, t_eval_train, axis=0) for x in X_train]
            model = ps.SINDy(optimizer=opt, feature_library=lib, differentiation_method=None)
            
            try:
                model.fit(
                    X_train, x_dot=X_dot_train, t=t_eval_train, 
                    feature_names=feature_names_list,
                )
                
                score, details = calculate_model_error(model, System.get_true_coefficients())
                
                run_info = {
                    "score_error": score,
                    "hyperparams": params,
                    "training_keys_mu1_mu2": current_keys,
                    "details": details,
                    "train_t_max": TRAIN_T_MAX,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Lógica Top N
                should_insert = False
                if len(top_runs_cache) < TOP_N_KEEP:
                    should_insert = True
                else:
                    worst_score = top_runs_cache[-1][0]
                    if score < worst_score:
                        should_insert = True
                
                if should_insert:
                    top_runs_cache.append((score, run_info, copy.deepcopy(model)))
                    top_runs_cache.sort(key=lambda x: x[0])
                    top_runs_cache = top_runs_cache[:TOP_N_KEEP]
                    
            except Exception:
                pass
            
            pbar.update(1)

    pbar.close()
    
    # =========================================================================
    # GUARDADO JERÁRQUICO
    # =========================================================================
    print("\n" + "="*60)
    print(f"RESULTADOS: TOP {TOP_N_KEEP} MODELOS")
    print("="*60)
    
    if not top_runs_cache:
        print("No se encontraron modelos válidos.")
        return

    # Obtener el siguiente ID de carpeta disponible (ej: 8)
    # Como vamos a guardar 5 modelos, reservaremos 5 IDs consecutivos
    # ej: 8, 9, 10, 11, 12
    start_id = get_next_folder_id(BATCH_DIR)
    
    # Lista para appendear al log histórico
    new_log_entries = []

    for i, (score, info, model_obj) in enumerate(top_runs_cache):
        current_id = start_id + i
        model_dir = os.path.join(BATCH_DIR, str(current_id))
        
        # Crear carpeta numerada (ej: output/8/12)
        os.makedirs(model_dir, exist_ok=True)
        
        # Rutas de archivo
        job_path = os.path.join(model_dir, "sindy_model.joblib")
        json_path = os.path.join(model_dir, "sindy_training_params.json")
        
        # 1. Guardar Modelo
        joblib.dump(model_obj, job_path)
        
        # 2. Guardar JSON Individual (compatible con run_comparison)
        # Aseguramos estructura compatible con scripts legacy
        json_content = {
            "keys_used": info['training_keys_mu1_mu2'], # Legacy
            "training_keys_mu1_mu2": info['training_keys_mu1_mu2'],
            "best_run_details": info,
            "model_path_abs": job_path
        }
        
        with open(json_path, "w") as f:
            json.dump(json_content, f, indent=2, cls=NpEncoder)
            
        # Preparar entrada para el log histórico
        log_entry = {
            "id": current_id,
            "path": f"{current_id}/", # Ruta relativa desde BATCH_DIR
            "score_error": score,
            "train_t_max": TRAIN_T_MAX,
            "hyperparams": info['hyperparams'],
            "training_keys": info['training_keys_mu1_mu2'],
            "timestamp": info['timestamp']
        }
        new_log_entries.append(log_entry)
        
        print(f"Guardado ID {current_id}: Error={score:.6f} en {model_dir}")

    # --- Actualizar Log Histórico (Append) ---
    full_log = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                full_log = json.load(f)
        except json.JSONDecodeError:
            print("Advertencia: El log existente estaba corrupto, se creará uno nuevo.")
            full_log = []
    
    # Agregar nuevos
    full_log.extend(new_log_entries)
    
    # Guardar log actualizado
    with open(LOG_FILE, "w") as f:
        json.dump(full_log, f, indent=2, cls=NpEncoder)
        
    print(f"\nLog histórico actualizado en: {LOG_FILE}")

if __name__ == "__main__":
    main()