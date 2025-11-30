#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de AJUSTE FINO (Fine-Tuning / Hill Climbing).

Objetivo: Tomar el MEJOR modelo encontrado por run_optimization.py y mejorarlo
localmente mediante perturbaciones pequeñas en:
1. El conjunto de datos de entrenamiento (cambiando 1 trayectoria a la vez).
2. Los hiperparámetros (threshold y alpha).

Flujo:
Carga Ganador -> Calcula Error Base -> Bucle de Mejora -> Guarda si mejora.
"""

import os
import h5py
import numpy as np
import pysindy as ps
import joblib
import json
import random
import sys
from tqdm import tqdm
import copy

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System
from core.io import parse_param_key

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
CARPETA_FILES = os.path.join(PROJECT_ROOT, "output")
HDF5_FILE = os.path.join(CARPETA_FILES, "trajectory_data.hdf5")

# Entrada (El mejor resultado previo)
INPUT_JSON_PATH = os.path.join(CARPETA_FILES, "optimization_results", "sindy_training_params.json")

# Salida (El resultado refinado)
OUTPUT_DIR = os.path.join(CARPETA_FILES, "fine_tuning_results")
REFINED_MODEL_PATH = os.path.join(OUTPUT_DIR, "refined_sindy_model.joblib")
REFINED_JSON_PATH = os.path.join(OUTPUT_DIR, "refined_params.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parámetros del Hill Climbing
ITERATIONS_DATA = 50      # Intentos de cambiar trayectorias
ITERATIONS_PARAMS = 20    # Intentos de ajustar hiperparámetros
FIXED_POLY_DEGREE = 3

# =============================================================================
# UTILIDADES (Reutilizadas para consistencia)
# =============================================================================
def calculate_model_error(model, true_coeffs_list):
    """Calcula el error absoluto promedio (Misma lógica que optimizador)."""
    learned_coeffs = model.coefficients()
    feature_names = model.get_feature_names()
    total_error = 0.0
    term_count = 0
    
    for eq_idx, true_terms in enumerate(true_coeffs_list):
        eq_error = 0.0
        # 1. Términos esperados
        for term, true_val in true_terms.items():
            if term in feature_names:
                feat_idx = feature_names.index(term)
                val = learned_coeffs[eq_idx, feat_idx]
                eq_error += abs(val - true_val)
            else:
                eq_error += abs(true_val)
            term_count += 1
        
        # 2. Fantasmas
        for feat_idx, feat_name in enumerate(feature_names):
            val = learned_coeffs[eq_idx, feat_idx]
            if abs(val) > 1e-6 and feat_name not in true_terms:
                eq_error += abs(val)
        
        total_error += eq_error

    return total_error / max(1, term_count)

def load_batch_data(hf, keys):
    """Carga datos del HDF5."""
    X_list = []
    sys_state_dim = len(System.state_names)
    for key in keys:
        try:
            param_vals = parse_param_key(key)
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

def train_and_evaluate(keys, hyperparams, t_eval):
    """Función helper para entrenar y obtener score."""
    # Cargar datos
    with h5py.File(HDF5_FILE, "r") as hf:
        X_train = load_batch_data(hf, keys)
    
    if not X_train: return float('inf'), None

    X_dot_train = [np.gradient(x, t_eval, axis=0) for x in X_train]
    
    # Configurar modelo
    lib = ps.PolynomialLibrary(degree=FIXED_POLY_DEGREE)
    opt = ps.STLSQ(**hyperparams)
    
    model = ps.SINDy(optimizer=opt, feature_library=lib, differentiation_method=None)
    
    try:
        model.fit(
            X_train, x_dot=X_dot_train, t=t_eval, 
            multiple_trajectories=True, 
            feature_names=System.state_names + System.param_names,
            quiet=True
        )
        score = calculate_model_error(model, System.get_true_coefficients())
        return score, model
    except Exception:
        return float('inf'), None

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("--- Iniciando Ajuste Fino (Fine-Tuning) ---")
    
    # 1. Cargar Configuración Ganadora Previa
    if not os.path.exists(INPUT_JSON_PATH):
        print("Error: No se encontró el resultado de la optimización previa.")
        return
        
    with open(INPUT_JSON_PATH, "r") as f:
        data = json.load(f)
        # Soporte para el formato nuevo (top_models) o viejo
        if "best_run_details" in data:
            base_config = data["best_run_details"]
        else:
            # Fallback si el json tiene formato directo
            base_config = data 

    current_keys = base_config["training_keys_mu1_mu2"]
    current_hyperparams = base_config["params"] if "params" in base_config else base_config["hyperparams"]
    
    # Limpiar hiperparámetros que no son de STLSQ (como 'degree' o 'batch_keys')
    valid_stslq_args = ["threshold", "alpha", "max_iter", "normalize_columns"]
    current_hyperparams = {k: v for k, v in current_hyperparams.items() if k in valid_stslq_args}

    print(f"Configuración Inicial:")
    print(f" - Error Base (Reportado): {base_config.get('score_error', 'N/A')}")
    print(f" - Keys: {len(current_keys)}")
    print(f" - Params: {current_hyperparams}")

    # 2. Preparar Entorno
    with h5py.File(HDF5_FILE, "r") as hf:
        t_eval = hf["t_eval"][:]
        all_available_keys = [k for k in hf.keys() if k != "t_eval"]

    # 3. Establecer Línea Base Real (Recalcular error para asegurar consistencia)
    print("\nRecalculando línea base...")
    best_score, best_model = train_and_evaluate(current_keys, current_hyperparams, t_eval)
    print(f" -> Error Base Recalculado: {best_score:.6f}")
    
    best_keys = list(current_keys)
    best_hyperparams = current_hyperparams.copy()
    
    # =========================================================================
    # FASE 1: PERTURBACIÓN DE DATOS (Data Hill Climbing)
    # =========================================================================
    print(f"\n[Fase 1] Refinando Datos ({ITERATIONS_DATA} iteraciones)...")
    
    pbar = tqdm(total=ITERATIONS_DATA)
    for _ in range(ITERATIONS_DATA):
        # Estrategia: Reemplazar 1 clave al azar
        new_keys = list(best_keys)
        idx_to_swap = random.randint(0, len(new_keys) - 1)
        new_key = random.choice(all_available_keys)
        
        # Evitar duplicados
        while new_key in new_keys:
            new_key = random.choice(all_available_keys)
            
        new_keys[idx_to_swap] = new_key
        
        # Probar
        score, model = train_and_evaluate(new_keys, best_hyperparams, t_eval)
        
        if score < best_score:
            best_score = score
            best_keys = new_keys
            best_model = model
            pbar.set_description(f"¡Mejora! Error: {best_score:.6f}")
        
        pbar.update(1)
    pbar.close()

    # =========================================================================
    # FASE 2: PERTURBACIÓN DE HIPERPARÁMETROS
    # =========================================================================
    print(f"\n[Fase 2] Refinando Hiperparámetros ({ITERATIONS_PARAMS} iteraciones)...")
    
    pbar = tqdm(total=ITERATIONS_PARAMS)
    for _ in range(ITERATIONS_PARAMS):
        new_hyperparams = best_hyperparams.copy()
        
        # Estrategia: Variar threshold o alpha un +/- 10% o 20%
        target = random.choice(["threshold", "alpha"])
        if target in new_hyperparams and isinstance(new_hyperparams[target], (int, float)):
            current_val = new_hyperparams[target]
            factor = random.uniform(0.8, 1.2) # Variación entre 80% y 120%
            new_hyperparams[target] = current_val * factor
            
        # Probar
        score, model = train_and_evaluate(best_keys, new_hyperparams, t_eval)
        
        if score < best_score:
            best_score = score
            best_hyperparams = new_hyperparams
            best_model = model
            pbar.set_description(f"¡Mejora! Error: {best_score:.6f}")
            
        pbar.update(1)
    pbar.close()

    # =========================================================================
    # RESULTADOS
    # =========================================================================
    print("\n" + "="*60)
    print("RESULTADOS DEL AJUSTE FINO")
    print("="*60)
    print(f"Error Final Refinado: {best_score:.6f}")
    print("Mejores Hiperparámetros:")
    print(json.dumps(best_hyperparams, indent=2))
    print("Mejores Claves:")
    print(json.dumps(best_keys, indent=2))
    
    # Guardar
    joblib.dump(best_model, REFINED_MODEL_PATH)
    
    final_info = {
        "source_base_error": base_config.get('score_error', 'N/A'),
        "final_error": best_score,
        "hyperparams": best_hyperparams,
        "training_keys_mu1_mu2": best_keys
    }
    
    with open(REFINED_JSON_PATH, "w") as f:
        json.dump(final_info, f, indent=2)
        
    print(f"\nModelo Refinado guardado en: {REFINED_MODEL_PATH}")
    print(f"Metadata guardada en: {REFINED_JSON_PATH}")

if __name__ == "__main__":
    main()