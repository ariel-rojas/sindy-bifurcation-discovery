#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de BARRIDO TEMPORAL (Time Sweep Discovery).

Objetivo:
1. Seleccionar un set FIJO de trayectorias.
2. Entrenar modelos SINDy incrementando el horizonte temporal.
3. Comparar modos Unbias=True vs Unbias=False.
4. Graficar evolución de coeficientes y medir tiempos.
"""

import os
import sys
import time
import h5py
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import random
import joblib
import json
from tqdm import tqdm

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System
from core.io import parse_param_key

# Rutas
BASE_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "tb_zone")
HDF5_FILE = os.path.join(BASE_OUTPUT_ROOT, "trajectory_data.hdf5")
SWEEP_OUTPUT_DIR = os.path.join(BASE_OUTPUT_ROOT, "time_sweep_analysis")

# Parámetros SINDy
POLY_DEGREE = 3
THRESHOLD = 0.1
N_RANDOM_CONFIGS = 5

# Intervalos de tiempo a evaluar (de 5s a 75s cada 5s para que no sea eterno, puedes bajarlo a 2.5)
TIME_STEPS = np.arange(5, 76, 15) 

# Valores Teóricos
THEORETICAL_MAP = {
    0: {'y': 1.0},
    1: {'mu1': -1.0, 'x mu2': -1.0, 'x^2': 1.0, 'x^3': -1.0, 'x^2 y': -1.0, 'x y': -1.0}
}

# =============================================================================
# UTILIDADES
# =============================================================================
def normalize_term_name(name):
    return " ".join(sorted(name.split()))

def load_fixed_data_full(hdf5_path, n_configs):
    """Carga los datos COMPLETOS una sola vez."""
    X_full_list = []
    t_start = time.time()
    
    print(f"Abriendo {hdf5_path}...")
    with h5py.File(hdf5_path, "r") as hf:
        if "t_eval" not in hf: raise ValueError("Falta t_eval")
        t_full = hf["t_eval"][:]
        
        keys = [k for k in hf.keys() if k != "t_eval"]
        selected_keys = random.sample(keys, min(n_configs, len(keys)))
        print(f"Selección FIJA de grupos: {selected_keys}")
        
        for key in tqdm(selected_keys, desc="Cargando datos a RAM"):
            param_vals = parse_param_key(key)
            trajs = hf[key]["trajectories"]["all_trajectories"][:]
            n_trajs, _, n_points = trajs.shape
            param_block = np.tile(param_vals, (n_points, 1))
            
            for i in range(n_trajs):
                traj_T = trajs[i].T
                X_aug = np.hstack((traj_T, param_block))
                X_full_list.append(X_aug)
                
    print(f"-> Carga completada en {time.time() - t_start:.2f} segundos.")
    return X_full_list, t_full

# =============================================================================
# LÓGICA DE BARRIDO (CORE)
# =============================================================================
def run_sweep_analysis(unbias_flag, X_full_list, t_full):
    """Ejecuta el barrido temporal completo para una configuración de unbias."""
    
    mode_str = "TRUE" if unbias_flag else "FALSE"
    mode_name = f"Unbias={mode_str}"

    print(f"\n{'='*60}")
    print(f"INICIANDO BARRIDO: {mode_name}")
    print(f"{'='*60}")
    
    start_sweep = time.time()
    
    # Estructura para guardar historia
    coeffs_history = {0: {}, 1: {}}
    
    # Bucle de Tiempo
    for t_max in tqdm(TIME_STEPS, desc=f"Entrenando ({mode_name})"):
        
        # 1. Recorte en Memoria
        valid_idx = np.where(t_full <= t_max)[0]
        limit = valid_idx[-1] + 1
        t_slice = t_full[:limit]
        X_slice_list = [x[:limit, :] for x in X_full_list]
        
        # 2. Entrenamiento
        # Configuración del optimizador según el flag
        optimizer = ps.STLSQ(threshold=THRESHOLD, unbias=unbias_flag, verbose = True)
        feature_library = ps.PolynomialLibrary(degree=POLY_DEGREE)
        feature_names = System.state_names + System.param_names
        
        model = ps.SINDy(
            optimizer=optimizer,
            feature_library=feature_library,
            differentiation_method=ps.FiniteDifference()
        )
        
        # Silenciamos output standard de fit para imprimir el nuestro
        model.fit(X_slice_list, t=t_slice, feature_names=feature_names)
        
        # 3. Print Customizado
        print(f"\n--- [T={t_max:.1f}s | {mode_name}] ---")
        model.print(precision=3)
        
        model_filename = f"model_unbias_{mode_str}_t_{t_max:.1f}.joblib"
        model_path = os.path.join(SWEEP_OUTPUT_DIR, model_filename)
        joblib.dump(model, model_path)
        
        # 4. Extraer Coeficientes
        feats = model.get_feature_names()
        coeffs = model.optimizer.coef_
        
        for target_i in range(2):
            for feat_i, feat_name in enumerate(feats):
                norm_name = normalize_term_name(feat_name)
                if norm_name not in coeffs_history[target_i]:
                    coeffs_history[target_i][norm_name] = []
                coeffs_history[target_i][norm_name].append(coeffs[target_i, feat_i])

    duration = time.time() - start_sweep
    print(f"\n-> Barrido {mode_name} finalizado en {duration:.2f} segundos.")
    
    # 5. Graficar
    plot_results(coeffs_history, unbias_flag)

def plot_results(coeffs_history, unbias_flag):
    """Genera y guarda el gráfico para el modo actual."""
    mode_str = "TRUE" if unbias_flag else "FALSE"
    print(f"Generando gráfico para Unbias={mode_str}...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    targets = ["x'", "y'"]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for target_i in range(2):
        ax = axes[target_i]
        hist_dict = coeffs_history[target_i]
        theo_dict = THEORETICAL_MAP[target_i]
        theo_norm = {normalize_term_name(k): v for k, v in theo_dict.items()}
        
        color_idx = 0
        
        for term, values in hist_dict.items():
            is_theoretical = term in theo_norm
            max_val = np.max(np.abs(values))
            final_val = values[-1]
            # Filtro para limpiar gráfico: Teóricos O Espurios grandes
            if is_theoretical or max_val > 0.05:
                c = colors[color_idx % 10]
                label = f"{term}: {final_val:.3f}"
                style = '-'
                width = 1
                alpha = 1.0
                
                if not is_theoretical:
                    label = f"{term} (Espurio)"
                    style = 'dotted'
                    width = 1
                    c = 'gray'
                    alpha = 0.6
                else:
                    color_idx += 1
                
                ax.plot(TIME_STEPS, values, label=label, color=c, ls=style, lw=width, alpha=alpha, marker='.', markersize=4)
                
                if is_theoretical:
                    val_true = theo_norm[term]
                    ax.axhline(val_true, color=c, ls='--', alpha=0.4)
        
        ax.set_title(f"Ecuación {targets[target_i]}")
        ax.set_xlabel("Tiempo de Entrenamiento (s)")
        ax.set_ylabel("Valor del Coeficiente")
        ax.grid(True, alpha=0.3)
        # Leyenda fuera para no tapar
        ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize='small', title="Términos")

    plt.suptitle(f"Evolución de Coeficientes vs Tiempo\n(Unbias={mode_str}, 5 Grupos Fijos)", fontsize=16)
    plt.tight_layout()
    
    filename = f"time_sweep_unbias_{str(mode_str).upper()}.png"
    save_path = os.path.join(SWEEP_OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150)
    print(f"-> Gráfico guardado en: {save_path}")
    plt.close() # Cerrar para liberar memoria

# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(SWEEP_OUTPUT_DIR, exist_ok=True)
    
    # 1. Cargar Datos (Común para ambos barridos)
    X_full_list, t_full = load_fixed_data_full(HDF5_FILE, N_RANDOM_CONFIGS)
    
    # 2. Ejecutar Barrido 1: CON unbias (El "problemático" en teoría)
    run_sweep_analysis(unbias_flag=True, X_full_list=X_full_list, t_full=t_full)
    
    # 3. Ejecutar Barrido 2: SIN unbias (El "robusto")
    run_sweep_analysis(unbias_flag=False, X_full_list=X_full_list, t_full=t_full)
    
    print("\n=== TODOS LOS PROCESOS TERMINADOS ===")

if __name__ == "__main__":
    main()