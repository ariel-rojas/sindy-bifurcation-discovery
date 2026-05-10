#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualizador de Convergencia de Coeficientes SINDy.

Este script carga un modelo entrenado (.joblib) y extrae el historial
del optimizador para graficar cómo evolucionaron los coeficientes.

CORRECCIÓN: Incluye parche para error 'No module named pysindy.pysindy'.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pysindy as ps

# =============================================================================
# PARCHE PARA ERROR DE JOBLIB/PICKLE
# =============================================================================
# 1. Parche para la clase principal SINDy
if 'pysindy.pysindy' not in sys.modules:
    sys.modules['pysindy.pysindy'] = ps

# 2. Parche para utilidades de ejes (Causa de tu último error)
# Mapeamos 'pysindy.utils.axes' al módulo genérico 'pysindy.utils'
if 'pysindy.utils.axes' not in sys.modules:
    sys.modules['pysindy.utils.axes'] = ps.utils

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
# Ajusta la ruta a tu modelo
MODEL_PATH = os.path.join(PROJECT_ROOT, "tb_zone", "time_sweep_analysis_1", "model_unbias_TRUE_t_20.0.joblib")

# --- VALORES TEÓRICOS (Target Value) ---
# Definimos el valor esperado para cada término en cada ecuación.
# Ecuación 0 (x_dot): y
# Ecuación 1 (y_dot): -mu1 - mu2*x + x^2 - x^3 - x^2*y - x*y

THEORETICAL_MAP = {
    # Ecuación 0: x' = y
    0: {
        'y': 1.0
    },
    # Ecuación 1: y' = ...
    1: {
        'mu1': -1.0,
        'x mu2': -1.0, # SINDy puede llamarlo "mu2 x" o "x mu2"
        'x^2': 1.0,
        'x^3': -1.0,
        'x^2 y': -1.0, # Termino -(x^2 + x)y -> -x^2y
        'x y': -1.0    # Termino -(x^2 + x)y -> -xy
    }
}

# =============================================================================
# UTILIDADES
# =============================================================================
def normalize_term_name(name):
    """Ordena los componentes de una multiplicación para comparar strings."""
    # Ejemplo: "mu2 x" -> "mu2 x" (orden alfabético de partes)
    parts = name.split()
    return " ".join(sorted(parts))

# =============================================================================
# FUNCIÓN DE VISUALIZACIÓN
# =============================================================================
def plot_convergence(model_path):
    print(f"Cargando modelo desde: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el archivo {model_path}")
        return

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    optimizer = model.optimizer
    if not hasattr(optimizer, 'history_') or not optimizer.history_:
        print("Error: El optimizador no guardó historial.")
        return

    # 1. Obtener Nombres
    try:
        feature_names = model.get_feature_names()
    except:
        feature_names = [f"f{i}" for i in range(len(optimizer.coef_[0]))]
    
    n_features_total = len(feature_names)
    n_targets_expected = model.optimizer.coef_.shape[0]

    # 2. Procesar Historial (Corrección de dimensiones)
    history = np.array(optimizer.history_)
    print(history)
    print(history.shape)
    shape = history.shape
    
    # Asegurar formato (Iter, Features, Targets)
    if shape[1] == n_targets_expected and shape[2] == n_features_total:
        history = np.swapaxes(history, 1, 2)
    
    n_iters = history.shape[0]
    
    # Nombres de ecuaciones
    target_names = ["x'", "y'"]
    if n_targets_expected > 2:
        for i in range(2, n_targets_expected): target_names.append(f"Eq {i}")

    # 3. Imprimir Ecuación Descubierta
    print("\n" + "="*40)
    print("ECUACIÓN DESCUBIERTA POR EL MODELO:")
    print("="*40)
    model.print()
    print("="*40 + "\n")

    # 4. Graficar
    n_plots = min(n_targets_expected, 2) 
    fig, axes = plt.subplots(1, n_plots, figsize=(14, 6), sharey=False)
    if n_plots == 1: axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for target_i in range(n_plots):
        ax = axes[target_i]
        
        # Mapa teórico para esta ecuación
        theoretical_terms_dict = THEORETICAL_MAP.get(target_i, {})
        
        # Para matchear flexiblemente, normalizamos las claves del diccionario teórico
        normalized_theoretical = {normalize_term_name(k): v for k, v in theoretical_terms_dict.items()}
        
        found_any = False
        color_idx = 0
        
        # Iterar sobre features del modelo
        for feat_i, name in enumerate(feature_names):
            norm_name = normalize_term_name(name)
            
            # Verificar si es un término teórico
            if norm_name in normalized_theoretical:
                found_any = True
                true_val = normalized_theoretical[norm_name]
                coef_evolution = history[:, feat_i, target_i]
                
                # Color consistente
                c = colors[color_idx % len(colors)]
                color_idx += 1
                
                # A) Línea de Convergencia (Sólida)
                ax.plot(range(n_iters), coef_evolution, 
                        label=f"{name}", 
                        color=c, linewidth=2.5, marker='o', markersize=4)
                
                # B) Línea Teórica (Punteada del mismo color)
                ax.axhline(y=true_val, color=c, linestyle='--', alpha=0.6, linewidth=1.5)
                
                # Anotación final
                final_val = coef_evolution[-1]
                ax.annotate(f"{final_val:.3f}", 
                            xy=(n_iters-1, final_val), 
                            xytext=(5, 5 if final_val > true_val else -15), 
                            textcoords='offset points', fontsize=9, color=c, fontweight='bold')

        ax.set_title(f"Convergencia: Ecuación {target_names[target_i]}", fontsize=14)
        ax.set_xlabel("Iteración", fontsize=12)
        ax.set_ylabel("Coeficiente", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if found_any:
            # Leyenda fuera
            ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', title="Términos (Sólido=Calc, Dashed=Real)")
        else:
            ax.text(0.5, 0.5, "No se encontraron términos teóricos", ha='center', transform=ax.transAxes)

    plt.suptitle(f"Convergencia vs Teoría\n{os.path.basename(model_path)}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_convergence(sys.argv[1])
    else:
        plot_convergence(MODEL_PATH)