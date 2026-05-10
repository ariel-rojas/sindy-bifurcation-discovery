#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualizador Personalizado de Resultados Estadísticos.

CORRECCIÓN:
- Ajustado para leer la clave "times" del JSON de configuración.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
BASE_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "tb_zone")

# Valores Teóricos (Takens-Bogdanov)
THEORETICAL_MAP = {
    0: {'y': 1.0},
    1: {'mu1': -1.0, 'x mu2': -1.0, 'x^2': 1.0, 'x^3': -1.0, 'x^2 y': -1.0, 'x y': -1.0}
}

# =============================================================================
# UTILIDADES
# =============================================================================
def normalize_term_name(name):
    return " ".join(sorted(name.split()))

def load_stats_data(zone, unbias):
    folder_name = f"statistical_sweep_analysis_zone{zone}"
    folder_path = os.path.join(BASE_OUTPUT_ROOT, folder_name)
    
    mode_str = "TRUE" if unbias else "FALSE"
    filename = f"stats_unbias_{mode_str}.json"
    file_path = os.path.join(folder_path, filename)
    
    print(f"Cargando datos desde: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        
    with open(file_path, "r") as f:
        data = json.load(f)
        
    return data, folder_path

# =============================================================================
# FUNCIÓN DE PLOTEO
# =============================================================================
def plot_custom_results(zone, unbias, show_spurious=True, show_error_bars=True):
    try:
        data_container, output_dir = load_stats_data(zone, unbias)
    except Exception as e:
        print(f"Error: {e}")
        return

    results = data_container["results"]
    config = data_container["config"]
    
    # --- CORRECCIÓN AQUÍ ---
    # Intentamos leer "times" (lo que genera el script nuevo) o "time_steps" (legacy)
    if "times" in config:
        time_steps = np.array(config["times"])
    elif "time_steps" in config:
        time_steps = np.array(config["time_steps"])
    else:
        raise KeyError("No se encontró el vector de tiempo ('times') en el JSON.")
    
    n_iters = config.get("iters", config.get("n_iters", "?"))
    mode_str = "TRUE" if unbias else "FALSE"
    
    # Configurar Gráfico
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    targets = ["x'", "y'"]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    print(f"Graficando Zona {zone} (Unbias={mode_str})...")

    for target_i in range(2):
        target_key = str(target_i)
        ax = axes[target_i]
        
        if target_key not in results:
            continue
            
        hist_dict = results[target_key]
        theo_dict = THEORETICAL_MAP[target_i]
        theo_norm = {normalize_term_name(k): v for k, v in theo_dict.items()}
        
        color_idx = 0
        
        for term, stats in hist_dict.items():
            means = np.array(stats['mean'])
            stds = np.array(stats['std'])
            
            norm_term = normalize_term_name(term)
            is_theoretical = norm_term in theo_norm
            
            # Filtros
            if not is_theoretical and not show_spurious:
                continue
            if not is_theoretical and np.max(np.abs(means)) < 0.02:
                continue

            final_mean = means[-1]
            final_std = stds[-1]
            
            # Estilos
            if is_theoretical:
                c = colors[color_idx % 10]; color_idx += 1
                label = f"{term}: {final_mean:.3f}"
                if show_error_bars: label += f" $\pm$ {final_std:.3f}"
                style = '-'; alpha_line = 1.0; lw = 2.5
            else:
                c = 'gray'
                label = f"{term} (Espurio)"
                style = ':'; alpha_line = 0.6; lw = 1.5
            
            ax.plot(time_steps, means, label=label, color=c, ls=style, lw=lw, alpha=alpha_line)
            
            if show_error_bars:
                ax.fill_between(time_steps, means - stds, means + stds, color=c, alpha=0.15)
            
            if is_theoretical:
                val_true = theo_norm[norm_term]
                ax.axhline(val_true, color=c, ls='--', alpha=0.5, lw=1)

        ax.set_title(f"Ecuación {targets[target_i]} (Zona {zone})")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Coeficiente")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize='x-small', title="Valor Final")

    plt.suptitle(f"Análisis SINDy Zona {zone} (Unbias={mode_str}, N={n_iters})", fontsize=16)
    plt.tight_layout()
    
    suffix_sp = "spurious" if show_spurious else "clean"
    suffix_err = "error" if show_error_bars else "noerror"
    filename = f"plot_custom_Z{zone}_U{mode_str}_{suffix_sp}_{suffix_err}.png"
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150)
    print(f"Gráfico guardado en: {save_path}")
    plt.close()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    
    # --- CONFIGURACIÓN ---
    TARGET_ZONE = 3    # <--- AJUSTAR ZONA
    USE_UNBIAS = True  # <--- AJUSTAR MODO
    
    # Generar gráficos
    plot_custom_results(TARGET_ZONE, USE_UNBIAS, show_spurious=True, show_error_bars=True)
    plot_custom_results(TARGET_ZONE, USE_UNBIAS, show_spurious=False, show_error_bars=True)