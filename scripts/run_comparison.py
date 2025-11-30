#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparador GENÉRICO (Ground Truth vs SINDy).
Reemplaza a compare_plots.py.

MEJORAS VISUALES RESTAURADAS:
- Dibuja TODAS las trayectorias (sin límite de 50).
- Clasifica puntos fijos (Silla/Nodo/Foco) si el sistema es 2D.
- Muestra leyenda detallada de estabilidad.
- Usa linewidth en streamplot (fix matplotlib).
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

# --- FIX DE IMPORTACIONES Y RUTAS ---
# 1. Identificar la raíz del proyecto (un nivel arriba de esta carpeta 'scripts')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# 2. Añadir la raíz al path de Python para poder importar 'core' y 'systems'
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# --- IMPORTACIONES MODULARES ---
from systems.takens_bogdanov import TakensBogdanov as System
from core.io import parse_param_key, make_param_key

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DATA_DIR = "output/5/15"
CARPETA_FILES = os.path.join(PROJECT_ROOT, DATA_DIR)
GT_HDF5 = os.path.join(os.path.join(PROJECT_ROOT, "output"), "trajectory_data.hdf5")
SINDY_HDF5 = os.path.join(CARPETA_FILES, "sindy_simulations.hdf5")
MODEL_PATH = os.path.join(CARPETA_FILES, "sindy_model.joblib")
TRAIN_PARAMS = os.path.join(CARPETA_FILES, "sindy_training_params.json")

# Colores específicos para estabilidad
FP_COLORS = {
    "Silla": "#7B1FA2",             # Violeta
    "Foco/Nodo Estable": "#2E7D32", # Verde
    "Foco/Nodo Inestable": "#C62828", # Rojo
    "Desconocido": "gray"
}

# =============================================================================
# UTILIDADES
# =============================================================================
def print_model_equations(model):
    print("\n" + "="*60)
    print(f"--- Modelo SINDy Descubierto ({System.name}) ---")
    
    # 1. Obtener datos crudos
    coeffs = model.coefficients()
    feature_names = model.get_feature_names()
    
    # 2. Iterar por cada ecuación (x', y', etc.)
    for i in range(coeffs.shape[0]):
        # Obtener nombre de la variable de estado (Lado Izquierdo)
        try:
            lhs = System.state_names[i]
        except IndexError:
            lhs = f"x{i}"
            
        rhs_terms = []
        
        # 3. Iterar por cada término (Lado Derecho)
        for j in range(coeffs.shape[1]):
            val = coeffs[i, j]
            
            # Solo mostramos términos distintos de cero (con un umbral mínimo de seguridad)
            if abs(val) > 1e-5:
                # Formato de alta precisión (ej: .8f para 8 decimales)
                # El espacio inicial en el f-string ayuda a separar signos luego
                term = f"{val:+} {feature_names[j]}"
                rhs_terms.append(term)
        
        # 4. Ensamblar la ecuación
        if not rhs_terms:
            equation_str = "0.0"
        else:
            # Unir términos
            equation_str = " ".join(rhs_terms)
            
            # Limpieza estética: 
            # El primer término no necesita el signo '+' si es positivo, 
            # pero para mantener la alineación a veces es útil dejarlo.
            # Si prefieres quitar el '+' inicial si existe:
            if equation_str.strip().startswith("+"):
                equation_str = equation_str.strip()[1:].strip()
        
        print(f"({lhs})' = {equation_str}")
        
    print("="*60 + "\n")
def get_points_from_keys(keys):
    """
    Convierte lista de strings 'k1_k2' a array numpy (N, 2).
    Filtra claves vacías o inválidas.
    """
    valid_points = []
    for k in keys:
        try:
            p = parse_param_key(k)
            valid_points.append(p)
        except Exception:
            continue
            
    if not valid_points:
        return np.empty((0, 2))
    return np.array(valid_points)

def classify_stability_2d(jacobian_elems):
    """Clasifica estabilidad para sistemas 2D."""
    traza, det = jacobian_elems
    if det < 0:
        return "Silla"
    elif traza < 0:
        return "Foco/Nodo Estable"
    else:
        return "Foco/Nodo Inestable"

def render_panel(ax, hf, key, title, traj_color):
    ax.clear()
    
    # 1. OBTENER LÍMITES FIJOS DEL SISTEMA (CRUCIAL)
    x_lims = System.state_limits[0]
    y_lims = System.state_limits[1]
    
    if key not in hf:
        ax.text(0.5, 0.5, "Sin datos", ha='center')
        ax.set_title(title)
        # Aun sin datos, fijamos ejes para mantener la estética
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        return

    grp = hf[key]
    
    # 2. Campo Vectorial (Fondo)
    if "vector_field" in grp:
        vf = grp["vector_field"]
        if "U" in vf:
            u = vf["U"][:]
            v = vf["V"][:]
            x_src = vf["x_vals"][:]
            y_src = vf["y_vals"][:]
            
            nx, ny = u.shape[1], u.shape[0]
            x_grid = np.linspace(x_src.min(), x_src.max(), nx)
            y_grid = np.linspace(y_src.min(), y_src.max(), ny)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            ax.streamplot(X, Y, u, v, color='k', density=0.8, linewidth=0.4, arrowsize=0.6)

    # 3. Trayectorias (TODAS)
    if "trajectories" in grp:
        trajs = grp["trajectories"]["all_trajectories"]
        time_step = 2 
        ax.set_autoscale_on(False) # Importante: No dejar que matplotlib cambie zoom
        for i in range(trajs.shape[0]):
            xy = trajs[i]
            ax.plot(xy[0, ::time_step], xy[1, ::time_step], 
                    color=traj_color, alpha=0.3, linewidth=0.4)

    # 4. Puntos Fijos
    if "fixed_points" in grp:
        fps = grp["fixed_points"][:] 
        if fps.size > 0:
            for i in range(fps.shape[0]):
                fp = fps[i]
                x_coord = fp[0]
                y_coord = 0.0 
                label_txt = f"FP x={x_coord:.2f}"
                color = "black"
                if fps.shape[1] >= 3: 
                    traza = fp[1]
                    det = fp[2]
                    stab_type = classify_stability_2d([traza, det])
                    color = FP_COLORS.get(stab_type, "black")
                    label_txt = f"{stab_type}"
                
                ax.scatter(x_coord, y_coord, c=color, s=60, zorder=20, 
                           edgecolors='white', linewidth=1.5, label=label_txt)

    # 5. FORZAR LÍMITES AL FINAL
    # Esto sobrescribe cualquier auto-ajuste que haya hecho streamplot o plot
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    # Leyenda
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='x-small', framealpha=0.9)

    try:
        params = parse_param_key(key)
        sub_title = f"$\mu_1$={params[0]:.4f}, $\mu_2$={params[1]:.4f}"
        ax.set_title(f"{title}\n{sub_title}", fontsize=11)
    except:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel(System.state_names[0])
# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    print_model_equations(model)
    
    # --- Cargar claves de ENTRENAMIENTO ---
    train_keys = []
    if os.path.exists(TRAIN_PARAMS):
        with open(TRAIN_PARAMS, 'r') as f:
            d = json.load(f)
            # Intentamos obtener las claves usadas
            train_keys = d.get("keys_used", [])
            if not train_keys:
                # Fallback para compatibilidad con versiones viejas del JSON
                train_keys = d.get("training_keys_mu1_mu2", [])
    else:
        print(f"Advertencia: No se encontró {TRAIN_PARAMS}. Puntos de entrenamiento no visibles.")

    # --- Cargar claves de VALIDACIÓN ---
    valid_keys = []
    if os.path.exists(SINDY_HDF5):
        with h5py.File(SINDY_HDF5, 'r') as f:
            valid_keys = [k for k in f.keys() if k != "t_eval"]
            
    # Convertir y depurar
    pts_train = get_points_from_keys(train_keys)
    pts_valid = get_points_from_keys(valid_keys)
    
    print(f"Puntos de Entrenamiento cargados: {len(pts_train)}")
    print(f"Puntos de Validación cargados: {len(pts_valid)}")
    
    # Setup Gráfico
    fig = plt.figure(figsize=(19, 7))
    
    ax_map = plt.subplot(1, 3, 1)
    ax_gt = plt.subplot(1, 3, 2)
    ax_sindy = plt.subplot(1, 3, 3, sharex=ax_gt, sharey=ax_gt)
    
    # --- Panel 1: Mapa ---
    curves = System().get_bifurcation_curves()
    for name, (cx, cy, c, s) in curves.items():
        ax_map.plot(cx, cy, color=c, linestyle=s, label=name, lw=2, alpha=0.8)
        
    # Dibujar puntos (con zorder alto para que queden encima)
    if pts_train.size > 0:
        ax_map.scatter(pts_train[:, 0], pts_train[:, 1], c='red', label='Entrenamiento', s=60, zorder=10, picker=True, edgecolors='k')
    
    if pts_valid.size > 0:
        ax_map.scatter(pts_valid[:, 0], pts_valid[:, 1], c='blue', marker='s', label='Validación', s=30, alpha=0.6, zorder=9, picker=True)
    
    ax_map.legend(loc='lower left', fontsize='small')
    ax_map.set_title(f"Mapa de Cobertura: {System.name}", fontsize=12)
    ax_map.set_xlabel(System.param_names[0])
    ax_map.set_ylabel(System.param_names[1])
    
    p_ranges = System.param_ranges
    ax_map.set_xlim(p_ranges[0])
    ax_map.set_ylim(p_ranges[1])
    ax_map.grid(True, linestyle=':', alpha=0.5)
    
    # --- Interacción ---
    hf_gt = h5py.File(GT_HDF5, 'r')
    hf_sindy = h5py.File(SINDY_HDF5, 'r') if os.path.exists(SINDY_HDF5) else None
    
    def on_pick(event):
        if event.artist not in ax_map.collections: return
        
        ind = event.ind[0]
        data = event.artist.get_offsets()
        p0, p1 = data[ind]
        
        key_str = make_param_key([p0, p1])
        print(f"Seleccionado: {key_str}")
        
        render_panel(ax_gt, hf_gt, key_str, "Ground Truth", "tomato")
        if hf_sindy:
            render_panel(ax_sindy, hf_sindy, key_str, "SINDy Model", "royalblue")
        else:
            ax_sindy.text(0.5, 0.5, "Falta archivo SINDy", ha='center')
        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    
    # Textos iniciales
    ax_gt.text(0.5, 0.5, "Haz clic en un punto\ndel mapa", ha='center', va='center', color='gray')
    ax_sindy.text(0.5, 0.5, "Haz clic en un punto\ndel mapa", ha='center', va='center', color='gray')
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    ax_sindy.set_xticks([]); ax_sindy.set_yticks([])

    plt.tight_layout()
    plt.show()
    
    hf_gt.close()
    if hf_sindy: hf_sindy.close()

if __name__ == "__main__":
    main()