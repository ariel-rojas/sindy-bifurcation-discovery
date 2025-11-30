#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visor Interactivo GENÉRICO (Versión Avanzada).

Funcionalidad:
1. Mapa de Bifurcación (Click selecciona parámetros).
2. Retrato de Fases (Click selecciona trayectoria).
   - [MEJORA] Muestra estabilidad de puntos fijos (Silla/Foco).
   - [MEJORA] Ejes fijos [-1.5, 1.5].
3. Series Temporales x(t), y(t) de la trayectoria seleccionada.
4. Retrato de Fase Individual y(x) de la trayectoria seleccionada.
"""

import os
import sys
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

# --- IMPORTACIONES MODULARES ---
from systems.takens_bogdanov import TakensBogdanov as System
from core.io import make_param_key, parse_param_key

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "output")
HDF5_PATH = os.path.join(DATA_DIR, "trajectory_data.hdf5")
METADATA_PATH = os.path.join(DATA_DIR, "grid_metadata.json")

# Estética General
TRAJ_COLOR_NORMAL = "orange"
TRAJ_COLOR_SELECT = "blue"  # Color al seleccionar
TRAJ_ALPHA_NORMAL = 0.4
TRAJ_ALPHA_SELECT = 1.0
TRAJ_LW_NORMAL    = 0.5
TRAJ_LW_SELECT    = 2.0

VF_COLOR = "red"
VF_ALPHA = 0.3

# Colores de Estabilidad
FP_COLORS = {
    "Silla": "#7B1FA2",             # Violeta
    "Foco/Nodo Estable": "#2E7D32", # Verde
    "Foco/Nodo Inestable": "#C62828", # Rojo
}

# Límites fijos solicitados para el Panel 2
FIXED_LIMITS = [-1, 1]

# =============================================================================
# LÓGICA DE CARGA Y UTILIDADES
# =============================================================================
def load_heatmap_data():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("No se encontró metadata.")
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    keys = list(metadata.keys())
    parsed_params = np.array([parse_param_key(k) for k in keys])
    
    if parsed_params.size == 0:
        raise ValueError("Metadatos vacíos.")

    p0_vals = sorted(list(set(parsed_params[:, 0])))
    p1_vals = sorted(list(set(parsed_params[:, 1])))
    
    z_data = np.zeros((len(p1_vals), len(p0_vals)))
    for i, p1 in enumerate(p1_vals):
        for j, p0 in enumerate(p0_vals):
            key = make_param_key([p0, p1])
            if key in metadata:
                z_data[i, j] = metadata[key].get("num_fixed_points", 0)
            
    return p0_vals, p1_vals, z_data

def classify_stability_2d(jacobian_elems):
    """[Traza, Det] -> Tipo de Estabilidad"""
    traza, det = jacobian_elems
    if det < 0: return "Silla"
    elif traza < 0: return "Foco/Nodo Estable"
    else: return "Foco/Nodo Inestable"

# =============================================================================
# RENDERIZADO PANEL 2 (RETRATO DE FASES)
# =============================================================================
def render_portrait_interactive(ax, hf, key, t_eval):
    """
    Dibuja el retrato de fases y configura las líneas para ser 'pickeables'.
    """
    ax.clear()
    
    if key not in hf:
        ax.text(0.5, 0.5, "Datos no encontrados", ha='center')
        return

    grp = hf[key]
    
    # 1. Campo Vectorial (Fondo)
    if "vector_field" in grp:
        vf = grp["vector_field"]
        if "U" in vf:
            u, v = vf["U"][:], vf["V"][:]
            x_src, y_src = vf["x_vals"][:], vf["y_vals"][:]
            
            # Grilla robusta
            nx, ny = u.shape[1], u.shape[0]
            X, Y = np.meshgrid(
                np.linspace(x_src.min(), x_src.max(), nx),
                np.linspace(y_src.min(), y_src.max(), ny)
            )
            ax.streamplot(X, Y, u, v, color=VF_COLOR, density=0.8, 
                          linewidth=0.5, arrowsize=0.8, zorder=1)

    # 2. Trayectorias (Interactive Lines)
    if "trajectories" in grp:
        trajs = grp["trajectories"]["all_trajectories"]
        step = 2
        
        # Dibujamos las trayectorias y guardamos la referencia
        for i in range(trajs.shape[0]):
            xy = trajs[i] # (2, steps)
            
            # OJO: Picker=5 significa 5 puntos de tolerancia para el clic
            line, = ax.plot(xy[0, ::step], xy[1, ::step], 
                            color=TRAJ_COLOR_NORMAL, 
                            alpha=TRAJ_ALPHA_NORMAL, 
                            linewidth=TRAJ_LW_NORMAL,
                            picker=5, 
                            zorder=2)
            
            # Guardamos los datos crudos dentro del objeto línea para recuperarlos al hacer click
            # (Guardamos la traj completa sin downsample para los plots de detalle)
            line.raw_data_xy = xy
            line.raw_t = t_eval

    # 3. Puntos Fijos con Estabilidad
    if "fixed_points" in grp:
        fps = grp["fixed_points"][:]
        if fps.size > 0:
            for i in range(fps.shape[0]):
                fp = fps[i]
                x_c, y_c = fp[0], 0.0 # Asumimos y=0 para TB
                
                color, label = "black", "FP"
                if fps.shape[1] >= 3:
                    stab = classify_stability_2d([fp[1], fp[2]])
                    color = FP_COLORS.get(stab, "black")
                    label = stab
                
                ax.scatter(x_c, y_c, c=color, s=80, zorder=10, edgecolors='white', label=label)

    # Leyenda única
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='x-small')

    # Configuración Final
    p_vals = parse_param_key(key)
    title = f"Retrato de Fases (Click en curva)\n$\mu_1$={p_vals[0]:.4f}, $\mu_2$={p_vals[1]:.4f}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(System.state_names[0])
    ax.set_ylabel(System.state_names[1])
    
    # LÍMITES FIJOS SOLICITADOS
    ax.set_xlim(FIXED_LIMITS)
    ax.set_ylim(FIXED_LIMITS)

# =============================================================================
# RENDERIZADO PANELES DETALLE (3 y 4)
# =============================================================================
def render_details(ax_time, ax_phase_single, xy_data, t_data):
    """Dibuja los gráficos detallados de la curva seleccionada."""
    
    # Panel 3: Series Temporales
    ax_time.clear()
    ax_time.plot(t_data, xy_data[0, :], label=f"${System.state_names[0]}(t)$", color="tab:blue")
    ax_time.plot(t_data, xy_data[1, :], label=f"${System.state_names[1]}(t)$", color="tab:orange")
    ax_time.set_title("Evolución Temporal", fontsize=10)
    ax_time.set_xlabel("Tiempo (t)")
    ax_time.legend(loc="upper right", fontsize='small')
    ax_time.grid(True, alpha=0.3)
    
    # Panel 4: Fase Individual (Zoom)
    ax_phase_single.clear()
    ax_phase_single.plot(xy_data[0, :], xy_data[1, :], color=TRAJ_COLOR_SELECT, lw=1.5)
    ax_phase_single.set_title(f"Trayectoria Seleccionada ({System.state_names[1]} vs {System.state_names[0]})", fontsize=10)
    ax_phase_single.set_xlabel(System.state_names[0])
    ax_phase_single.set_ylabel(System.state_names[1])
    ax_phase_single.grid(True, alpha=0.3)
    
    # Marcar inicio y fin
    ax_phase_single.plot(xy_data[0, 0], xy_data[1, 0], 'go', label="Inicio")
    ax_phase_single.plot(xy_data[0, -1], xy_data[1, -1], 'rx', label="Fin")
    ax_phase_single.legend(fontsize='x-small')

# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.exists(HDF5_PATH):
        print(f"No se encontraron datos HDF5.")
        return

    # Cargar datos estáticos
    try:
        p0_vals, p1_vals, z_data = load_heatmap_data()
        
        # Cargar vector de tiempo una sola vez
        with h5py.File(HDF5_PATH, "r") as hf:
            if "t_eval" not in hf: raise ValueError("HDF5 corrupto")
            t_eval_global = hf["t_eval"][:]
            
    except Exception as e:
        print(f"Error inicializando: {e}")
        return
    
    # --- CONFIGURACIÓN DEL LAYOUT (2x2) ---
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1]) # Mapa y Fase más grandes arriba
    
    ax_map = fig.add_subplot(gs[0, 0])
    ax_portrait = fig.add_subplot(gs[0, 1])
    ax_time = fig.add_subplot(gs[1, 0])
    ax_single = fig.add_subplot(gs[1, 1])
    
    # --- Panel 1: Mapa de Bifurcación (Estático) ---
    cmap = plt.get_cmap("viridis", int(np.max(z_data)) + 1)
    im = ax_map.pcolormesh(p0_vals, p1_vals, z_data, cmap=cmap, shading='nearest')
    plt.colorbar(im, ax=ax_map, label="N° Puntos Fijos", fraction=0.046, pad=0.04)
    
    curves = System().get_bifurcation_curves() 
    for name, (cx, cy, color, style) in curves.items():
        ax_map.plot(cx, cy, color=color, linestyle=style, label=name, lw=2)
    
    ax_map.set_xlabel(System.param_names[0])
    ax_map.set_ylabel(System.param_names[1])
    ax_map.set_xlim(System.param_ranges[0])
    ax_map.set_ylim(System.param_ranges[1])
    ax_map.set_title(f"1. Mapa de Bifurcación: {System.name}")
    ax_map.legend(loc='lower right', fontsize='x-small')

    # Handle HDF5 (Mantener abierto)
    hf = h5py.File(HDF5_PATH, "r")

    # --- EVENTO 1: Click en MAPA -> Actualiza Panel 2 ---
    def on_click_map(event):
        if event.inaxes != ax_map or event.xdata is None: return
        
        # Buscar param más cercano
        ix = (np.abs(np.array(p0_vals) - event.xdata)).argmin()
        iy = (np.abs(np.array(p1_vals) - event.ydata)).argmin()
        key = make_param_key([p0_vals[ix], p1_vals[iy]])
        
        print(f"Mapa Click: {key}")
        render_portrait_interactive(ax_portrait, hf, key, t_eval_global)
        
        # Limpiar paneles inferiores al cambiar de parámetros
        ax_time.clear(); ax_single.clear()
        ax_time.text(0.5, 0.5, "Selecciona trayectoria arriba", ha='center', color='gray')
        ax_single.text(0.5, 0.5, "Selecciona trayectoria arriba", ha='center', color='gray')
        
        fig.canvas.draw_idle()

    # --- EVENTO 2: Click en TRAYECTORIA -> Actualiza Paneles 3 y 4 ---
    def on_pick_traj(event):
        if event.mouseevent.inaxes != ax_portrait: return
        
        # La línea seleccionada
        this_line = event.artist
        
        # 1. Resetear estilos de otras líneas en el Panel 2
        for line in ax_portrait.lines:
            # Ignorar líneas que no sean trayectorias (ej. bordes de scatter o streamplot hacks)
            if hasattr(line, 'raw_data_xy'): 
                line.set_color(TRAJ_COLOR_NORMAL)
                line.set_alpha(TRAJ_ALPHA_NORMAL)
                line.set_linewidth(TRAJ_LW_NORMAL)
        
        # 2. Resaltar selección
        this_line.set_color(TRAJ_COLOR_SELECT)
        this_line.set_alpha(TRAJ_ALPHA_SELECT)
        this_line.set_linewidth(TRAJ_LW_SELECT)
        
        # 3. Recuperar datos y dibujar detalles
        if hasattr(this_line, 'raw_data_xy'):
            xy = this_line.raw_data_xy
            t = this_line.raw_t
            print(f"Trayectoria seleccionada. Puntos: {xy.shape[1]}")
            render_details(ax_time, ax_single, xy, t)
        
        fig.canvas.draw_idle()

    # Conectar eventos
    fig.canvas.mpl_connect('button_press_event', on_click_map)
    fig.canvas.mpl_connect('pick_event', on_pick_traj)
    
    # Textos iniciales
    ax_portrait.text(0.5, 0.5, "Haz clic en el Mapa (1)", ha='center', color='gray')
    ax_time.text(0.5, 0.5, "Haz clic en una Curva (2)", ha='center', color='gray')
    ax_single.text(0.5, 0.5, "Detalle de Curva", ha='center', color='gray')
    
    # Quitar ejes vacíos iniciales
    for ax in [ax_portrait, ax_time, ax_single]:
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(h_pad=3.0, rect=[0, 0.03, 1, 0.95]) 

    plt.show()
    
    hf.close()
    print("Cerrando aplicación.")
    
if __name__ == "__main__":
    main()