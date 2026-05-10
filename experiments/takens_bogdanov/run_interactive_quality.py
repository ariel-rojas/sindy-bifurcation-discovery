#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visor Interactivo de CALIDAD DE DATOS (Dashboard Dinámico).
"""

import os
import sys
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RadioButtons, Slider

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

from core.io import make_param_key, parse_param_key
from core.metrics import calculate_velocity_norm
from systems.takens_bogdanov import TakensBogdanov as System

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "output")
HDF5_FILE = os.path.join(DATA_DIR, "trajectory_data.hdf5")
METRICS_JSON = os.path.join(DATA_DIR, "data_quality_metrics.json")
METADATA_PATH = os.path.join(DATA_DIR, "grid_metadata.json")

METRIC_CONFIG = {
    "t_eff": {"label": "T_eff (Convergencia)", "cmap": "viridis", "dynamic": False},
    "t_eff_std": {"label": "Std Dev T_eff", "cmap": "magma", "dynamic": False},
    "mean_velocity": {"label": "Velocidad Media (Dinámico)", "cmap": "cividis", "dynamic": True}
}

FP_COLORS = {
    "Silla": "#7B1FA2",             
    "Foco/Nodo Estable": "#2E7D32", 
    "Foco/Nodo Inestable": "#C62828", 
    "Desconocido": "gray"
}

TRAJ_COLOR_BG = "gray"
TRAJ_ALPHA_BG = 0.15
TRAJ_LW_BG    = 0.5

# =============================================================================
# LÓGICA DE CARGA
# =============================================================================
def load_bifurcation_data():
    if not os.path.exists(METADATA_PATH): raise FileNotFoundError("Falta metadata.")
    with open(METADATA_PATH, "r") as f: metadata = json.load(f)

    keys = list(metadata.keys())
    parsed = np.array([parse_param_key(k) for k in keys])
    p0 = sorted(list(set(parsed[:, 0])))
    p1 = sorted(list(set(parsed[:, 1])))
    
    z = np.zeros((len(p1), len(p0)))
    for i, y in enumerate(p1):
        for j, x in enumerate(p0):
            k = make_param_key([x, y])
            if k in metadata: z[i, j] = metadata[k].get("num_fixed_points", 0)
    return p0, p1, z

def load_quality_metrics():
    if not os.path.exists(METRICS_JSON): return None, None, None
    with open(METRICS_JSON, "r") as f: data = json.load(f)
        
    metrics_map = data["metrics"]
    meta = data.get("meta", {})
    n_frames = meta.get("dynamic_frames", 1) # Default 1 si no existe

    keys = list(metrics_map.keys())
    parsed = np.array([parse_param_key(k) for k in keys])
    p0 = sorted(list(set(parsed[:, 0])))
    p1 = sorted(list(set(parsed[:, 1])))
    
    grids = {}
    
    # Inicializar grids. Si es dinámico, hacemos array 3D.
    for m, cfg in METRIC_CONFIG.items():
        if cfg["dynamic"]:
            grids[m] = np.full((len(p1), len(p0), n_frames), np.nan)
        else:
            grids[m] = np.full((len(p1), len(p0)), np.nan)
        
    for i, y in enumerate(p1):
        for j, x in enumerate(p0):
            k = make_param_key([x, y])
            if k in metrics_map and metrics_map[k]:
                for m in METRIC_CONFIG:
                    val = metrics_map[k].get(m, np.nan)
                    
                    # Manejo de listas vs escalares
                    if isinstance(val, list):
                        grids[m][i, j, :] = val # Rellenar vector de profundidad
                    else:
                        grids[m][i, j] = val
                        
    return p0, p1, grids

def classify_stability_2d(jacobian_elems):
    traza, det = jacobian_elems
    if det < 0: return "Silla"
    elif traza < 0: return "Foco/Nodo Estable"
    else: return "Foco/Nodo Inestable"

# =============================================================================
# VISUALIZADOR
# =============================================================================
class InteractiveDashboard:
    def __init__(self):
        try:
            self.p0, self.p1, self.z_bif = load_bifurcation_data()
            _, _, self.grids_q = load_quality_metrics()
            self.hf = h5py.File(HDF5_FILE, "r")
            self.t_eval = self.hf["t_eval"][:]
            self.dt = self.t_eval[1] - self.t_eval[0]
            self.t_max_absolute = self.t_eval[-1]
        except Exception as e:
            print(f"Error inicializando: {e}")
            return

        self.current_metric = "t_eff"
        self.current_t_max = self.t_max_absolute
        
        self.selected_key = None
        self.sel_ix = None
        self.sel_iy = None
        self.sample_trajs_cache = None 
        
        # Setup Figura
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.25)
        
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        
        plt.subplots_adjust(left=0.15, bottom=0.15) 
        
        # Panel 1
        cmap1 = plt.get_cmap("viridis", int(np.nanmax(self.z_bif)) + 1)
        self.ax1.pcolormesh(self.p0, self.p1, self.z_bif, cmap=cmap1, shading='nearest')
        
        curves = System().get_bifurcation_curves()
        for name, (cx, cy, c, s) in curves.items():
            self.ax1.plot(cx, cy, color=c, linestyle=s, label=name, lw=1.5)
            
        self.ax1.set_title("1. Diagrama de Bifurcación")
        self.ax1.set_xlabel(System.param_names[0])
        self.ax1.set_ylabel(System.param_names[1])
        self.ax1.set_xlim(System.param_ranges[0])
        self.ax1.set_ylim(System.param_ranges[1])
        self.ax1.legend(loc='lower right', fontsize='x-small')
        
        self.mark1, = self.ax1.plot([], [], 'kx', ms=12, mew=2)
        self.mark2, = self.ax2.plot([], [], 'kx', ms=12, mew=2)
        
        # Panel 2
        self.im_q = None
        self.cbar_q = None
        self.text_val = self.ax2.text(0.02, 0.98, "", transform=self.ax2.transAxes, 
                                      va='top', ha='left', fontsize=10,
                                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        self.update_quality_map()
        
        # Widgets
        ax_slider = plt.axes([0.25, 0.05, 0.50, 0.03])
        self.slider = Slider(ax_slider, 'Recorte T', 0.1, self.t_max_absolute, valinit=self.t_max_absolute)
        self.slider.on_changed(self.on_slider_change)

        ax_radio = plt.axes([0.02, 0.5, 0.10, 0.15], facecolor='#f0f0f0')
        self.radio = RadioButtons(ax_radio, list(METRIC_CONFIG.keys()))
        self.radio.on_clicked(self.change_metric)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_map)
        
        self.ax3.text(0.5, 0.5, "Selecciona un punto arriba", ha='center', color='gray')
        self.ax4.text(0.5, 0.5, "Selecciona un punto arriba", ha='center', color='gray')
        self.ax3.set_xticks([]); self.ax3.set_yticks([])
        self.ax4.set_xticks([]); self.ax4.set_yticks([])
        
        plt.suptitle(f"Dashboard de Calidad: {System.name}", fontsize=16)
        plt.show()

    def update_quality_map(self):
        """Actualiza el mapa de calor. Maneja métricas 2D (estáticas) y 3D (dinámicas)."""
        if self.cbar_q:
            try: self.cbar_q.remove()
            except Exception: pass
        self.ax2.clear()
        
        if self.grids_q is None:
            self.ax2.text(0.5, 0.5, "No hay métricas (.json)", ha='center')
            return

        grid_data = self.grids_q[self.current_metric]
        cfg = METRIC_CONFIG[self.current_metric]
        
        # LÓGICA DINÁMICA: Si el grid es 3D, sacamos la rebanada correspondiente al tiempo actual
        data_to_plot = grid_data
        
        if grid_data.ndim == 3:
            n_frames = grid_data.shape[2]
            # Índice basado en el slider (0 a 1 -> 0 a n_frames-1)
            # current_t_max va de 0 a t_max_absolute
            fraction = self.current_t_max / self.t_max_absolute
            frame_idx = int(fraction * (n_frames - 1))
            frame_idx = max(0, min(frame_idx, n_frames - 1))
            
            data_to_plot = grid_data[:, :, frame_idx]
            title_suffix = f" (T={self.current_t_max:.1f})"
        else:
            title_suffix = ""
        
        self.im_q = self.ax2.pcolormesh(self.p0, self.p1, data_to_plot, cmap=cfg["cmap"], shading='nearest')
        self.ax2.set_title(f"2. Mapa: {cfg['label']}{title_suffix}")
        self.ax2.set_xlabel(System.param_names[0])
        self.ax2.set_yticklabels([]) 
        
        self.cbar_q = self.fig.colorbar(self.im_q, ax=self.ax2, fraction=0.046)
        self.ax2.set_xlim(System.param_ranges[0])
        self.ax2.set_ylim(System.param_ranges[1])
        
        # Recrear el texto de valor (se borró con clear)
        self.text_val = self.ax2.text(0.02, 0.98, "", transform=self.ax2.transAxes, 
                                      va='top', ha='left', fontsize=10,
                                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Restaurar marcas de selección si existen
        if self.sel_ix is not None:
            val0 = self.p0[self.sel_ix]
            val1 = self.p1[self.sel_iy]
            self.mark2, = self.ax2.plot(val0, val1, 'kx', ms=12, mew=2)
            
            # Actualizar valor numérico en el texto
            val = data_to_plot[self.sel_iy, self.sel_ix]
            self.text_val.set_text(f"Val: {val:.4f}")
            
        self.fig.canvas.draw_idle()

    def change_metric(self, label):
        self.current_metric = label
        self.update_quality_map()
        # Si hay selección activa, forzar actualización del texto dinámico
        if self.selected_key:
            self.update_lower_panels()

    def update_lower_panels(self):
        """Actualiza histograma y texto dinámico."""
        if not self.selected_key or self.sample_trajs_cache is None: return
        
        idx_max = int(self.current_t_max / self.dt)
        if idx_max < 2: idx_max = 2
        
        # --- Panel 3: Retrato ---
        self.ax3.clear()
        trajs = self.sample_trajs_cache
        
        for i in range(trajs.shape[0]):
            xy = trajs[i]
            self.ax3.plot(xy[0, :idx_max:4], xy[1, :idx_max:4], 
                          color=TRAJ_COLOR_BG, alpha=TRAJ_ALPHA_BG, lw=TRAJ_LW_BG)
        
        if self.selected_key in self.hf:
            grp = self.hf[self.selected_key]
            if "fixed_points" in grp:
                fps = grp["fixed_points"][:]
                for i in range(fps.shape[0]):
                    fp = fps[i]
                    color = FP_COLORS.get(classify_stability_2d([fp[1], fp[2]]), "black")
                    self.ax3.scatter(fp[0], 0.0, c=color, s=80, zorder=10, edgecolors='w')

        self.ax3.set_title(f"3. Fase (t=0..{self.current_t_max:.1f}s)")
        lims = System.state_limits
        self.ax3.set_xlim(lims[0])
        self.ax3.set_ylim(lims[1])
        self.ax3.set_xlabel("x")
        self.ax3.set_ylabel("y")

        # --- Panel 4: Histograma Dinámico ---
        self.ax4.clear()
        trajs_cut = trajs[:, :, :idx_max]
        
        all_vels = []
        for i in range(trajs_cut.shape[0]):
            v = calculate_velocity_norm(trajs_cut[i], self.dt)
            all_vels.append(v)
        
        mean_vel_dynamic = 0.0
        if all_vels:
            all_vels_flat = np.concatenate(all_vels)
            self.ax4.hist(all_vels_flat, bins=50, color='purple', alpha=0.7, log=True)
            mean_vel_dynamic = np.mean(all_vels_flat)
            
        self.ax4.set_title(f"4. Vel. ||v|| (t=0..{self.current_t_max:.1f}s)")
        self.ax4.set_xlabel("||v||")
        self.ax4.set_ylabel("Log Frecuencia")
        
        # --- Actualizar Texto Valor Dinámico (Panel 2) ---
        # Si la métrica activa es mean_velocity, el texto coincide con el mapa dinámico
        if self.current_metric == "mean_velocity":
             self.text_val.set_text(f"Val (T={self.current_t_max:.1f}): {mean_vel_dynamic:.4f}")
        
        self.fig.canvas.draw_idle()

    def on_click_map(self, event):
        if event.inaxes not in [self.ax1, self.ax2]: return
        
        ix = (np.abs(np.array(self.p0) - event.xdata)).argmin()
        iy = (np.abs(np.array(self.p1) - event.ydata)).argmin()
        
        self.sel_ix = ix
        self.sel_iy = iy
        
        val0 = self.p0[ix]
        val1 = self.p1[iy]
        key = make_param_key([val0, val1])
        self.selected_key = key
        
        self.mark1.set_data([val0], [val1])
        if self.mark2 not in self.ax2.lines:
             self.mark2, = self.ax2.plot([val0], [val1], 'kx', ms=12, mew=2)
        else:
             self.mark2.set_data([val0], [val1])
        
        # Actualizar texto del Panel 2 inmediatamente
        # Obtenemos el valor actual del grid (sea 2D o slice 3D)
        grid_data = self.grids_q[self.current_metric]
        if grid_data.ndim == 3:
            n_frames = grid_data.shape[2]
            fraction = self.current_t_max / self.t_max_absolute
            frame_idx = int(fraction * (n_frames - 1))
            val = grid_data[iy, ix, frame_idx]
        else:
            val = grid_data[iy, ix]
            
        self.text_val.set_text(f"Val: {val:.4f}")
        
        print(f"Seleccionado: {key}")
        
        if key in self.hf:
            grp = self.hf[key]
            if "trajectories" in grp:
                all_trajs = grp["trajectories"]["all_trajectories"][:]
                self.sample_trajs_cache = all_trajs 
                self.update_lower_panels()
        else:
            print("Datos no encontrados en HDF5.")
            
        self.fig.canvas.draw_idle()

    def on_slider_change(self, val):
        self.current_t_max = val
        
        # SI la métrica actual es dinámica (Velocidad), actualizamos el Mapa entero
        if METRIC_CONFIG[self.current_metric]["dynamic"]:
            self.update_quality_map()
            
        # Siempre actualizamos los paneles inferiores si hay selección
        if self.selected_key:
            self.update_lower_panels()

if __name__ == "__main__":
    InteractiveDashboard()