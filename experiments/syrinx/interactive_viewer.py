#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Syrinx Interactive Viewer v2.6 - "The Oracle"
---------------------------------------------
Visor interactivo dimensional Cloud-Native.
- Resiliencia: Manejo de paginación de Drive (soporta >100 archivos).
- Precisión: Búsqueda de puntos mediante distancia normalizada.
- Estabilidad: Arquitectura In-Memory (sin caché local) para datos en tiempo real.
"""

import io
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba

# =============================================================================
# RUTAS E IMPORTACIONES
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from systems.syrinx import SyrinxModel as System
from core.io import make_param_key, parse_param_key
from core.drive_auth import get_drive_service, get_target_folder_id

# Configuración Visual
TRAJ_STYLE = {"color": "orange", "alpha": 0.4, "lw": 0.7, "zorder": 2}
SELECT_STYLE = {"color": "blue", "alpha": 1.0, "lw": 2.2, "zorder": 5}
FP_COLORS = {
    "Silla": "#7B1FA2", 
    "Foco/Nodo Estable": "#2E7D32", 
    "Foco/Nodo Inestable": "#C62828", 
    "Degenerado": "gray"
}

# =============================================================================
# DRIVE UTILS (SIN CACHÉ, DIRECTO A MEMORIA)
# =============================================================================

def get_drive_index_full(session, folder_id):
    """Obtiene todos los archivos de la carpeta manejando la paginación usando REST API."""
    idx = {}
    token = None
    url = "https://www.googleapis.com/drive/v3/files"
    
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken, files(id, name)",
            "pageSize": 1000
        }
        if token:
            params["pageToken"] = token
            
        response = session.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        
        for f in results.get("files", []):
            idx[f["name"]] = f["id"]
            
        token = results.get("nextPageToken")
        if not token: 
            break
            
    return idx

def download_bytes(session, file_id):
    """Descarga el contenido del archivo directamente a la RAM como bytes."""
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    response = session.get(url)
    response.raise_for_status()
    return response.content

# =============================================================================
# CLASE PRINCIPAL DEL VISOR
# =============================================================================

class InteractiveViewer:
    def __init__(self, service, drive_idx, metadata, t_eval, traj_folder_id):
        self.service = service
        self.idx = drive_idx
        self.meta = metadata
        self.t_eval = t_eval
        
        # 1. Reconstrucción de la grilla de parámetros
        self.points_info = self._get_mesh_data()
        if self.points_info is None:
            print("❌ No se encontraron datos válidos en la metadata."); return

        # 2. Configuración de la Figura
        self.fig = plt.figure(figsize=(15, 9))
        self.gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 0.8], hspace=0.3, wspace=0.2)
        
        self.ax_map = self.fig.add_subplot(self.gs[0, 0])
        self.ax_phase = self.fig.add_subplot(self.gs[0, 1])
        self.ax_time = self.fig.add_subplot(self.gs[1, 0])
        self.ax_single = self.fig.add_subplot(self.gs[1, 1])
        
        self.selection_marker, = self.ax_map.plot([], [], 'kx', ms=12, mew=2.5, zorder=50)
        
        # 3. Renderizado Inicial
        self._render_map()
        self._init_empty_axes()
        
        # 4. Conexión de Eventos
        self.fig.canvas.mpl_connect("button_press_event", self.on_click_map)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick_trajectory)
        
        plt.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.06)
        self._print_base_ode_equation()
        print("✅ Visor listo. Haz clic en el mapa de parámetros.")

    def _get_mesh_data(self):
        points = []
        for k in self.meta.keys():
            if k.startswith("_") or f"{k}.npz" not in self.idx: continue
            points.append(parse_param_key(k))
        if not points: return None
        
        pts = np.array(points)
        p0_axis = np.unique(pts[:, 0]) # kappa1
        p1_axis = np.unique(pts[:, 1]) # psub
        
        z = np.full((len(p1_axis), len(p0_axis)), np.nan)
        for p in pts:
            i0 = np.where(p0_axis == p[0])[0][0]
            i1 = np.where(p1_axis == p[1])[0][0]
            z[i1, i0] = self.meta[make_param_key(p)].get("num_fixed_points", 0)
            
        return {"p0": p0_axis, "p1": p1_axis, "Z": z, "raw_points": pts}

    def _render_map(self):
        info = self.points_info
        # Heatmap de puntos fijos
        mesh = self.ax_map.pcolormesh(info["p1"], info["p0"], info["Z"].T, 
                                      cmap="viridis", shading="nearest", alpha=0.7)
        plt.colorbar(mesh, ax=self.ax_map, label="N° Puntos Fijos")

        # Curvas Teóricas
        sys_info = self.meta.get("_system_info", {})
        phys_params = sys_info.get("physical_base_params", {})
        if phys_params:
            try:
                curves = System.get_physical_bifurcation_curves(phys_params)
                for name, d in curves.items():
                    self.ax_map.plot(d["psub"], d["kappa1"], color=d["color"], 
                                     ls=d["linestyle"], lw=2, label=name)
                self.ax_map.legend(loc="upper left", fontsize="x-small")
            except Exception as e:
                print(f"Omitiendo curvas teóricas: {e}")

        # Forzar límites a la región simulada
        sys_info = self.meta.get("_system_info", {})
        sweep_ranges = sys_info.get("sweep_ranges")

        if sweep_ranges:
            # sweep_ranges[0] es kappa1 (Y), sweep_ranges[1] es psub (X)
            k1_range = sweep_ranges[0]
            psub_range = sweep_ranges[1]
            self.ax_map.set_xlim(psub_range[0], psub_range[1])
            self.ax_map.set_ylim(k1_range[0], k1_range[1])
        else:
            # Fallback seguro por si se carga un JSON viejo
            self.ax_map.set_xlim(np.min(info["p1"]), np.max(info["p1"]))
            self.ax_map.set_ylim(np.min(info["p0"]), np.max(info["p0"]))

        self.ax_map.set_title("1. Mapa de Bifurcación (Selecciona un punto)")
        self.ax_map.set_xlabel(r"Presión Subglótica $p_{sub}$ (dyn/cm$^2$)")
        self.ax_map.set_ylabel(r"Rigidez $\kappa_1$ (dyn/cm)")

    def _init_empty_axes(self):
        for ax in [self.ax_phase, self.ax_time, self.ax_single]:
            ax.text(0.5, 0.5, "Esperando selección...", ha='center', va='center', color='gray')
            ax.set_xticks([]); ax.set_yticks([])

    def on_click_map(self, event):
        if event.inaxes != self.ax_map or event.xdata is None: return
        
        # Búsqueda por Distancia Normalizada
        pts = self.points_info["raw_points"]
        norm_x = (pts[:, 1] - event.xdata) / (np.max(pts[:, 1]) - np.min(pts[:, 1]))
        norm_y = (pts[:, 0] - event.ydata) / (np.max(pts[:, 0]) - np.min(pts[:, 0]))
        
        idx_closest = np.argmin(norm_x**2 + norm_y**2)
        best_p = pts[idx_closest]
        
        # Actualizar marcador y cargar
        self.selection_marker.set_data([best_p[1]], [best_p[0]])
        self._load_data_point(best_p)
        self.fig.canvas.draw_idle()

    def _load_data_point(self, p):
        key = make_param_key(p)
        file_name = f"{key}.npz"
        print(f"Cargando desde la nube: {file_name}")
        
        if file_name not in self.idx:
            self.ax_phase.clear()
            self.ax_phase.text(0.5, 0.5, f"Falta archivo:\n{file_name}", ha='center', color='red')
            return

        # Carga In-Memory (Sin disco)
        raw_bytes = download_bytes(self.service, self.idx[file_name])
        with np.load(io.BytesIO(raw_bytes), allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}
        
        self._render_phase_portrait(data, p)

    def _render_phase_portrait(self, data, p):
        ax = self.ax_phase
        ax.clear()
        
        # 1. Campo Vectorial
        if "U" in data:
            X, Y = np.meshgrid(data["x_vals"], data["y_vals"])
            ax.streamplot(X, Y, data["U"], data["V"], 
                          color=to_rgba("firebrick", 0.6), 
                          density=0.7, linewidth=0.5)

        # 2. Trayectorias
        if "trajectories" in data:
            for tr in data["trajectories"]:
                if tr.shape[1] < 2: continue
                line, = ax.plot(tr[0, :], tr[1, :], picker=True, pickradius=5, **TRAJ_STYLE)
                line.full_data = tr 

        # 3. Puntos Fijos
        if "fixed_points" in data:
            for fp in data["fixed_points"]:
                stab = self._classify_stability(fp[2], fp[3])
                ax.scatter(fp[0], fp[1], c=FP_COLORS.get(stab, "black"), 
                           s=100, edgecolors="white", zorder=10, label=stab)
        
        # 4. Línea de Singularidad
        phys = self.meta["_system_info"]["physical_base_params"]
        xl_plot = np.array([np.min(data["x_vals"]), np.max(data["x_vals"])])
        yl_sing = (-phys["a01"] - xl_plot) / phys["tau"]
        ax.plot(xl_plot, yl_sing, 'r:', lw=1.5, alpha=0.4, label="Singularidad")

        ax.set_ylim(np.min(data["y_vals"]), np.max(data["y_vals"]))
        ax.set_title(fr"Retrato: $\kappa_1$={p[0]:.4f}, $p_{{sub}}$={p[1]:.1f}")
        ax.set_xlabel("x (cm)"); ax.set_ylabel("v (cm/s)")
        ax.grid(True, alpha=0.2)

    def on_pick_trajectory(self, event):
        line = event.artist
        # Reset de otros estilos
        for l in self.ax_phase.get_lines():
            if hasattr(l, "full_data"): l.set(**TRAJ_STYLE)
        
        line.set(**SELECT_STYLE)
        tr = line.full_data
        t_vec = self.t_eval[:tr.shape[1]]

        for artist in list(self.ax_phase.collections):
            if getattr(artist, 'is_trajectory_marker', False):
                artist.remove()

        # Dibujar punto de inicio (t=0)
        start_pt = self.ax_phase.scatter(tr[0, 0], tr[1, 0], c='green', s=60, 
                                        zorder=6, label='Inicio')
        start_pt.is_trajectory_marker = True

        # Dibujar punto de fin (t=final)
        end_pt = self.ax_phase.scatter(tr[0, -1], tr[1, -1], marker='x', c='red', 
                                      s=80, zorder=6, label='Fin')
        end_pt.is_trajectory_marker = True
        
        # Detalle Temporal
        self.ax_time.clear()
        self.ax_time.plot(t_vec, tr[0, :], label="Posición (x)", color="tab:blue")
        self.ax_time.plot(t_vec, tr[1, :], label="Velocidad (v)", color="tab:orange", alpha=0.7)
        self.ax_time.set_title("Evolución Temporal")
        self.ax_time.set_xlabel("Tiempo (s)"); self.ax_time.legend(fontsize="x-small")
        self.ax_time.grid(True, alpha=0.3)

        self.ax_single.clear()
        self.ax_single.plot(tr[0, :], tr[1, :], **SELECT_STYLE)
        self.ax_single.scatter(tr[0, 0], tr[1, 0], c='green', s=60, zorder=6, label='Inicio')
        self.ax_single.scatter(tr[0, -1], tr[1, -1], marker='x', c='red', s=80, zorder=6, label='Fin')
        
        self.ax_single.set_title("Órbita Seleccionada")
        self.ax_single.set_xlabel("x (cm)")
        self.ax_single.set_ylabel("v (cm/s)")
        self.ax_single.grid(True, alpha=0.3)
        self.ax_single.legend(fontsize="x-small") 
        
        self.fig.canvas.draw_idle()

    def _classify_stability(self, tr, det):
        if not np.isfinite(tr) or abs(det) < 1e-11: return "Degenerado"
        if det < 0: return "Silla"
        return "Foco/Nodo Estable" if tr < 0 else "Foco/Nodo Inestable"
    def _print_base_ode_equation(self):
        """Imprime la EDO al inicio con los valores fisiológicos y psub/kappa1 simbólicos."""
        phys = self.meta.get("_system_info", {}).get("physical_base_params", {})
        if not phys:
            print("⚠️ No se encontraron parámetros físicos en la metadata para imprimir la EDO.")
            return

        # Extraer parámetros estructurales (ignorando los de control)
        m = phys.get("m")
        g1 = phys.get("gamma1")
        g2 = phys.get("gamma2")
        k2 = phys.get("kappa2")
        c = phys.get("c")
        f0 = phys.get("f0")
        alab = phys.get("alab")
        a01 = phys.get("a01")
        da = phys.get("delta_a")
        tau = phys.get("tau")

        # Precalcular coeficientes fijos
        inv_m = 1.0 / m
        two_tau = 2.0 * tau

        print(f"\n" + "="*80)
        print(f"🔬 ECUACIÓN BASE DEL SISTEMA (Órdenes de Magnitud Fisiológicos)")
        print(f"   Variables de control de la grilla: kappa1 y psub")
        print("="*80)
        print(f" dx/dt = y")
        print(f" dy/dt = {inv_m:.2f} * [")
        print(f"           - (kappa1 + {k2:.1f} x²) x                       (Restitución)")
        print(f"           - ({g1:.4f} + {g2:.1f} y²) y                       (Disipación)")
        print(f"           - {c:.4f} x² y                                 (Acoplamiento)")
        print(f"           + {f0:.4f}                                       (Fuerza base)")
        print(f"           + ({alab:.5f} * psub) * ({da:.4f} + {two_tau:.5f} y) / ({a01:.4f} + x + {tau:.5f} y)")
        print(f"         ]")
        print("="*80 + "\n")

# =============================================================================
# BOOTSTRAP
# =============================================================================

def main():
    service = get_drive_service()
    base_id = get_target_folder_id()
    
    try:
        url = "https://www.googleapis.com/drive/v3/files"
        
        # 1. Buscar carpeta del sistema
        params_sys = {
            "q": f"'{base_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            "fields": "files(id, name, modifiedTime)"
        }
        res = service.get(url, params=params_sys)
        res.raise_for_status()
        folders = res.json().get("files", [])
        
        candidates = sorted([f for f in folders if "syrinx" in f["name"].lower()], 
                            key=lambda x: x["modifiedTime"], reverse=True)
        if not candidates:
            print("❌ No se encontró ninguna carpeta 'syrinx' en Drive."); return
        
        root_id = candidates[0]["id"]
        print(f"Abriendo experimento: {candidates[0]['name']}")

        # 2. Buscar subcarpeta de trayectorias
        params_traj = {"q": f"'{root_id}' in parents and name='trajectories'"}
        res = service.get(url, params=params_traj)
        res.raise_for_status()
        traj_files = res.json().get("files", [])
        
        if not traj_files: 
            print("❌ Falta la carpeta 'trajectories'."); return
            
        traj_id = traj_files[0]["id"]

        # 3. Listar archivos
        idx = get_drive_index_full(service, traj_id)
        
        # 4. Cargar Maestros (In-Memory)
        if "t_eval.npz" not in idx:
            print("❌ Falta 't_eval.npz'. Ejecuta el simulador primero."); return
            
        m_file = [n for n in idx if "metadata" in n][0]
        meta_bytes = download_bytes(service, idx[m_file])
        metadata = json.loads(meta_bytes.decode('utf-8'))
        
        t_eval_bytes = download_bytes(service, idx["t_eval.npz"])
        t_eval = np.load(io.BytesIO(t_eval_bytes))["t_eval"]
        
        # 5. LANZAR VISOR
        global viewer_app
        viewer_app = InteractiveViewer(service, idx, metadata, t_eval, traj_id)
        plt.show()

    except Exception as e:
        print(f"Fallo crítico: {e}")

if __name__ == "__main__":
    main()