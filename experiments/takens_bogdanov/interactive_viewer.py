#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Takens-Bogdanov Interactive Viewer (Cloud-Native, In-Memory)
============================================================

Visor interactivo del retrato de fases y mapa de bifurcación para el sistema
de Takens-Bogdanov. 

Paneles
-------
1. Mapa de bifurcación: heatmap de ``num_fixed_points`` + curvas teóricas.
2. Retrato de fases: streamplot del campo + trayectorias clicables + puntos fijos.
3. Series temporales x(t), y(t) de la trayectoria seleccionada.
4. Órbita individual de la trayectoria seleccionada.
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
# RUTAS E IMPORTACIONES DEL PROYECTO
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System

from core.io import make_param_key, parse_param_key
from core.drive_auth import get_drive_service, get_target_folder_id

# --- SELECCIÓN DE REGIÓN DE EXPLORACIÓN ---
TARGET_REGION = "far_z5" # Opciones: "base", "far_z1", "far_z2", "far_z5"
System.set_region(TARGET_REGION)

# =============================================================================
# CONSTANTES DE NEGOCIO Y ESTÉTICA
# =============================================================================
SYSTEM_FOLDER_NAME = System.name.lower().replace("-", "_").replace(" ", "_")
TRAJECTORIES_FOLDER_NAME = "trajectories"
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"

# Estilos de trazado
TRAJ_STYLE = {"color": "orange", "alpha": 0.4, "lw": 0.7, "zorder": 2}
SELECT_STYLE = {"color": "blue", "alpha": 1.0, "lw": 2.2, "zorder": 5}

# Color por tipo de equilibrio
FP_COLORS = {
    "Silla": "#7B1FA2",              # Violeta
    "Foco/Nodo Estable": "#2E7D32",  # Verde
    "Foco/Nodo Inestable": "#C62828",  # Rojo
    "Degenerado": "gray",
}

# Límites del retrato de fase, adaptados dinámicamente según la región si se usa System.state_limits
FIXED_PHASE_LIMITS = System.state_limits


# =============================================================================
# UTILIDADES DE GOOGLE DRIVE (REST PURO, SIN CACHÉ EN DISCO)
# =============================================================================
def get_drive_index_full(session, folder_id):
    idx = {}
    token = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken, files(id, name)",
            "pageSize": 1000,
        }
        if token:
            params["pageToken"] = token

        response = session.get(DRIVE_FILES_URL, params=params)
        response.raise_for_status()
        body = response.json()
        for f in body.get("files", []):
            idx[f["name"]] = f["id"]
        token = body.get("nextPageToken")
        if not token:
            break
    return idx


def download_bytes(session, file_id):
    url = f"{DRIVE_FILES_URL}/{file_id}?alt=media"
    response = session.get(url)
    response.raise_for_status()
    return response.content


def find_subfolder(session, parent_id, name, fuzzy=False):
    folders = []
    token = None
    while True:
        params = {
            "q": (f"'{parent_id}' in parents and "
                  f"mimeType='application/vnd.google-apps.folder' and trashed=false"),
            "fields": "nextPageToken, files(id, name, modifiedTime)",
            "pageSize": 1000,
        }
        if token:
            params["pageToken"] = token
        response = session.get(DRIVE_FILES_URL, params=params)
        response.raise_for_status()
        body = response.json()
        folders.extend(body.get("files", []))
        token = body.get("nextPageToken")
        if not token:
            break

    if fuzzy:
        candidates = sorted(
            [f for f in folders if name.lower() in f["name"].lower()],
            key=lambda x: x.get("modifiedTime", ""),
            reverse=True,
        )
    else:
        candidates = [f for f in folders if f["name"] == name]

    if not candidates:
        raise FileNotFoundError(
            f"No se encontró la subcarpeta '{name}' bajo el padre '{parent_id}'."
        )
    return candidates[0]["id"]


# =============================================================================
# CLASIFICACIÓN DE ESTABILIDAD LINEAL 2D
# =============================================================================
def classify_stability_2d(traza, det):
    if not np.isfinite(traza) or not np.isfinite(det) or abs(det) < 1e-11:
        return "Degenerado"
    if det < 0:
        return "Silla"
    return "Foco/Nodo Estable" if traza < 0 else "Foco/Nodo Inestable"


# =============================================================================
# CLASE PRINCIPAL DEL VISOR
# =============================================================================
class InteractiveViewer:
    """Visor interactivo de 4 paneles basado en eventos de matplotlib."""

    def __init__(self, session, drive_idx, metadata, t_eval):
        self.session = session
        self.idx = drive_idx
        self.meta = metadata
        self.t_eval = t_eval

        self.points_info = self._get_mesh_data()
        if self.points_info is None:
            print("❌ No se encontraron datos válidos en la metadata.")
            self.fig = None
            return

        self.fig = plt.figure(figsize=(15, 9))
        self.fig.suptitle(f"Visor Interactivo — {System.name} ({TARGET_REGION})", fontsize=13, fontweight="bold")
        self.gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[1.2, 0.8],
            hspace=0.35,
            wspace=0.22,
        )
        self.ax_map = self.fig.add_subplot(self.gs[0, 0])
        self.ax_phase = self.fig.add_subplot(self.gs[0, 1])
        self.ax_time = self.fig.add_subplot(self.gs[1, 0])
        self.ax_single = self.fig.add_subplot(self.gs[1, 1])

        self.selection_marker, = self.ax_map.plot(
            [], [], 'kx', ms=12, mew=2.5, zorder=50
        )

        self._render_map()
        self._init_empty_axes()

        self.fig.canvas.mpl_connect("button_press_event", self.on_click_map)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick_trajectory)

        plt.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.07)
        print("✅ Visor listo. Hacé clic en el mapa de parámetros (panel 1).")

    def _get_mesh_data(self):
        points = []
        for k in self.meta.keys():
            if k.startswith("_") or f"{k}.npz" not in self.idx:
                continue
            points.append(parse_param_key(k))
        if not points:
            return None

        pts = np.array(points)
        p0_axis = np.unique(pts[:, 0])  # mu_1 (eje X del mapa)
        p1_axis = np.unique(pts[:, 1])  # mu_2 (eje Y del mapa)

        z = np.full((len(p1_axis), len(p0_axis)), np.nan)
        for p in pts:
            i0 = np.where(p0_axis == p[0])[0][0]
            i1 = np.where(p1_axis == p[1])[0][0]
            z[i1, i0] = self.meta[make_param_key(p)].get("num_fixed_points", 0)

        return {"p0": p0_axis, "p1": p1_axis, "Z": z, "raw_points": pts}

    def _render_map(self):
        info = self.points_info

        z_max = int(np.nanmax(info["Z"])) if np.any(np.isfinite(info["Z"])) else 1
        cmap = plt.get_cmap("viridis", max(z_max + 1, 2))
        mesh = self.ax_map.pcolormesh(
            info["p0"], info["p1"], info["Z"],
            cmap=cmap, shading="nearest", alpha=0.8,
        )
        plt.colorbar(mesh, ax=self.ax_map, label="N° Puntos Fijos",
                     fraction=0.046, pad=0.04)

        try:
            detected_hc = self.meta.get("_detected_homoclinic_curve", None)
            curves = System().get_bifurcation_curves(detected_homoclinic=detected_hc)
            for name, (cx, cy, color, style) in curves.items():
                self.ax_map.plot(cx, cy, color=color, linestyle=style,
                                 label=name, lw=2)
            self.ax_map.legend(loc="lower right", fontsize="x-small")
        except Exception as e:
            print(f"⚠️  Omitiendo curvas teóricas: {e}")

        sys_info = self.meta.get("_system_info", {})
        sweep_ranges = sys_info.get("sweep_ranges")
        if sweep_ranges:
            self.ax_map.set_xlim(sweep_ranges[0][0], sweep_ranges[0][1])
            self.ax_map.set_ylim(sweep_ranges[1][0], sweep_ranges[1][1])
        else:
            self.ax_map.set_xlim(System.param_ranges[0])
            self.ax_map.set_ylim(System.param_ranges[1])

        self.ax_map.set_xlabel(rf"${System.param_names[0]}$")
        self.ax_map.set_ylabel(rf"${System.param_names[1]}$")
        self.ax_map.set_title(f"1. Mapa de Bifurcación: {System.name}")

    def _init_empty_axes(self):
        msgs = {
            self.ax_phase: "Hacé clic en el mapa (1)",
            self.ax_time: "Seleccioná una trayectoria (2)",
            self.ax_single: "Detalle de la órbita seleccionada",
        }
        for ax, msg in msgs.items():
            ax.text(0.5, 0.5, msg, ha='center', va='center', color='gray')
            ax.set_xticks([])
            ax.set_yticks([])

    def on_click_map(self, event):
        if event.inaxes != self.ax_map or event.xdata is None:
            return

        pts = self.points_info["raw_points"]
        rng_x = max(np.ptp(pts[:, 0]), 1e-12)
        rng_y = max(np.ptp(pts[:, 1]), 1e-12)
        dx = (pts[:, 0] - event.xdata) / rng_x
        dy = (pts[:, 1] - event.ydata) / rng_y
        idx_closest = int(np.argmin(dx ** 2 + dy ** 2))
        best_p = pts[idx_closest]

        self.selection_marker.set_data([best_p[0]], [best_p[1]])

        self.ax_phase.clear()
        key = make_param_key(best_p)
        self.ax_phase.text(0.5, 0.5, f"Cargando {key}...",
                           ha='center', va='center', color='blue')
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

        self._load_data_point(best_p)
        self.fig.canvas.draw_idle()

    def _load_data_point(self, p):
        key = make_param_key(p)
        file_name = f"{key}.npz"
        print(f"Cargando desde la nube: {file_name}")

        if file_name not in self.idx:
            self.ax_phase.clear()
            self.ax_phase.text(0.5, 0.5, f"Falta archivo:\n{file_name}",
                               ha='center', color='red')
            self._reset_detail_axes("Selección sin datos")
            return

        try:
            raw_bytes = download_bytes(self.session, self.idx[file_name])
            with np.load(io.BytesIO(raw_bytes), allow_pickle=True) as d:
                data = {k: d[k] for k in d.files}
        except Exception as e:
            self.ax_phase.clear()
            self.ax_phase.text(0.5, 0.5, f"Error descargando:\n{e}",
                               ha='center', color='red')
            return

        self._render_phase_portrait(data, p)
        self._reset_detail_axes("Hacé clic en una trayectoria")

    def _render_phase_portrait(self, data, p):
        ax = self.ax_phase
        ax.clear()

        if all(k in data for k in ("U", "V", "x_vals", "y_vals")):
            X, Y = np.meshgrid(data["x_vals"], data["y_vals"])
            ax.streamplot(
                X, Y, data["U"], data["V"],
                color=to_rgba("firebrick", 0.6),
                density=0.7, linewidth=0.5,
            )

        if "trajectories" in data:
            trajs = data["trajectories"]
            step = 2 
            for tr in trajs:
                if tr.shape[1] < 2:
                    continue
                line, = ax.plot(
                    tr[0, ::step], tr[1, ::step],
                    picker=True, pickradius=5, **TRAJ_STYLE,
                )
                line.full_data = tr
                line.full_t = self.t_eval[:tr.shape[1]]

        if "fixed_points" in data:
            fps = data["fixed_points"]
            if fps.ndim == 2 and fps.size > 0:
                for fp in fps:
                    if fps.shape[1] == 4:
                        x_c, y_c, tr_v, det_v = fp[0], fp[1], fp[2], fp[3]
                    elif fps.shape[1] == 3:
                        x_c, y_c, tr_v, det_v = fp[0], 0.0, fp[1], fp[2]
                    else:
                        x_c, y_c, tr_v, det_v = fp[0], 0.0, np.nan, np.nan

                    stab = classify_stability_2d(tr_v, det_v)
                    ax.scatter(
                        x_c, y_c,
                        c=FP_COLORS.get(stab, "black"),
                        s=100, edgecolors="white", zorder=10, label=stab,
                    )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),
                      loc='upper right', fontsize='x-small')

        ax.set_title(
            fr"Retrato: ${System.param_names[0]}={p[0]:.4f}$, "
            fr"${System.param_names[1]}={p[1]:.4f}$",
            fontsize=11,
        )
        ax.set_xlabel(System.state_names[0])
        ax.set_ylabel(System.state_names[1])
        ax.grid(True, alpha=0.2)

        if FIXED_PHASE_LIMITS is not None:
            ax.set_xlim(FIXED_PHASE_LIMITS[0])
            ax.set_ylim(FIXED_PHASE_LIMITS[1])

    def _reset_detail_axes(self, msg):
        for ax in [self.ax_time, self.ax_single]:
            ax.clear()
            ax.text(0.5, 0.5, msg, ha='center', va='center', color='gray')
            ax.set_xticks([])
            ax.set_yticks([])

    def on_pick_trajectory(self, event):
        if event.mouseevent.inaxes != self.ax_phase:
            return

        line = event.artist

        for l in self.ax_phase.get_lines():
            if hasattr(l, "full_data"):
                l.set(**TRAJ_STYLE)

        line.set(**SELECT_STYLE)

        if not hasattr(line, "full_data"):
            return

        tr = line.full_data
        t_vec = getattr(line, "full_t", self.t_eval[:tr.shape[1]])

        self.ax_time.clear()
        self.ax_time.plot(t_vec, tr[0, :],
                          label=f"${System.state_names[0]}(t)$",
                          color="tab:blue")
        self.ax_time.plot(t_vec, tr[1, :],
                          label=f"${System.state_names[1]}(t)$",
                          color="tab:orange", alpha=0.7)
        self.ax_time.set_title("Evolución Temporal", fontsize=10)
        self.ax_time.set_xlabel("t")
        self.ax_time.legend(loc="upper right", fontsize="x-small")
        self.ax_time.grid(True, alpha=0.3)

        self.ax_single.clear()
        self.ax_single.plot(tr[0, :], tr[1, :], **SELECT_STYLE)
        self.ax_single.scatter(tr[0, 0], tr[1, 0], c='green', s=50,
                               zorder=6, label='Inicio')
        self.ax_single.scatter(tr[0, -1], tr[1, -1], marker='x', c='red',
                               s=70, zorder=6, label='Fin')
        self.ax_single.set_title("Órbita Seleccionada", fontsize=10)
        self.ax_single.set_xlabel(System.state_names[0])
        self.ax_single.set_ylabel(System.state_names[1])
        self.ax_single.grid(True, alpha=0.3)
        self.ax_single.legend(fontsize="x-small")

        self.fig.canvas.draw_idle()


# =============================================================================
# BOOTSTRAP
# =============================================================================
def _resolve_metadata_name(idx):
    expected = f"grid_metadata_{SYSTEM_FOLDER_NAME}_{TARGET_REGION}.json"
    if expected in idx:
        return expected
    candidates = [n for n in idx
                  if n.startswith("grid_metadata") and n.endswith(".json")]
    return candidates[0] if candidates else None


def main():
    try:
        session = get_drive_service()
        base_id = get_target_folder_id()

        try:
            sys_id = find_subfolder(session, base_id, SYSTEM_FOLDER_NAME)
        except FileNotFoundError:
            print(f"⚠️  No hay carpeta '{SYSTEM_FOLDER_NAME}' exacta. Buscando por substring...")
            sys_id = find_subfolder(session, base_id, SYSTEM_FOLDER_NAME, fuzzy=True)
        
        print(f"✅ Carpeta del sistema localizada.")

        # Buscar subcarpeta de la región
        region_id = find_subfolder(session, sys_id, TARGET_REGION)

        # Buscar subcarpeta de trayectorias
        traj_id = find_subfolder(session, region_id, TRAJECTORIES_FOLDER_NAME)

        print("Sincronizando índice de archivos con Google Drive...")
        idx = get_drive_index_full(session, traj_id)
        print(f"Índice listo: {len(idx)} archivos en '{TRAJECTORIES_FOLDER_NAME}'.")

        if "t_eval.npz" not in idx:
            print("❌ Falta 't_eval.npz' en Drive. ¿Corriste el simulador?")
            return

        meta_name = _resolve_metadata_name(idx)
        if meta_name is None:
            print("❌ No se encontró ningún archivo de metadata (*.json).")
            return
        print(f"Descargando metadata: {meta_name}")

        meta_bytes = download_bytes(session, idx[meta_name])
        metadata = json.loads(meta_bytes.decode("utf-8"))

        t_eval_bytes = download_bytes(session, idx["t_eval.npz"])
        t_eval = np.load(io.BytesIO(t_eval_bytes))["t_eval"]

        global viewer_app
        viewer_app = InteractiveViewer(session, idx, metadata, t_eval)
        if viewer_app.fig is not None:
            plt.show()

    except Exception as e:
        print(f"❌ Fallo crítico: {e}")


if __name__ == "__main__":
    main()