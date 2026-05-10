#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manual Homoclinic Classifier
=============================
Extiende la curva homoclina manualmente sobre el diagrama de bifurcaciones.

Flujo de uso
------------
1. El mapa de bifurcaciones se abre con la curva rosa ya detectada.
2. Hacés clic en cualquier punto de la Zona 4 (amarillo).
   - Se carga el retrato de fases (igual que en el InteractiveViewer original).
   - Aparece un panel de clasificación en el borde inferior de la figura.
3. Pulsás "Zona 5 — Ciclo Límite" o "Zona 4 — Escape" según lo que ves.
   - El punto queda marcado en el mapa con (Z5) o (Z4).
   - Si hay al menos un par (Z4, Z5) en el mismo nivel mu2, se calcula el
     punto medio en mu1 y se añade a la extensión de la curva rosa.
4. "Deshacer" elimina la última clasificación.
5. "Guardar Curva" sube la metadata actualizada a Drive con la curva extendida.

Algoritmo de extensión
----------------------
Para cada mu2 con al menos un punto clasificado como Z5 y otro como Z4:
    boundary_mu1 = (max(mu1_Z5_en_ese_mu2) + min(mu1_Z4_en_ese_mu2)) / 2

Esos puntos de frontera se insertan en la curva original y se reordenan por mu2.
"""

import io
import os
import sys
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from matplotlib.widgets import Button

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
TARGET_REGION = "base" # Opciones: "base", "far_z1", "far_z2", "far_z5"
System.set_region(TARGET_REGION)

# =============================================================================
# CONSTANTES
# =============================================================================
SYSTEM_FOLDER_NAME       = System.name.lower().replace("-", "_").replace(" ", "_")
TRAJECTORIES_FOLDER_NAME = "trajectories"
DRIVE_FILES_URL          = "https://www.googleapis.com/drive/v3/files"

TRAJ_STYLE   = {"color": "orange", "alpha": 0.4, "lw": 0.7,  "zorder": 2}
SELECT_STYLE = {"color": "blue",   "alpha": 1.0, "lw": 2.2,  "zorder": 5}

FP_COLORS = {
    "Silla":                "#7B1FA2",
    "Foco/Nodo Estable":    "#2E7D32",
    "Foco/Nodo Inestable":  "#C62828",
    "Degenerado":           "gray",
}

# Límites fijos. Si deseas auto-escalado basado en la región, cámbialo a None.
FIXED_PHASE_LIMITS = System.state_limits

# Colores de los marcadores manuales sobre el mapa
CLR_Z5_MARKER = "#00C853"   # verde brillante → ciclo límite
CLR_Z4_MARKER = "#FF1744"   # rojo  brillante → escape


# =============================================================================
# UTILIDADES DE DRIVE
# =============================================================================

def get_drive_index_full(session, folder_id):
    idx, token = {}, None
    while True:
        params = {
            "q":        f"'{folder_id}' in parents and trashed=false",
            "fields":   "nextPageToken, files(id, name)",
            "pageSize": 1000,
        }
        if token:
            params["pageToken"] = token
        r = session.get(DRIVE_FILES_URL, params=params)
        r.raise_for_status()
        body = r.json()
        for f in body.get("files", []):
            idx[f["name"]] = f["id"]
        token = body.get("nextPageToken")
        if not token:
            break
    return idx


def download_bytes(session, file_id):
    r = session.get(f"{DRIVE_FILES_URL}/{file_id}?alt=media")
    r.raise_for_status()
    return r.content


def find_subfolder(session, parent_id, name, fuzzy=False):
    folders, token = [], None
    while True:
        params = {
            "q": (
                f"'{parent_id}' in parents and "
                "mimeType='application/vnd.google-apps.folder' and trashed=false"
            ),
            "fields":   "nextPageToken, files(id, name, modifiedTime)",
            "pageSize": 1000,
        }
        if token:
            params["pageToken"] = token
        r = session.get(DRIVE_FILES_URL, params=params)
        r.raise_for_status()
        body = r.json()
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
        raise FileNotFoundError(f"No se encontró la subcarpeta '{name}'.")
    return candidates[0]["id"]


def upload_metadata_to_drive(drive_classic, data_dict, file_name, folder_id):
    """Reemplaza el archivo JSON de metadata en Drive."""
    from googleapiclient.http import MediaIoBaseUpload

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return super().default(obj)

    # Borrar versión anterior (cualquier duplicado también)
    q = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
    for item in drive_classic.files().list(q=q, fields="files(id)").execute().get("files", []):
        try:
            drive_classic.files().delete(fileId=item["id"]).execute()
        except Exception as e:
            print(f"  ⚠️  No se pudo borrar versión anterior: {e}")

    raw   = json.dumps(data_dict, indent=2, cls=NumpyEncoder).encode("utf-8")
    fh    = io.BytesIO(raw)
    media = MediaIoBaseUpload(fh, mimetype="application/json", resumable=True)
    drive_classic.files().create(
        media_body=media,
        body={"name": file_name, "parents": [folder_id]},
    ).execute()
    print(f"✅ Metadata guardada en Drive como '{file_name}' ({len(raw)} bytes).")


# =============================================================================
# CLASIFICACIÓN DE EQUILIBRIOS
# =============================================================================

def classify_stability_2d(traza, det):
    if not np.isfinite(traza) or not np.isfinite(det) or abs(det) < 1e-11:
        return "Degenerado"
    if det < 0:
        return "Silla"
    return "Foco/Nodo Estable" if traza < 0 else "Foco/Nodo Inestable"


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class ManualHomoclinicClassifier:
    """Visor interactivo con clasificación manual de puntos en la frontera homoclina."""

    def __init__(self, session, drive_classic, drive_idx, metadata, t_eval, traj_folder_id):
        self.session        = session
        self.drive_classic  = drive_classic   # googleapiclient Resource ya construido
        self.idx            = drive_idx
        self.meta           = metadata
        self.t_eval         = t_eval
        self.traj_folder_id = traj_folder_id

        # Curva homoclina ya calculada (base)
        hc = metadata.get("_detected_homoclinic_curve", [[0.0], [0.0]])
        self.mu1_base = np.array(hc[0], dtype=np.float64)
        self.mu2_base = np.array(hc[1], dtype=np.float64)

        # Clasificaciones manuales: (mu1, mu2) → zone  (4 o 5)
        prev = metadata.get("_manual_classifications", {})
        self.manual: dict[tuple, int] = {}
        for k, v in prev.items():
            try:
                p = parse_param_key(k)
                self.manual[(float(p[0]), float(p[1]))] = int(v)
            except Exception:
                pass

        self.history: list[tuple] = []   # pila para Deshacer
        self.current_point = None        # punto seleccionado actualmente

        # Grilla de parámetros
        self.points_info = self._get_mesh_data()
        if self.points_info is None:
            print("❌ No hay datos en la metadata.")
            self.fig = None
            return

        self._build_figure()
        self._render_map()
        self._init_empty_axes()
        self._update_manual_markers()    # pintar clasificaciones previas
        self._refresh_extended_curve()  # dibujar extensión si ya había datos

        # Eventos
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("pick_event",         self._on_pick_trajectory)

        self._set_classify_buttons_enabled(False)
        print("✅ Clasificador listo.")
        print("   → Clic en zona amarilla para cargar retrato y clasificar.")

    def _get_mesh_data(self):
        points = []
        for k in self.meta:
            if k.startswith("_") or f"{k}.npz" not in self.idx:
                continue
            points.append(parse_param_key(k))
        if not points:
            return None
        pts     = np.array(points)
        p0_axis = np.unique(pts[:, 0])
        p1_axis = np.unique(pts[:, 1])
        Z = np.full((len(p1_axis), len(p0_axis)), np.nan)
        for p in pts:
            i0 = np.where(p0_axis == p[0])[0][0]
            i1 = np.where(p1_axis == p[1])[0][0]
            Z[i1, i0] = self.meta[make_param_key(p)].get("num_fixed_points", 0)
        return {"p0": p0_axis, "p1": p1_axis, "Z": Z, "raw_points": pts}

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(
            f"Clasificador Homoclínico Manual — {System.name} ({TARGET_REGION})",
            fontsize=13, fontweight="bold",
        )

        self.gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[1.2, 0.8],
            hspace=0.38, wspace=0.24,
            left=0.07, right=0.97,
            top=0.93, bottom=0.17,
        )
        self.ax_map    = self.fig.add_subplot(self.gs[0, 0])
        self.ax_phase  = self.fig.add_subplot(self.gs[0, 1])
        self.ax_time   = self.fig.add_subplot(self.gs[1, 0])
        self.ax_single = self.fig.add_subplot(self.gs[1, 1])

        self.selection_marker, = self.ax_map.plot(
            [], [], 'kx', ms=14, mew=2.8, zorder=50,
        )
        self.ext_line, = self.ax_map.plot(
            [], [], color="magenta", ls="-", lw=2.5,
            alpha=0.9, zorder=20, label="Homoclínica (ext. manual)",
        )

        self.scatter_z5 = self.ax_map.scatter(
            [], [], c=CLR_Z5_MARKER, s=70, marker="o",
            edgecolors="white", lw=0.8, zorder=30,
            label="Clasificado Z5 (ciclo)",
        )
        self.scatter_z4 = self.ax_map.scatter(
            [], [], c=CLR_Z4_MARKER, s=70, marker="o",
            edgecolors="white", lw=0.8, zorder=30,
            label="Clasificado Z4 (escape)",
        )

        BTN_Y  = 0.04
        BTN_H  = 0.065
        BTN_W  = 0.13
        gap    = 0.015

        starts = [0.07 + i * (BTN_W + gap) for i in range(5)]

        ax_b5   = self.fig.add_axes([starts[0], BTN_Y, BTN_W, BTN_H])
        ax_b4   = self.fig.add_axes([starts[1], BTN_Y, BTN_W, BTN_H])
        ax_undo = self.fig.add_axes([starts[2], BTN_Y, BTN_W, BTN_H])
        ax_save = self.fig.add_axes([starts[3], BTN_Y, BTN_W, BTN_H])
        ax_quit = self.fig.add_axes([starts[4], BTN_Y, BTN_W, BTN_H])

        # Textos actualizados sin emojis para evitar advertencias de tipografía
        self.btn_z4 = Button(ax_b4, "ZONA 4\n  Ciclo Límite", color="#c8f7c5") 
        self.btn_z5 = Button(ax_b5, "ZONA 5\n  Escape", color="#ffc9c9")      
        self.btn_undo = Button(ax_undo, "<- Deshacer\n  Último",        color="#fff9c4")
        self.btn_save = Button(ax_save, "Guardar\n  Curva",          color="#b3d9ff")
        self.btn_quit = Button(ax_quit, "Cerrar\n  sin Guardar",     color="#e0e0e0")

        for btn, fn in (
            (self.btn_z5,   self._classify_z5),
            (self.btn_z4,   self._classify_z4),
            (self.btn_undo, self._undo),
            (self.btn_save, self._save_and_close),
            (self.btn_quit, lambda _: plt.close(self.fig)),
        ):
            btn.on_clicked(fn)

        self.ax_status = self.fig.add_axes([starts[3] + BTN_W + gap, BTN_Y,
                                            0.97 - (starts[3] + BTN_W + gap), BTN_H * 2])
        self.ax_status.axis("off")
        self.status_text = self.ax_status.text(
            0.0, 0.5, "Sin selección activa.",
            va="center", ha="left", fontsize=8, wrap=True,
            transform=self.ax_status.transAxes,
        )

    def _render_map(self):
        info = self.points_info
        z_max = int(np.nanmax(info["Z"])) if np.any(np.isfinite(info["Z"])) else 1
        cmap  = plt.get_cmap("viridis", max(z_max + 1, 2))
        mesh  = self.ax_map.pcolormesh(
            info["p0"], info["p1"], info["Z"],
            cmap=cmap, shading="nearest", alpha=0.8,
        )
        plt.colorbar(mesh, ax=self.ax_map, label="N° Puntos Fijos",
                     fraction=0.046, pad=0.04)

        try:
            hc_base = (self.mu1_base, self.mu2_base) if len(self.mu1_base) > 1 else None
            curves  = System().get_bifurcation_curves(detected_homoclinic=hc_base)
            for name, (cx, cy, color, style) in curves.items():
                self.ax_map.plot(cx, cy, color=color, ls=style, lw=2, label=name)
        except Exception as e:
            print(f"⚠️  Curvas teóricas: {e}")

        sys_info     = self.meta.get("_system_info", {})
        sweep_ranges = sys_info.get("sweep_ranges")
        if sweep_ranges:
            self.ax_map.set_xlim(sweep_ranges[0])
            self.ax_map.set_ylim(sweep_ranges[1])
        else:
            self.ax_map.set_xlim(System.param_ranges[0])
            self.ax_map.set_ylim(System.param_ranges[1])

        self.ax_map.set_xlabel(rf"${System.param_names[0]}$")
        self.ax_map.set_ylabel(rf"${System.param_names[1]}$")
        self.ax_map.set_title("1. Mapa de Bifurcación (clic para clasificar)")
        self.ax_map.legend(loc="lower right", fontsize="xx-small")

    def _init_empty_axes(self):
        msgs = {
            self.ax_phase:  "Clic en el mapa →",
            self.ax_time:   "Seleccioná una trayectoria",
            self.ax_single: "Detalle de la órbita",
        }
        for ax, msg in msgs.items():
            ax.text(0.5, 0.5, msg, ha="center", va="center", color="gray")
            ax.set_xticks([])
            ax.set_yticks([])

    def _on_click(self, event):
        if event.inaxes != self.ax_map or event.xdata is None:
            return

        pts   = self.points_info["raw_points"]
        rng_x = max(np.ptp(pts[:, 0]), 1e-12)
        rng_y = max(np.ptp(pts[:, 1]), 1e-12)
        dx    = (pts[:, 0] - event.xdata) / rng_x
        dy    = (pts[:, 1] - event.ydata) / rng_y
        best  = pts[int(np.argmin(dx**2 + dy**2))]

        self.current_point = (float(best[0]), float(best[1]))
        self.selection_marker.set_data([best[0]], [best[1]])

        self.ax_phase.clear()
        self.ax_phase.text(
            0.5, 0.5, f"Cargando ({best[0]:.4f}, {best[1]:.4f})…",
            ha="center", va="center", color="steelblue",
        )
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

        self._load_and_render(best)

        zone = System.classify_point(best)
        if zone == 4:
            self._set_classify_buttons_enabled(True)
            self._set_status(
                f"Punto ({best[0]:.4f}, {best[1]:.4f})\n"
                f"Zona 4 detectada. ¿Ciclo o escape?"
            )
        else:
            self._set_classify_buttons_enabled(False)
            self._set_status(
                f"Punto ({best[0]:.4f}, {best[1]:.4f})\n"
                f"Zona {zone} — no clasificable."
            )

        self.fig.canvas.draw_idle()

    def _load_and_render(self, p):
        key       = make_param_key(p)
        file_name = f"{key}.npz"

        if file_name not in self.idx:
            self.ax_phase.clear()
            self.ax_phase.text(0.5, 0.5, f"Falta: {file_name}",
                               ha="center", color="red")
            self._reset_detail_axes("Sin datos")
            return

        try:
            raw = download_bytes(self.session, self.idx[file_name])
            with np.load(io.BytesIO(raw), allow_pickle=True) as d:
                data = {k: d[k] for k in d.files}
        except Exception as e:
            self.ax_phase.clear()
            self.ax_phase.text(0.5, 0.5, f"Error:\n{e}", ha="center", color="red")
            return

        self._render_phase_portrait(data, p)
        self._reset_detail_axes("Clic en una trayectoria →")

    def _render_phase_portrait(self, data, p):
        ax = self.ax_phase
        ax.clear()

        if all(k in data for k in ("U", "V", "x_vals", "y_vals")):
            X, Y = np.meshgrid(data["x_vals"], data["y_vals"])
            ax.streamplot(X, Y, data["U"], data["V"],
                          color=to_rgba("firebrick", 0.55),
                          density=0.7, linewidth=0.5)

        if "trajectories" in data:
            for tr in data["trajectories"]:
                if tr.shape[1] < 2:
                    continue
                line, = ax.plot(tr[0, ::2], tr[1, ::2],
                                picker=True, pickradius=5, **TRAJ_STYLE)
                line.full_data = tr
                line.full_t    = self.t_eval[:tr.shape[1]]

        if "fixed_points" in data:
            fps = data["fixed_points"]
            if fps.ndim == 2 and fps.size > 0:
                for fp in fps:
                    if fps.shape[1] == 4:
                        xc, yc, trv, detv = fp[0], fp[1], fp[2], fp[3]
                    elif fps.shape[1] == 3:
                        xc, yc, trv, detv = fp[0], 0.0, fp[1], fp[2]
                    else:
                        xc, yc, trv, detv = fp[0], 0.0, np.nan, np.nan
                    stab = classify_stability_2d(trv, detv)
                    ax.scatter(xc, yc, c=FP_COLORS.get(stab, "black"),
                               s=100, edgecolors="white", zorder=10, label=stab)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),
                      loc="upper right", fontsize="x-small")

        ax.set_title(
            fr"${System.param_names[0]}={p[0]:.4f}$, "
            fr"${System.param_names[1]}={p[1]:.4f}$",
            fontsize=10,
        )
        ax.set_xlabel(System.state_names[0])
        ax.set_ylabel(System.state_names[1])
        ax.grid(True, alpha=0.2)
        if FIXED_PHASE_LIMITS:
            ax.set_xlim(FIXED_PHASE_LIMITS[0])
            ax.set_ylim(FIXED_PHASE_LIMITS[1])

    def _reset_detail_axes(self, msg):
        for ax in [self.ax_time, self.ax_single]:
            ax.clear()
            ax.text(0.5, 0.5, msg, ha="center", va="center", color="gray")
            ax.set_xticks([])
            ax.set_yticks([])

    def _on_pick_trajectory(self, event):
        if event.mouseevent.inaxes != self.ax_phase:
            return
        line = event.artist
        for l in self.ax_phase.get_lines():
            if hasattr(l, "full_data"):
                l.set(**TRAJ_STYLE)
        line.set(**SELECT_STYLE)
        if not hasattr(line, "full_data"):
            return

        tr    = line.full_data
        t_vec = getattr(line, "full_t", self.t_eval[:tr.shape[1]])

        self.ax_time.clear()
        self.ax_time.plot(t_vec, tr[0, :], label=f"${System.state_names[0]}(t)$",
                          color="tab:blue")
        self.ax_time.plot(t_vec, tr[1, :], label=f"${System.state_names[1]}(t)$",
                          color="tab:orange", alpha=0.7)
        self.ax_time.set_title("Evolución Temporal", fontsize=10)
        self.ax_time.set_xlabel("t")
        self.ax_time.legend(fontsize="x-small")
        self.ax_time.grid(True, alpha=0.3)

        self.ax_single.clear()
        self.ax_single.plot(tr[0, :], tr[1, :], **SELECT_STYLE)
        self.ax_single.scatter(tr[0, 0],  tr[1, 0],  c="green", s=50, zorder=6, label="Inicio")
        self.ax_single.scatter(tr[0, -1], tr[1, -1], marker="x", c="red",
                               s=70, zorder=6, label="Fin")
        self.ax_single.set_title("Órbita Seleccionada", fontsize=10)
        self.ax_single.set_xlabel(System.state_names[0])
        self.ax_single.set_ylabel(System.state_names[1])
        self.ax_single.grid(True, alpha=0.3)
        self.ax_single.legend(fontsize="x-small")

        self.fig.canvas.draw_idle()

    def _classify_z5(self, _event):
        self._record(zone=5)

    def _classify_z4(self, _event):
        self._record(zone=4)

    def _record(self, zone: int):
        if self.current_point is None:
            return
        key = self.current_point
        prev = self.manual.get(key)
        self.manual[key] = zone
        self.history.append((key, prev))   # para Deshacer

        n5 = sum(1 for z in self.manual.values() if z == 5)
        n4 = sum(1 for z in self.manual.values() if z == 4)
        self._set_status(
            f"✔ Clasificado ({key[0]:.4f}, {key[1]:.4f}) → Zona {zone}\n"
            f"Total: {n5} Z5  |  {n4} Z4  |  {len(self.manual)} puntos"
        )

        self._update_manual_markers()
        self._refresh_extended_curve()
        self._set_classify_buttons_enabled(False)
        self.current_point = None
        self.fig.canvas.draw_idle()

    def _undo(self, _event):
        if not self.history:
            self._set_status("Nada que deshacer.")
            return
        key, prev = self.history.pop()
        if prev is None:
            self.manual.pop(key, None)
        else:
            self.manual[key] = prev
        self._set_status(f"↩ Deshecho ({key[0]:.4f}, {key[1]:.4f})")
        self._update_manual_markers()
        self._refresh_extended_curve()
        self.fig.canvas.draw_idle()

    def _update_manual_markers(self):
        pts_z5 = [(mu1, mu2) for (mu1, mu2), z in self.manual.items() if z == 5]
        pts_z4 = [(mu1, mu2) for (mu1, mu2), z in self.manual.items() if z == 4]

        if pts_z5:
            arr = np.array(pts_z5)
            self.scatter_z5.set_offsets(arr)
        else:
            self.scatter_z5.set_offsets(np.empty((0, 2)))

        if pts_z4:
            arr = np.array(pts_z4)
            self.scatter_z4.set_offsets(arr)
        else:
            self.scatter_z4.set_offsets(np.empty((0, 2)))

    def _compute_extended_curve(self):
        by_mu1 = defaultdict(lambda: {"z4": [], "z5": []}) # z4=Ciclo, z5=Escape
        for (m1, m2), zone in self.manual.items():
            key = "z4" if zone == 4 else "z5"
            by_mu1[m1][key].append(m2)

        ext_mu1, ext_mu2 = [], []
        for m1 in sorted(by_mu1.keys()):
            ciclos = by_mu1[m1]["z4"]
            escapes = by_mu1[m1]["z5"]
            if ciclos and escapes:
                vertical_pts = [(m2, 4) for m2 in ciclos] + [(m2, 5) for m2 in escapes]
                vertical_pts.sort()
                for (m2_a, z_a), (m2_b, z_b) in zip(vertical_pts[:-1], vertical_pts[1:]):
                    if z_a != z_b:
                        ext_mu1.append(m1)
                        ext_mu2.append((m2_a + m2_b) / 2.0)
                        break

        all_mu1 = np.concatenate([self.mu1_base, ext_mu1])
        all_mu2 = np.concatenate([self.mu2_base, ext_mu2])
        
        order = np.argsort(all_mu1)
        
        return all_mu1[order], all_mu2[order]

    def _refresh_extended_curve(self):
        mu1_ext, mu2_ext = self._compute_extended_curve()

        print("Extensión actual:", len(mu1_ext), "puntos")  # DEBUG

        self.ext_line.set_data(mu1_ext, mu2_ext)

        self.ax_map.relim()
        self.ax_map.autoscale_view()
        self.ax_map.legend(loc="lower right", fontsize="xx-small")

        self.fig.canvas.draw()

    def _save_and_close(self, _event):
        mu1_ext, mu2_ext = self._compute_extended_curve()

        self.meta["_detected_homoclinic_curve"] = [mu1_ext.tolist(), mu2_ext.tolist()]

        self.meta["_manual_classifications"] = {
            make_param_key(list(k)): int(v)
            for k, v in self.manual.items()
        }

        meta_name = f"grid_metadata_{SYSTEM_FOLDER_NAME}_{TARGET_REGION}.json"
        print(f"Guardando {len(mu1_ext)} puntos en la curva homoclina…")
        try:
            upload_metadata_to_drive(
                self.drive_classic, self.meta, meta_name, self.traj_folder_id
            )
            self._set_status(
                f"✅ Guardado: {len(mu1_ext)} puntos en la curva.\n"
                f"  Base: {len(self.mu1_base)}  +  Manual: {len(mu1_ext) - len(self.mu1_base)}"
            )
            self.fig.canvas.draw_idle()
            print("Podés cerrar la ventana o seguir clasificando.")
        except Exception as e:
            self._set_status(f"❌ Error al guardar:\n{e}")
            self.fig.canvas.draw_idle()

    # --- FUNCIONES RESTAURADAS PARA MANEJO DE UI ---
    def _set_classify_buttons_enabled(self, enabled: bool):
        alpha = 1.0 if enabled else 0.35
        for btn in (self.btn_z5, self.btn_z4):
            btn.ax.set_alpha(alpha)
            btn.label.set_alpha(alpha)

    def _set_status(self, msg: str):
        self.status_text.set_text(msg)


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

        from googleapiclient.discovery import build
        drive_classic = build(
            "drive", "v3",
            credentials=getattr(session, "credentials", session),
        )

        # 1. Carpeta del sistema
        try:
            sys_id = find_subfolder(session, base_id, SYSTEM_FOLDER_NAME)
        except FileNotFoundError:
            sys_id = find_subfolder(session, base_id, SYSTEM_FOLDER_NAME, fuzzy=True)
        
        # 2. Carpeta de la Región
        region_id = find_subfolder(session, sys_id, TARGET_REGION)

        # 3. Carpeta de trayectorias
        traj_id = find_subfolder(session, region_id, TRAJECTORIES_FOLDER_NAME)

        print("Sincronizando índice…")
        idx = get_drive_index_full(session, traj_id)
        print(f"{len(idx)} archivos encontrados.")

        if "t_eval.npz" not in idx:
            print("❌ Falta 't_eval.npz'. Corrí el simulador primero.")
            return

        meta_name = _resolve_metadata_name(idx)
        if meta_name is None:
            print("❌ No se encontró metadata JSON.")
            return

        print(f"Descargando metadata: {meta_name}")
        metadata = json.loads(download_bytes(session, idx[meta_name]).decode("utf-8"))
        t_eval   = np.load(io.BytesIO(download_bytes(session, idx["t_eval.npz"])))["t_eval"]

        global app
        app = ManualHomoclinicClassifier(session, drive_classic, idx, metadata, t_eval, traj_id)
        if app.fig is not None:
            plt.show()

    except Exception as e:
        import traceback
        print(f"❌ Fallo crítico: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()