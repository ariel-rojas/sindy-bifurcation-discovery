#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparador Interactivo Cloud-Native (Ground Truth vs SINDy).
------------------------------------------------------------
Visor analítico dimensional.
- Descarga asíncrona: Trae trayectorias GT y simulaciones SINDy on-demand.
- Mapeo de Datos: Distingue visualmente los datos de Entrenamiento vs Validación.
- UI Robusta: Búsqueda de puntos normalizada y manejo de fallos por datos faltantes.
"""

import io
import os
import sys
import json
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
import joblib
import pysindy as ps
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
# =============================================================================
# RUTAS E IMPORTACIONES LOCALES
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from systems.syrinx import SyrinxModel as System
from core.io import make_param_key, parse_param_key
from core.drive_auth import get_drive_service, get_target_folder_id

# Fix temporal para la retrocompatibilidad de Joblib con PySINDy
if 'pysindy.pysindy' not in sys.modules: sys.modules['pysindy.pysindy'] = ps
if 'pysindy.utils.axes' not in sys.modules: sys.modules['pysindy.utils.axes'] = ps.utils

# =============================================================================
# CONFIGURACIÓN DE EJECUCIÓN
# =============================================================================
VERSION_TO_LOAD = 7     # La versión del modelo a visualizar

SYSTEM_FOLDER_NAME = System.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
TRAJECTORIES_FOLDER = "trajectories"
MODELS_FOLDER = "sindy_models"
SIMS_OUT_FOLDER = "simulations"

CACHE_DIR = os.path.join(tempfile.gettempdir(), f"syrinx_compare_cache_{SYSTEM_FOLDER_NAME}")
os.makedirs(CACHE_DIR, exist_ok=True)

# Estilos visuales
TRAJ_STYLE = {"alpha": 0.5, "lw": 0.8, "zorder": 2}
FP_COLORS = {
    "Silla": "#7B1FA2", 
    "Foco/Nodo Estable": "#2E7D32", 
    "Foco/Nodo Inestable": "#C62828", 
    "Degenerado": "gray"
}

# =============================================================================
# FUNCIONES DE GOOGLE DRIVE (PAGINADAS)
# =============================================================================
def get_drive_index_full(service, folder_id):
    """Obtiene todos los archivos de la carpeta manejando la paginación."""
    idx = {}
    token = None
    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageToken=token, pageSize=1000
        ).execute()
        for f in results.get("files", []):
            idx[f["name"]] = f["id"]
        token = results.get("nextPageToken")
        if not token: break
    return idx

def download_file(service, file_id, name):
    """Descarga un archivo al caché local de forma segura."""
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path): return path
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: _, done = downloader.next_chunk()
    return path

# =============================================================================
# CLASE PRINCIPAL DEL COMPARADOR
# =============================================================================
class InteractiveComparator:
    def __init__(self, service, indices, metadata, train_keys, model):
        self.service = service
        self.idx_traj = indices["traj"]
        self.idx_sims = indices["sims"]
        self.meta = metadata
        self.train_keys = set(train_keys)
        
        # Extraer puntos validos (existen en simulación o trayectoria)
        self.points_info = self._get_mesh_data()
        
        # Configuración de la Figura (1 Fila, 3 Columnas)
        self.fig = plt.figure(figsize=(18, 6))
        self.gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.25)
        
        self.ax_map = self.fig.add_subplot(self.gs[0])
        self.ax_gt = self.fig.add_subplot(self.gs[1])
        # Compartir ejes para que los retratos de fase estén sincronizados al hacer zoom
        self.ax_sindy = self.fig.add_subplot(self.gs[2], sharex=self.ax_gt, sharey=self.ax_gt)
        
        self.selection_marker, = self.ax_map.plot([], [], 'kx', ms=12, mew=2.5, zorder=50)
        
        # Renderizado
        self._print_model(model)
        self._render_map()
        self._init_empty_axes()
        
        # Eventos
        self.fig.canvas.mpl_connect("button_press_event", self.on_click_map)
        
        plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.1)
        print("\n✅ Visor Comparativo Listo. Haz clic en el mapa para cargar simulaciones.")

    def _print_model(self, model):
        print("\n" + "="*60)
        print(f"--- Modelo SINDy Cargado (Batch rs_batch | v{VERSION_TO_LOAD}) ---")
        try:
            model.print(precision=4)
        except Exception as e:
            print(f"Ecuación no disponible: {e}")
        print("="*60)

    def _get_mesh_data(self):
        pts_train, pts_valid = [], []
        
        for k in self.meta.keys():
            if k.startswith("_"): continue
            
            # Formato de archivo
            file_name = f"{k}.npz"
            
            # Determinar si este punto se simuló en GT o en SINDy
            has_gt = file_name in self.idx_traj
            has_sindy = file_name in self.idx_sims
            
            if has_gt or has_sindy:
                p = parse_param_key(k)
                if file_name in self.train_keys:
                    pts_train.append(p)
                elif has_sindy:
                    # Solo pintamos de azul los que efectivamente se validaron
                    pts_valid.append(p)
                
        pts_train = np.array(pts_train) if pts_train else np.empty((0, 2))
        pts_valid = np.array(pts_valid) if pts_valid else np.empty((0, 2))
        
        all_pts = np.vstack((pts_train, pts_valid)) if pts_valid.size > 0 else pts_train
        return {"train": pts_train, "valid": pts_valid, "all": all_pts}

    def _render_map(self):
        info = self.points_info
        
        # 1. Curvas Teóricas Físicas
        sys_info = self.meta.get("_system_info", {})
        phys_params = sys_info.get("physical_base_params", {})
        if phys_params:
            curves = System.get_physical_bifurcation_curves(phys_params)
            for name, d in curves.items():
                self.ax_map.plot(d["psub"], d["kappa1"], color=d["color"], 
                                 ls=d["linestyle"], lw=2, label=name, zorder=5)

        # 2. Scatter de Puntos (Entrenamiento vs Validación)
        if info["train"].size > 0:
            self.ax_map.scatter(info["train"][:, 1], info["train"][:, 0], 
                                color="red", s=30, label=f"Train ({len(info['train'])})", zorder=10)
        if info["valid"].size > 0:
            self.ax_map.scatter(info["valid"][:, 1], info["valid"][:, 0], 
                                color="royalblue", marker="s", s=20, label=f"Validation ({len(info['valid'])})", zorder=9)

        # 3. Formato del mapa
        if info["all"].size > 0:
            self.ax_map.set_xlim(np.min(info["all"][:, 1]), np.max(info["all"][:, 1]) )
            self.ax_map.set_ylim(np.min(info["all"][:, 0]), np.max(info["all"][:, 0]) )
            
        self.ax_map.set_title("Espacio de Parámetros (Haz clic para explorar)", fontsize=12)
        self.ax_map.set_xlabel(r"Presión Subglótica $p_{sub}$ (dyn/cm$^2$)")
        self.ax_map.set_ylabel(r"Rigidez $\kappa_1$ (dyn/cm)")
        self.ax_map.legend(loc="best", fontsize="small")
        self.ax_map.grid(True, alpha=0.3)

    def _init_empty_axes(self):
        for ax, title in [(self.ax_gt, "Ground Truth"), (self.ax_sindy, "SINDy Prediction")]:
            ax.text(0.5, 0.5, "Esperando selección...", ha='center', va='center', color='gray')
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
            

    def on_click_map(self, event):
        if event.inaxes != self.ax_map or event.xdata is None: return
        
        pts = self.points_info["all"]
        if pts.size == 0: return
        
        # pts[:, 0] = kappa1 (Y), pts[:, 1] = psub (X)
        norm_x = (pts[:, 1] - event.xdata) / (np.max(pts[:, 1]) - np.min(pts[:, 1]) + 1e-9)
        norm_y = (pts[:, 0] - event.ydata) / (np.max(pts[:, 0]) - np.min(pts[:, 0]) + 1e-9)
        
        idx_closest = np.argmin(norm_x**2 + norm_y**2)
        best_p = pts[idx_closest]
        
        # Mover la cruz negra
        self.selection_marker.set_data([best_p[1]], [best_p[0]])
        self.fig.canvas.draw_idle()
        
        self._load_and_compare(best_p)

    def _load_and_compare(self, p):
        key = make_param_key(p)
        file_name = f"{key}.npz"
        
        self.ax_gt.clear(); self.ax_sindy.clear()
        self.ax_gt.set_title(fr"GT: $\kappa_1$={p[0]:.2f}, $p_{{sub}}$={p[1]:.0f}")
        self.ax_sindy.set_title(fr"SINDy: $\kappa_1$={p[0]:.2f}, $p_{{sub}}$={p[1]:.0f}")
        
        # 1. Cargar Ground Truth
        if file_name in self.idx_traj:
            path_gt = download_file(self.service, self.idx_traj[file_name], f"gt_{file_name}")
            with np.load(path_gt, allow_pickle=True) as d:
                data_gt = {k: d[k] for k in d.files}
            self._render_phase_portrait(self.ax_gt, data_gt, "tomato")
        else:
            self.ax_gt.text(0.5, 0.5, "Archivo GT no encontrado", ha='center')

        # 2. Cargar SINDy
        if file_name in self.idx_sims:
            path_sindy = download_file(self.service, self.idx_sims[file_name], f"sindy_{file_name}")
            with np.load(path_sindy, allow_pickle=True) as d:
                data_sindy = {k: d[k] for k in d.files}
            self._render_phase_portrait(self.ax_sindy, data_sindy, "royalblue")
        else:
            self.ax_sindy.text(0.5, 0.5, "No simulado por SINDy", ha='center', color='red')

        self.fig.canvas.draw_idle()

    def _render_phase_portrait(self, ax, data, traj_color):
        # 1. Campo Vectorial
        if "U" in data and "V" in data:
            X, Y = np.meshgrid(data["x_vals"], data["y_vals"])
            ax.streamplot(X, Y, data["U"], data["V"], 
                          color=to_rgba("gray", 0.5), density=0.8, linewidth=0.6)

        # 2. Trayectorias
        if "trajectories" in data:
            trajs = data["trajectories"]
            if isinstance(trajs, np.ndarray) and trajs.ndim == 0:
                # Fallback por si las trayectorias se guardaron como objeto anidado
                trajs = trajs.item().get("all_trajectories", [])
                
            for tr in trajs:
                if len(tr) > 0 and tr.shape[1] > 2:
                    # Saltamos puntos (::2) para acelerar el renderizado
                    ax.plot(tr[0, ::2], tr[1, ::2], color=traj_color, **TRAJ_STYLE)

        # 3. Puntos Fijos (Solo suele estar en GT)
        if "fixed_points" in data:
            for fp in data["fixed_points"]:
                stab = self._classify_stability(fp[2], fp[3])
                ax.scatter(fp[0], fp[1], c=FP_COLORS.get(stab, "black"), 
                           s=80, edgecolors="white", zorder=10, label=stab)
                
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='x-small')

        # 4. Singularidad Fisiológica
        phys = self.meta["_system_info"]["physical_base_params"]
        xl_plot = np.array([np.min(data["x_vals"]), np.max(data["x_vals"])])
        yl_sing = (-phys["a01"] - xl_plot) / phys["tau"]
        ax.plot(xl_plot, yl_sing, 'k:', lw=1.5, alpha=0.6, label="Singularidad")

        # 5. Fijar límites según la grilla de condiciones iniciales
        if "x_vals" in data and "y_vals" in data:
            ax.set_xlim(np.min(data["x_vals"]), np.max(data["x_vals"]))
            ax.set_ylim(np.min(data["y_vals"]), np.max(data["y_vals"]))

        ax.set_xlabel("Desplazamiento x (cm)")
        ax.set_ylabel("Velocidad v (cm/s)")
        ax.grid(True, alpha=0.2)

    def _classify_stability(self, tr, det):
        if not np.isfinite(tr) or abs(det) < 1e-11: return "Degenerado"
        if det < 0: return "Silla"
        return "Foco/Nodo Estable" if tr < 0 else "Foco/Nodo Inestable"

# =============================================================================
# BOOTSTRAP Y CONEXIÓN
# =============================================================================
def main():
    print("Iniciando conexión a Google Drive...")
    
    try:
        raw_service = get_drive_service()
        # FIX: Si get_drive_service() retorna un AuthorizedSession, construimos el cliente API v3
        if not hasattr(raw_service, 'files'):
            creds = getattr(raw_service, 'credentials', raw_service)
            service = build('drive', 'v3', credentials=creds)
        else:
            service = raw_service

        base_id = get_target_folder_id()
    
        # 1. Navegación jerárquica
        sys_query = service.files().list(q=f"name='{SYSTEM_FOLDER_NAME}' and '{base_id}' in parents and trashed=false", fields='files(id)').execute()
        if not sys_query.get('files'): raise FileNotFoundError("Carpeta del sistema no encontrada.")
        sys_id = sys_query.get('files')[0]['id']
        
        traj_id = service.files().list(q=f"name='{TRAJECTORIES_FOLDER}' and '{sys_id}' in parents and trashed=false", fields='files(id)').execute().get('files')[0]['id']
        models_id = service.files().list(q=f"name='{MODELS_FOLDER}' and '{sys_id}' in parents and trashed=false", fields='files(id)').execute().get('files')[0]['id']
        
        
        batch_query = service.files().list(q=f"name='rs_batch' and '{models_id}' in parents and trashed=false", fields='files(id)').execute()
        if not batch_query.get('files'): raise FileNotFoundError("Carpeta 'rs_batch' no encontrada en Drive.")
        batch_id = batch_query.get('files')[0]['id']
        
        ver_query = service.files().list(q=f"name='v{VERSION_TO_LOAD}' and '{batch_id}' in parents and trashed=false", fields='files(id)').execute()
        if not ver_query.get('files'): raise FileNotFoundError(f"Carpeta 'v{VERSION_TO_LOAD}' no encontrada.")
        ver_id = ver_query.get('files')[0]['id']
        
        sims_query = service.files().list(q=f"name='{SIMS_OUT_FOLDER}' and '{ver_id}' in parents and trashed=false", fields='files(id)').execute()
        sims_id = sims_query.get('files')[0]['id'] if sims_query.get('files') else None
        
        if not sims_id: raise FileNotFoundError("La carpeta de simulaciones de validación no existe.")

        # 2. Sincronización de índices paginados
        print("Sincronizando índices de archivos (esto puede tomar unos segundos)...")
        idx_traj = get_drive_index_full(service, traj_id)
        idx_ver = get_drive_index_full(service, ver_id)
        idx_sims = get_drive_index_full(service, sims_id)
        
        indices = {"traj": idx_traj, "sims": idx_sims}

        # 3. Descarga de Metadatos y Modelo
        meta_file = [n for n in idx_traj if "metadata" in n][0]
        metadata = json.load(open(download_file(service, idx_traj[meta_file], meta_file)))
        
        params_file = "sindy_training_params.json"
        params_path = download_file(service, idx_ver[params_file], f"v{VERSION_TO_LOAD}_{params_file}")
        with open(params_path, "r") as f:
            train_keys = json.load(f).get("keys_used", [])
            # Asegurar la extensión para hacer match con el índice
            train_keys = [k if k.endswith('.npz') else f"{k}.npz" for k in train_keys]
            
        model_file = "sindy_model.joblib"
        model_path = download_file(service, idx_ver[model_file], f"v{VERSION_TO_LOAD}_{model_file}")
        model = joblib.load(model_path)

        # 4. Lanzar Visor
        global app
        app = InteractiveComparator(service, indices, metadata, train_keys, model)
        plt.show()

    except Exception as e:
        print(f"❌ Fallo crítico en la inicialización: {e}")

if __name__ == "__main__":
    main()