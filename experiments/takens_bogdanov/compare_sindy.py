#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparador CLOUD-NATIVE (Ground Truth vs SINDy Ensamble).

FUNCIONALIDAD MÁXIMA (Validación Cruzada):
- Define TRAIN_REGION (dónde se entrenó) y TEST_REGION (dónde se simula).
- Selecciona una de las 11 configuraciones de muestreo.
- Conecta a Drive y localiza el último 'run' de esa configuración.
- Descarga la metadata (JSON) del ensamble y muestra las ecuaciones descubiertas.
- Mapea el espacio de parámetros coloreando los puntos de Entrenamiento vs Validación.
- Al hacer clic en un punto, descarga instantáneamente el .npz original y el 
  predicho por SINDy desde la nube para compararlos lado a lado.
"""

import os
import sys
import json
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import io
from enum import Enum
from googleapiclient.http import MediaIoBaseDownload

# --- FIX DE IMPORTACIONES Y RUTAS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System
from core.io import parse_param_key, make_param_key
from core.drive_auth import get_drive_service, get_target_folder_id

# Importamos el parche de compatibilidad para Drive
from experiments.takens_bogdanov.data_zone_manager import _build_drive_service

# =============================================================================
# ENUMS Y CONFIGURACIÓN DE BÚSQUEDA CRUZADA
# =============================================================================
class SamplingMode(Enum):
    ALL_ZONES = "all_zones"
    ONE_ZONE_VAR_PARAMS = "one_zone_var_params"
    ONE_ZONE_FIXED_PARAMS = "one_zone_fixed_params"

# --- CONFIGURACIÓN DE VALIDACIÓN CRUZADA ---
TRAIN_REGION = "base"  # De dónde salió el modelo entrenado (ej. "base", "far_z5")
TEST_REGION  = "far_z5"    # Dónde lo vamos a evaluar y visualizar visualmente

# Forzamos la clase a adaptar los límites a la zona de Test
System.set_region(TEST_REGION)

# --- ELIGE AQUÍ QUÉ CONFIGURACIÓN (LOTE) QUIERES VISUALIZAR ---
TARGET_MODE = SamplingMode.ALL_ZONES
TARGET_ZONE = "Todas" # 1, 2, 3, 4, 5 (o "Todas" si el modo es ALL_ZONES)

# Nombres de Carpetas
SYSTEM_FOLDER_NAME = System.name.lower().replace("-", "_").replace(" ", "_")
TRAJECTORIES_FOLDER_NAME = "trajectories"
MODELS_FOLDER_NAME = "sindy_models"
SIMS_OUT_FOLDER_NAME = f"simulations_{TEST_REGION}"  # Carpeta dinámica de la Fase 5

CACHE_DIR = os.path.join(tempfile.gettempdir(), f"sindy_compare_cache_{SYSTEM_FOLDER_NAME}_{TEST_REGION}")
os.makedirs(CACHE_DIR, exist_ok=True)

FP_COLORS = {
    "Silla": "#7B1FA2",             
    "Foco/Nodo Estable": "#2E7D32", 
    "Foco/Nodo Inestable": "#C62828", 
    "Desconocido": "gray"
}

# =============================================================================
# LÓGICA DE GOOGLE DRIVE (CLOUD-NATIVE)
# =============================================================================
def download_json_in_memory(service, file_id):
    """Descarga y parsea un JSON directamente en memoria."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done: _, done = downloader.next_chunk()
    fh.seek(0)
    return json.load(fh)

def _get_folder_id(service, parent_id, name):
    query = f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    res = service.files().list(q=query, fields='files(id)').execute().get('files', [])
    if not res: return None
    return res[0]['id']

def get_latest_run_paths(service, models_root_id):
    """Navega la jerarquía de modelos y encuentra la ejecución más reciente."""
    batch_name = f"batch_{TARGET_MODE.value}" if TARGET_MODE == SamplingMode.ALL_ZONES else f"batch_{TARGET_MODE.value}_zone_{TARGET_ZONE}"
    batch_id = _get_folder_id(service, models_root_id, batch_name)
    
    if not batch_id: raise FileNotFoundError(f"No se encontró la carpeta {batch_name} en el modelo entrenado.")
    
    # Listar todos los run_ y ordenarlos por nombre (timestamp)
    q_runs = f"'{batch_id}' in parents and name contains 'run_' and trashed=false"
    runs = service.files().list(q=q_runs, fields='files(id, name)').execute().get('files', [])
    if not runs: raise FileNotFoundError(f"No se encontraron ejecuciones en {batch_name}")
    
    latest_run = sorted(runs, key=lambda x: x['name'], reverse=True)[0]
    run_id = latest_run['id']
    run_name = latest_run['name']
    
    sims_id = _get_folder_id(service, run_id, SIMS_OUT_FOLDER_NAME)
    
    return run_id, sims_id, run_name

def build_drive_index(service, folder_id):
    if not folder_id: return {}
    index = {}
    page_token = None
    while True:
        results = service.files().list(q=f"'{folder_id}' in parents and trashed=false", spaces='drive', fields='nextPageToken, files(id, name)', pageToken=page_token, pageSize=1000).execute()
        for item in results.get('files', []): index[item['name']] = item['id']
        page_token = results.get('nextPageToken', None)
        if not page_token: break
    return index

def download_file_to_cache(service, file_id, local_file_name):
    local_path = os.path.join(CACHE_DIR, local_file_name)
    if os.path.exists(local_path): return local_path 
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done: _, done = downloader.next_chunk()
    return local_path

# =============================================================================
# UTILIDADES VISUALES Y MATEMÁTICAS
# =============================================================================
def print_ensemble_equations(metadata):
    """Reconstruye e imprime las ecuaciones a partir del JSON del ensamble."""
    print("\n" + "="*70)
    print(f"--- Modelo SINDy Descubierto (Media del Ensamble) ---")
    print(f"    Entrenado en Región: '{TRAIN_REGION}'")
    
    disc_coefs = metadata.get("discovered_coefficients", {})
    
    for eq_name in ["x'", "y'"]:
        if eq_name not in disc_coefs: continue
        
        rhs_terms = []
        for term_name, stats in disc_coefs[eq_name].items():
            mean_val = stats["predictions_stats"]["mean"]
            if abs(mean_val) > 1e-4:
                rhs_terms.append(f"{mean_val:+.4f} {term_name}")
                
        equation_str = "0.0" if not rhs_terms else " ".join(rhs_terms)
        if equation_str.strip().startswith("+"): equation_str = equation_str.strip()[1:].strip()
        print(f"{eq_name} = {equation_str}")
    print("="*70 + "\n")

def get_points_from_keys(keys):
    valid_points = []
    for k in keys:
        try: valid_points.append(parse_param_key(k))
        except Exception: continue
    return np.array(valid_points) if valid_points else np.empty((0, 2))

def get_train_keys(metadata):
    """Extrae las llaves usadas durante el entrenamiento desde los logs del ensamble."""
    train_keys = set()
    for log in metadata.get("iteration_logs", []):
        for key in log.get("sampling_record", {}).keys():
            train_keys.add(key)
    return list(train_keys)

def classify_stability_2d(jacobian_elems):
    traza, det = jacobian_elems
    if det < 0: return "Silla"
    elif traza < 0: return "Foco/Nodo Estable"
    else: return "Foco/Nodo Inestable"

def render_panel(ax, data_dict, key, title, traj_color):
    """Dibuja usando diccionarios numpy."""
    ax.clear()
    
    x_lims, y_lims = System.state_limits[0], System.state_limits[1]
    
    if data_dict is None:
        ax.text(0.5, 0.5, "Sin datos en Drive", ha='center')
        ax.set_title(title)
        ax.set_xlim(x_lims); ax.set_ylim(y_lims)
        return

    # Campo Vectorial
    if "U" in data_dict and "V" in data_dict:
        u, v = data_dict["U"], data_dict["V"]
        x_src, y_src = data_dict["x_vals"], data_dict["y_vals"]
        nx, ny = u.shape[1], u.shape[0]
        x_grid = np.linspace(x_src.min(), x_src.max(), nx)
        y_grid = np.linspace(y_src.min(), y_src.max(), ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        ax.streamplot(X, Y, u, v, color='k', density=0.8, linewidth=0.4, arrowsize=0.6)

    # Trayectorias
    if "trajectories" in data_dict:
        trajs = data_dict["trajectories"]
        if isinstance(trajs, np.ndarray):
            time_step = 2 
            ax.set_autoscale_on(False) 
            for i in range(trajs.shape[0]):
                xy = trajs[i]
                ax.plot(xy[0, ::time_step], xy[1, ::time_step], color=traj_color, alpha=0.3, linewidth=0.4)

    # Puntos Fijos
    if "fixed_points" in data_dict:
        fps = data_dict["fixed_points"]
        if fps.ndim == 2 and fps.size > 0:
            for i in range(fps.shape[0]):
                fp = fps[i]
                x_coord, y_coord = fp[0], 0.0 
                label_txt = f"FP x={x_coord:.2f}"
                color = "black"
                if fps.shape[1] >= 3: 
                    stab_type = classify_stability_2d([fp[1], fp[2]])
                    color = FP_COLORS.get(stab_type, "black")
                    label_txt = f"{stab_type}"
                ax.scatter(x_coord, y_coord, c=color, s=60, zorder=20, edgecolors='white', linewidth=1.5, label=label_txt)

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='x-small', framealpha=0.9)

    try:
        params = parse_param_key(key)
        ax.set_title(fr"{title}\n$\mu_1$={params[0]:.4f}, $\mu_2$={params[1]:.4f}", fontsize=11)
    except:
        ax.set_title(title, fontsize=11)

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def main():
    print(f"--- Comparador Interactivo Cloud-Native ---")
    print(f"📥 Entrenado en: '{TRAIN_REGION}' | 📈 Evaluado en: '{TEST_REGION}'")
    print(f"Modo: {TARGET_MODE.value} | Zona: {TARGET_ZONE}")
    
    try:
        raw_service = get_drive_service()
        service = _build_drive_service(raw_service)
        base_id = get_target_folder_id()
        
        # Bifurcación de jerarquía Drive
        sys_id = _get_folder_id(service, base_id, SYSTEM_FOLDER_NAME)
        train_region_id = _get_folder_id(service, sys_id, TRAIN_REGION)
        test_region_id = _get_folder_id(service, sys_id, TEST_REGION)
        
        if not train_region_id or not test_region_id:
            print("Error: No se encontró alguna de las regiones especificadas en Drive.")
            return

        # GT sale del Test Region, Modelos salen del Train Region
        traj_test_id = _get_folder_id(service, test_region_id, TRAJECTORIES_FOLDER_NAME)
        models_root_id = _get_folder_id(service, train_region_id, MODELS_FOLDER_NAME)
        
        # 1. Obtener IDs y construir índices de Drive
        run_id, sims_id, run_name = get_latest_run_paths(service, models_root_id)
        print(f"Modelo encontrado: {run_name}")
        
        print("Sincronizando índices...")
        traj_index = build_drive_index(service, traj_test_id)
        run_index = build_drive_index(service, run_id)
        sims_index = build_drive_index(service, sims_id)
        
        # 2. Descargar Metadata JSON
        if "ensemble_metadata.json" not in run_index:
            print("Error: No se encontró ensemble_metadata.json en Drive.")
            return
            
        metadata = download_json_in_memory(service, run_index["ensemble_metadata.json"])
        
    except Exception as e:
        print(f"Error fatal conectando a Drive: {e}")
        return

    # Imprimir Ecuaciones del Ensamble
    print_ensemble_equations(metadata)
    
    # --- Extraer Claves de Entrenamiento vs Validación ---
    train_keys = get_train_keys(metadata)
    
    # IMPORTANTE: Si evaluamos el modelo en otra región (far_z5 vs base),
    # los puntos de entrenamiento NO van a aparecer en este mapa porque
    # pertenecen espacialmente a otra zona. Solo mostramos los rojos si coinciden.
    
    valid_keys = [k.replace('.npz', '') for k in sims_index.keys() if k != "t_eval.npz"]
    
    pts_train = get_points_from_keys(train_keys)
    pts_valid = get_points_from_keys(valid_keys)
    
    # --- Configuración Gráfica ---
    fig = plt.figure(figsize=(19, 7))
    ax_map = plt.subplot(1, 3, 1)
    ax_gt = plt.subplot(1, 3, 2)
    ax_sindy = plt.subplot(1, 3, 3, sharex=ax_gt, sharey=ax_gt)
    
    # Mapa
    curves = System().get_bifurcation_curves()
    for name, (cx, cy, c, s) in curves.items():
        ax_map.plot(cx, cy, color=c, linestyle=s, label=name, lw=2, alpha=0.8)
        
    if pts_train.size > 0 and TRAIN_REGION == TEST_REGION:
        ax_map.scatter(pts_train[:, 0], pts_train[:, 1], c='red', label='Entrenamiento', s=60, zorder=10, picker=True, edgecolors='k')
    if pts_valid.size > 0:
        ax_map.scatter(pts_valid[:, 0], pts_valid[:, 1], c='blue', marker='s', label='Validación', s=30, alpha=0.6, zorder=9, picker=True)
    
    ax_map.legend(loc='lower left', fontsize='small')
    ax_map.set_title(f"Train: {TRAIN_REGION} | Test: {TEST_REGION}\n{run_name}", fontsize=11)
    ax_map.set_xlabel(System.param_names[0])
    ax_map.set_ylabel(System.param_names[1])
    ax_map.set_xlim(System.param_ranges[0])
    ax_map.set_ylim(System.param_ranges[1])
    ax_map.grid(True, linestyle=':', alpha=0.5)
    
    # --- Interacción On-Demand ---
    def on_pick(event):
        if event.artist not in ax_map.collections: return
        
        ind = event.ind[0]
        data = event.artist.get_offsets()
        p0, p1 = data[ind]
        
        key_str = make_param_key([p0, p1])
        file_name = f"{key_str}.npz"
        print(f"Seleccionado: {key_str}")
        
        ax_gt.clear(); ax_sindy.clear()
        ax_gt.text(0.5, 0.5, "Descargando GT...", ha='center')
        ax_sindy.text(0.5, 0.5, f"Descargando SINDy...", ha='center')
        fig.canvas.draw_idle()
        plt.pause(0.01)
        
        # 1. Cargar Ground Truth (De la región TEST)
        if file_name in traj_index:
            try:
                local_gt_name = f"gt_{key_str}.npz"
                gt_path = download_file_to_cache(service, traj_index[file_name], local_gt_name)
                with np.load(gt_path, allow_pickle=True) as data:
                    gt_dict = {k: data[k] for k in data.files}
                    if "trajectories" in gt_dict and isinstance(gt_dict["trajectories"], np.ndarray) and gt_dict["trajectories"].ndim == 0:
                        gt_dict = gt_dict["trajectories"].item() 
                        gt_dict["trajectories"] = gt_dict["trajectories"]["all_trajectories"]
                render_panel(ax_gt, gt_dict, key_str, "Ground Truth", "tomato")
            except Exception as e:
                render_panel(ax_gt, None, key_str, f"Error GT: {e}", "tomato")
        else:
            render_panel(ax_gt, None, key_str, "Ground Truth", "tomato")

        # 2. Cargar SINDy Simulation (De la carpeta "simulations_TEST" dentro del run)
        if sims_index and file_name in sims_index:
            try:
                local_sindy_name = f"sindy_{run_name}_{key_str}.npz"
                sindy_path = download_file_to_cache(service, sims_index[file_name], local_sindy_name)
                with np.load(sindy_path, allow_pickle=True) as data:
                    sindy_dict = {k: data[k] for k in data.files}
                render_panel(ax_sindy, sindy_dict, key_str, "SINDy Model", "royalblue")
            except Exception as e:
                render_panel(ax_sindy, None, key_str, f"Error SINDy: {e}", "royalblue")
        else:
            render_panel(ax_sindy, None, key_str, "SINDy Model", "royalblue")

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    
    ax_gt.text(0.5, 0.5, "Haz clic en un punto\ndel mapa", ha='center', va='center', color='gray')
    ax_sindy.text(0.5, 0.5, "Haz clic en un punto\ndel mapa", ha='center', va='center', color='gray')
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    ax_sindy.set_xticks([]); ax_sindy.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()