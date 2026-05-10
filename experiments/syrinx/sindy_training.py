#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orquestador CLOUD-NATIVE de Descubrimiento de Ecuaciones (SINDy) para SyrinxModel.

FUNCIONALIDAD:
- Búsqueda Aleatoria (Random Search): Explora hiperparámetros (Zona, Tuplas, Umbrales, Alpha, T_Max).
- Filtrado Físico Dimensional: Extrae y recorta trayectorias directamente de Google Drive.
- Meta-Programación Numba: Testea dinámicamente la estabilidad de las ecuaciones inferidas.
- Tolerancia a Fallos: Descarta automáticamente los modelos divergentes (Blow-ups).
- Trazabilidad: Loguea métricas, hiperparámetros, gráficos y modelos (.joblib) en la nube.
"""

import os
import sys
import time
import json
import random
import tempfile
import io
import shutil
from datetime import datetime

import numpy as np
import pysindy as ps
import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from numba import jit

from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.discovery import build

# =============================================================================
# CONFIGURACIÓN DE RUTAS LOCALES E IMPORTACIONES DEL PROYECTO
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from systems.syrinx import SyrinxModel as System
from core.io import parse_param_key
from core.drive_auth import get_drive_service, get_target_folder_id
from core.integrators import rk4_general

# Fix de retrocompatibilidad de PySINDy con Joblib
if 'pysindy.pysindy' not in sys.modules: sys.modules['pysindy.pysindy'] = ps
if 'pysindy.utils.axes' not in sys.modules: sys.modules['pysindy.utils.axes'] = ps.utils

# =============================================================================
# HIPERPARÁMETROS DEL RANDOM SEARCH Y CONFIGURACIÓN GLOBAL
# =============================================================================
N_RANDOM_SEARCH_ITERS = 10  # Número total de iteraciones/modelos a probar

PARAM_GRID = {
    "SELECTION_MODE": ["ZONE"],
    "TARGET_ZONE": ["Zona 2"],
    "N_TUPLES": [5, 10, 15, 20, 30],
    "N_TRAJS_PER_TUPLE": [1, 2, 3],
    "THRESHOLD": [0.01, 0.05, 0.1, 0.2, 0.4],
    "RIDGE_ALPHA": [0.01, 0.05, 0.1, 0.5],       # Regularización L2
    "T_MAX_TRAIN_FRAC": [0.25, 0.5, 0.75, 1.0]   # Porcentaje del T_Max original a usar
}

DEBUG_PLOT_ZONE = False
MANUAL_RANGE_FILTER = False
MANUAL_K_RANGE = (100.0, 1000.0)
MANUAL_P_RANGE = (5000.0, 15000.0)

POLY_DEGREE = 3
UNBIAS = True

# Nomenclatura de Nube y Caché
SYSTEM_FOLDER_NAME = System.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
TRAJECTORIES_FOLDER_NAME = "trajectories"
MODELS_FOLDER_NAME = "sindy_models"
METADATA_FILE_NAME = f"grid_metadata_{SYSTEM_FOLDER_NAME}.json"

TMP_DIR = os.path.join(tempfile.gettempdir(), f"sindy_train_cache_{SYSTEM_FOLDER_NAME}")
os.makedirs(TMP_DIR, exist_ok=True)


# =============================================================================
# MÓDULO 1: UTILIDADES DE NUBE Y ARCHIVOS (GOOGLE DRIVE)
# =============================================================================
class NpEncoder(json.JSONEncoder):
    """Permite serializar tipos de NumPy a JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def create_or_get_subfolder(service, parent_id, folder_name):
    """Busca una carpeta en Drive por nombre; si no existe, la crea."""
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if not items:
        metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_id]}
        return service.files().create(body=metadata, fields='id').execute().get('id')
    return items[0]['id']

def download_file_to_cache(service, file_id, file_name):
    """Descarga un archivo desde Drive al caché temporal local."""
    local_path = os.path.join(TMP_DIR, file_name)
    if os.path.exists(local_path): return local_path 
    for attempt in range(4): 
        try:
            request = service.files().get_media(fileId=file_id)
            fh = io.FileIO(local_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done: _, done = downloader.next_chunk()
            return local_path
        except Exception as e:
            try: fh.close()
            except: pass
            if attempt < 3: time.sleep(2 ** attempt)
            else: raise ConnectionError(f"Fallo crítico al descargar {file_name}: {e}")

def upload_file_to_drive(service, local_path, file_name, folder_id, existing_file_id=None):
    """Sube o actualiza un archivo local a una carpeta en Drive."""
    for attempt in range(4):
        try:
            with open(local_path, "rb") as fd:
                media = MediaIoBaseUpload(fd, mimetype='application/octet-stream', resumable=True)
                if existing_file_id:
                    service.files().update(fileId=existing_file_id, media_body=media).execute()
                else:
                    metadata = {'name': file_name, 'parents': [folder_id]}
                    service.files().create(media_body=media, body=metadata).execute()
            if os.path.exists(local_path):
                os.remove(local_path)
            return True
        except Exception as e:
            if attempt < 3: time.sleep(2 ** attempt)
            else: 
                print(f"\n[Fallo Red] No se pudo subir {file_name}: {e}")
                return False


# =============================================================================
# MÓDULO 2: VISUALIZACIÓN DE TRAYECTORIAS E HISTORIAL DE STLSQ
# =============================================================================
def plot_optimizer_history(model, feature_names, save_path):
    """Genera un gráfico mostrando la evolución de los coeficientes iteración a iteración."""
    history = getattr(model.optimizer, "history_", None)
    if not history: return

    if len(feature_names) != history[0].shape[1]:
        feature_names = model.get_feature_names()

    n_iters, n_targets, n_features = len(history), history[0].shape[0], history[0].shape[1]
    fig, axes = plt.subplots(n_targets, 1, figsize=(10, 5 * n_targets))
    if n_targets == 1: axes = [axes]
    
    target_names = [r"$\dot{x}$", r"$\dot{y}$"]

    for i in range(n_targets):
        ax = axes[i]
        for j in range(n_features):
            coef_vals = [history[k][i, j] for k in range(n_iters)]
            if np.any(np.abs(coef_vals) > 1e-10):
                ax.plot(range(n_iters), coef_vals, marker='o', label=feature_names[j])
        
        ax.set_title(f"Evolución de Coeficientes para {target_names[i]}")
        ax.set_xlabel("Iteración de STLSQ")
        ax.set_ylabel("Valor del Coeficiente")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_sample_trajectories(X_list, save_path):
    """Grafica el espacio de fase dimensional nativo de hasta 10 trayectorias aleatorias."""
    fig, ax = plt.subplots(figsize=(8, 6))
    n_plot = min(10, len(X_list))
    sample_indices = random.sample(range(len(X_list)), n_plot)
    colors = plt.cm.tab10.colors
    
    for color_idx, i in enumerate(sample_indices):
        traj = X_list[i]
        x_dim, v_dim = traj[:, 0], traj[:, 1]
        c = colors[color_idx % len(colors)]
        ax.plot(x_dim, v_dim, color=c, lw=1.5, alpha=0.8, label=f"Trayectoria {color_idx+1}")
        ax.scatter(x_dim[0], v_dim[0], color=c, marker='o', s=40, zorder=5) 
        
    ax.set_xlabel(r"Desplazamiento $x$ [cm]", fontsize=12)
    ax.set_ylabel(r"Velocidad $v$ [cm/s]", fontsize=12)
    ax.set_title("Espacio de Fase Dimensional (Muestra de Entrenamiento)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
def plot_validation_trajectories(model, X_list_full, U_list, t_list_full, save_path):
    """Compara visualmente el Ground Truth (línea continua) vs Predicción SINDy (línea punteada)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    feature_names = model.get_feature_names()
    ode_func = generate_local_sindy_ode_jit(model.optimizer.coef_, feature_names)
    
    n_plot = min(5, len(X_list_full))
    sample_indices = random.sample(range(len(X_list_full)), n_plot)
    colors = plt.cm.tab10.colors
    
    for color_idx, i in enumerate(sample_indices):
        traj_gt = X_list_full[i]
        t_array = t_list_full[i]
        
        x0 = traj_gt[0].astype(np.float32)
        dt = np.float32(t_array[1] - t_array[0])
        t_start, t_end = np.float32(t_array[0]), np.float32(t_array[-1])
        param_arr = U_list[i][0].astype(np.float32)
        
        # Simulamos usando la ecuación descubierta por SINDy
        traj_sindy = rk4_general(ode_func, x0, t_start, t_end, dt, param_arr)
        
        c = colors[color_idx % len(colors)]
        
        # Plot Ground Truth (Línea continua gruesa y semitransparente)
        ax.plot(traj_gt[:, 0], traj_gt[:, 1], color=c, lw=1.2, ls="-")
        
        # Plot SINDy (Línea punteada, fina y nítida)
        ax.plot(traj_sindy[0, :], traj_sindy[1, :], color=c, lw=1.2, ls="--", alpha=0.8)
        
        # Marcador de condición inicial
        ax.scatter(x0[0], x0[1], color=c, marker='o', s=40, zorder=5) 
        
    ax.set_xlabel(r"Desplazamiento $x$ [cm]", fontsize=12)
    ax.set_ylabel(r"Velocidad $v$ [cm/s]", fontsize=12)
    ax.set_title("Validación Extrapolatoria: Ground Truth vs SINDy", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Leyenda personalizada para no saturar con colores
    custom_lines = [
        Line2D([0], [0], color='k', lw=1, ls = "-"),
        Line2D([0], [0], color='k', lw=1, ls="--")
    ]
    ax.legend(custom_lines, ['Ground Truth (Original)', 'Predicción SINDy'], loc='best', fontsize='small')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# =============================================================================
# MÓDULO 3: GEOMETRÍA DEL ESPACIO DE PARÁMETROS Y FILTRADO
# =============================================================================
def build_boundary_interpolator(curve_data, x_key, y_key):
    """Crea un interpolador matemático 1D a partir de una curva discreta teórica."""
    x_vals, y_vals = curve_data[x_key], curve_data[y_key]
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_v, y_v = x_vals[valid], y_vals[valid]
    if len(y_v) == 0: return lambda y: np.nan
    
    sort_idx = np.argsort(y_v)
    y_sorted, x_sorted = y_v[sort_idx], x_v[sort_idx]
    y_uniq, unq_idx = np.unique(y_sorted, return_index=True)
    x_uniq = x_sorted[unq_idx]
    return lambda y_input: float(np.interp(y_input, y_uniq, x_uniq, left=np.nan, right=np.nan))

def filter_keys(all_keys, sys_info, selection_mode, target_zone):
    """Clasifica y filtra las claves disponibles según rango dimensional y zona dinámica."""
    phys_base = sys_info.get("physical_base_params", System.default_params)

    p0_search_min, p0_search_max = -np.inf, np.inf 
    p1_search_min, p1_search_max = -np.inf, np.inf 

    if MANUAL_RANGE_FILTER:
        p0_search_min, p0_search_max = MANUAL_K_RANGE
        p1_search_min, p1_search_max = MANUAL_P_RANGE

    if selection_mode == "ZONE" or DEBUG_PLOT_ZONE:
        curves = System.get_physical_bifurcation_curves(phys_base, x_min=0.001, x_max=0.5, n_points=50000)
        hopf_func = build_boundary_interpolator(curves.get("Hopf", {}), "psub", "kappa1")
        sn_func = build_boundary_interpolator(curves.get("Saddle-Node", {}), "psub", "kappa1")

    valid_keys = []
    for key in all_keys:
        vals = parse_param_key(key.replace(".npz", ""))
        if len(vals) < 2: continue
        
        p0_val, p1_val = float(vals[0]), float(vals[1])
        if not (p0_search_min <= p0_val <= p0_search_max and p1_search_min <= p1_val <= p1_search_max):
            continue
        
        if selection_mode == "ZONE":
            p_hopf = hopf_func(p0_val)
            p_sn = sn_func(p0_val)
            
            if np.isnan(p_hopf) and np.isnan(p_sn): zona = "Zona 0"
            elif np.isnan(p_hopf): zona = "Zona 1" if p1_val <= p_sn else "Zona 3"
            elif np.isnan(p_sn): zona = "Zona 1" if p1_val <= p_hopf else "Zona 3"
            else:
                left_b, right_b = min(p_hopf, p_sn), max(p_hopf, p_sn)
                if p1_val <= left_b: zona = "Zona 1"
                elif p1_val <= right_b: zona = "Zona 2"
                else: zona = "Zona 3"
                
            if zona != target_zone:
                continue 

        valid_keys.append(key)
            
    return valid_keys


# =============================================================================
# MÓDULO 4: EXTRACCIÓN DE DATOS Y TEST DE ESTABILIDAD SINDy
# =============================================================================
def fetch_cloud_data(service, drive_index, t_full, t_max_train, sys_info, selection_mode, target_zone, n_tuples, n_trajs_per_tuple):
    """Muestrea aleatoriamente trayectorias desde Drive y devuelve series truncadas y completas."""
    X_list, X_list_full, U_list, t_list_train, t_list_full = [], [], [], [], []
    sys_state_dim = len(System.state_names)
    
    all_keys = [k for k in drive_index.keys() if k.endswith('.npz') and k not in ["t_eval.npz", METADATA_FILE_NAME]]
    if not all_keys: raise ValueError("No se encontraron trayectorias en Drive.")
    
    valid_keys = filter_keys(all_keys, sys_info, selection_mode, target_zone)
    if not valid_keys: raise ValueError("Ningún archivo cumple con los filtros geométricos.")
        
    param_keys = random.sample(valid_keys, min(n_tuples, len(valid_keys)))
    sampling_record = {}
    
    valid_t_idx = t_full <= t_max_train
    n_valid_train = np.sum(valid_t_idx)

    for key in param_keys:
        try:
            local_path = download_file_to_cache(service, drive_index[key], key)
            with np.load(local_path, allow_pickle=True) as data:
                trajs, swept_vals = data["trajectories"], data["swept_vals"]
                
            n_sims = len(trajs)
            if n_sims == 0: continue
                
            actual_trajs = min(n_trajs_per_tuple, n_sims)
            selected_idx = random.sample(range(n_sims), actual_trajs)
            sampling_record[key] = selected_idx
            
            for i in selected_idx:
                traj = trajs[i]
                if traj.shape[0] != sys_state_dim: continue
                
                actual_steps_orig = traj.shape[1]
                actual_steps_train = min(actual_steps_orig, n_valid_train)
                
                if actual_steps_train < 3: 
                    continue
                
                traj_T = traj.T[:actual_steps_train, :] 
                traj_T_full = traj.T[:actual_steps_orig, :] # <-- EXTRAEMOS LA COMPLETA
                param_block = np.tile(swept_vals, (actual_steps_train, 1))
                
                X_list.append(traj_T)
                X_list_full.append(traj_T_full)             # <-- GUARDAMOS LA COMPLETA
                U_list.append(param_block) 
                t_list_train.append(t_full[:actual_steps_train])
                t_list_full.append(t_full[:actual_steps_orig]) 
                
            if os.path.exists(local_path): os.remove(local_path)
        except Exception:
            pass
            
    return X_list, X_list_full, U_list, t_list_train, t_list_full, param_keys, sampling_record

def generate_local_sindy_ode_jit(coeffs, feature_names):
    """Genera código JIT dinámico para evaluar la ODE inferida por SINDy."""
    state_dim = len(System.state_names)
    coeffs_states = coeffs[:state_dim, :].astype(np.float32)
    coeffs_T = np.ascontiguousarray(coeffs_states.T)

    lines = [
        "@jit(nopython=True)",
        "def sindy_ode_wrapper(t, state_arr, param_arr):",
        "    x = state_arr[0]",
        "    y = state_arr[1]",
        "    kappa1 = param_arr[0]",
        "    psub = param_arr[1]",
        f"    theta = np.empty({len(feature_names)}, dtype=np.float32)"
    ]
    for i, feat in enumerate(feature_names):
        val = "1.0" if feat == "1" else feat.replace(" ", " * ").replace("^", "**")
        lines.append(f"    theta[{i}] = {val}")
        
    lines.append("    d_state = np.dot(theta, coeffs_T)")
    lines.append("    return d_state.astype(np.float32)")

    code = "\n".join(lines)
    namespace = {"np": np, "jit": jit, "coeffs_T": coeffs_T}
    exec(code, namespace)
    return namespace["sindy_ode_wrapper"]

def test_sindy_stability(model, X_list, U_list, t_list_full):
    """Simula una muestra de condiciones iniciales usando el tiempo ORIGINAL completo para descartar divergencias."""
    feature_names = model.get_feature_names()
    ode_func = generate_local_sindy_ode_jit(model.optimizer.coef_, feature_names)

    n_tests = min(3, len(X_list))
    test_indices = random.sample(range(len(X_list)), n_tests)
    max_safe_val = 20 

    for idx in test_indices:
        x0 = X_list[idx][0].astype(np.float32)
        
        # USAMOS EL VECTOR DE TIEMPO ORIGINAL, NO EL TRUNCADO
        t_array = t_list_full[idx] 
        dt = np.float32(t_array[1] - t_array[0])
        t_start, t_end = np.float32(t_array[0]), np.float32(t_array[-1])

        param_arr = U_list[idx][0].astype(np.float32)
        sol = rk4_general(ode_func, x0, t_start, t_end, dt, param_arr)

        expected_steps = len(t_array)
        
        # Si la simulación explota ANTES del tiempo que alcanzó la original, es inestable
        if sol.shape[1] < expected_steps - 1 or np.any(np.abs(sol) > max_safe_val) or np.any(np.isnan(sol)):
            return False 

    return True


# =============================================================================
# MÓDULO 5: ORQUESTADOR PRINCIPAL (RANDOM SEARCH PIPELINE)
# =============================================================================
def main():
    print(f"--- Iniciando SINDy Discovery Cloud-Native ({System.name}) ---")
    if os.path.exists(TMP_DIR): shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)
    
    sys_info = {}
    
    # 1. Autenticación y Construcción de Índices de Nube
    try:
        raw_service = get_drive_service()
        if not hasattr(raw_service, 'files'):
            creds = getattr(raw_service, 'credentials', raw_service)
            service = build('drive', 'v3', credentials=creds)
        else:
            service = raw_service

        base_id = get_target_folder_id()
        sys_id = create_or_get_subfolder(service, base_id, SYSTEM_FOLDER_NAME)
        traj_id = create_or_get_subfolder(service, sys_id, TRAJECTORIES_FOLDER_NAME)
        
        print("Sincronizando índice de archivos de Drive...")
        drive_index = {}
        page_token = None
        while True:
            res = service.files().list(q=f"'{traj_id}' in parents and trashed=false", spaces='drive', fields='nextPageToken, files(id, name)', pageToken=page_token).execute()
            for i in res.get('files', []): drive_index[i['name']] = i['id']
            page_token = res.get('nextPageToken', None)
            if not page_token: break
            
        teval_path = download_file_to_cache(service, drive_index["t_eval.npz"], "t_eval.npz")
        with np.load(teval_path) as data: t_full = data["t_eval"]

        if METADATA_FILE_NAME in drive_index:
            print(f"Descargando metadatos estructurales...")
            meta_path = download_file_to_cache(service, drive_index[METADATA_FILE_NAME], METADATA_FILE_NAME)
            with open(meta_path, "r", encoding="utf-8") as f:
                sys_info = json.load(f).get("_system_info", {})
        else:
            print("⚠️ ADVERTENCIA: JSON de metadatos no encontrado. Fallando a defaults locales.")
            
    except Exception as e:
        print(f"Error fatal conectando a Google Drive: {e}")
        return

    # 2. Bucle Principal de Búsqueda Aleatoria
    t_max_original = t_full[-1]
    
    for iter_idx in range(N_RANDOM_SEARCH_ITERS):
        sel_mode = random.choice(PARAM_GRID["SELECTION_MODE"])
        tgt_zone = random.choice(PARAM_GRID["TARGET_ZONE"])
        n_tup = random.choice(PARAM_GRID["N_TUPLES"])
        n_traj = random.choice(PARAM_GRID["N_TRAJS_PER_TUPLE"])
        thresh = random.choice(PARAM_GRID["THRESHOLD"])
        alpha = random.choice(PARAM_GRID["RIDGE_ALPHA"])
        t_frac = random.choice(PARAM_GRID["T_MAX_TRAIN_FRAC"])
        
        t_max_train = t_max_original * t_frac

        print("\n" + "="*75)
        print(f"🔍 [Iteración {iter_idx+1}/{N_RANDOM_SEARCH_ITERS}] Evaluando hiperparámetros:")
        print(f"   ► Muestreo : {sel_mode} | Zona: {tgt_zone.split('(')[0].strip()}")
        print(f"   ► Datos    : Tuplas={n_tup} | Traj/Tupla={n_traj} | T_Max_Train={t_max_train:.2f}s")
        print(f"   ► STLSQ    : Threshold={thresh} | Alpha={alpha}")
        print("="*75)

        try:
            # Ahora desempaquetamos X_list_full
            X_list, X_list_full, U_list, t_list_train, t_list_full, used_keys, sampling_record = fetch_cloud_data(
                service, drive_index, t_full, t_max_train, sys_info, sel_mode, tgt_zone, n_tup, n_traj
            )
        except ValueError as e:
            print(f"Saltando (Datos insuficientes): {e}")
            continue
            
        if not X_list: continue

        # SINDy se entrena usando los datos TRUNCADOS
        X_dot_list = [np.gradient(X, t, axis=0) for X, t in zip(X_list, t_list_train)]

        # 3. Entrenamiento SINDy
        state_names = System.state_names
        control_names = ["kappa1", "psub"]
        all_feature_names = state_names + control_names
        
        optimizer = ps.STLSQ(threshold=thresh, alpha=alpha, max_iter=100, unbias=UNBIAS, normalize_columns=True)
        feature_library = ps.PolynomialLibrary(degree=POLY_DEGREE)

        model = ps.SINDy(optimizer=optimizer, feature_library=feature_library)
        model.fit(x=X_list, t=t_list_train, x_dot=X_dot_list, u=U_list, feature_names=all_feature_names)

        # 4. Impresión de Ecuaciones
        model.print(precision=4)

        DISPLAY_THRESHOLD = 0.1 
        print(f"\n--- Ecuación Dominante (Filtro Visual Abs > {DISPLAY_THRESHOLD}) ---")
        coefs = model.coefficients()
        features = model.get_feature_names()
        
        for i, target in enumerate(state_names):
            terms = []
            for j, feature in enumerate(features):
                coef = coefs[i, j]
                if abs(coef) >= DISPLAY_THRESHOLD:
                    terms.append(f"{coef:.4f} {feature}")
            eq_str = " + ".join(terms).replace("+ -", "- ") if terms else "0"
            print(f"({target})' = {eq_str}")

        # 5. Validación de Estabilidad Física (Anti-Blowups extrapolatorios)
        print("\nVerificando estabilidad dinámica del modelo inferido...")
        # La validación se hace usando el tiempo COMPLETO ORIGINAL
        is_stable = test_sindy_stability(model, X_list, U_list, t_list_full)
        
        if not is_stable:
            print("❌ El modelo inferido divergió (Blow-up). Descartando configuración.")
            continue
            
        print("✅ El modelo superó el test de estabilidad. Guardando en Drive...")

        # 6. Generación de Gráficos y Backups
        traj_plot_path = os.path.join(TMP_DIR, "training_trajectories.png")
        hist_plot_path = os.path.join(TMP_DIR, "stlsq_history.png")
        val_plot_path = os.path.join(TMP_DIR, "validation_trajectories.png")
        
        plot_sample_trajectories(X_list, traj_plot_path)
        plot_optimizer_history(model, model.get_feature_names(), hist_plot_path)
        plot_validation_trajectories(model, X_list_full, U_list, t_list_full, val_plot_path)

        models_root_id = create_or_get_subfolder(service, sys_id, MODELS_FOLDER_NAME)
        batch_folder_id = create_or_get_subfolder(service, models_root_id, "rs_batch")
        
        log_name = "rs_log.json"
        log_file_id, full_log = None, []
        
        res = service.files().list(q=f"name='{log_name}' and '{batch_folder_id}' in parents and trashed=false", fields='files(id)').execute()
        if res.get('files'):
            log_file_id = res.get('files')[0]['id']
            local_log = download_file_to_cache(service, log_file_id, log_name)
            try:
                with open(local_log, "r") as f: full_log = json.load(f)
            except json.JSONDecodeError: pass

        version_id = 1 if not full_log else max([entry.get("id", 0) for entry in full_log]) + 1
        version_folder_id = create_or_get_subfolder(service, batch_folder_id, f"v{version_id}")
        
        model_path = os.path.join(TMP_DIR, "sindy_model.joblib")
        params_path = os.path.join(TMP_DIR, "sindy_training_params.json")
        joblib.dump(model, model_path)
        
        # Metadatos del Experimento
        filter_metadata = {
            "selection_mode": sel_mode,
            "target_zone": tgt_zone if sel_mode == "ZONE" else "ALL",
            "manual_range_filter_active": MANUAL_RANGE_FILTER,
            "manual_k_range": MANUAL_K_RANGE if MANUAL_RANGE_FILTER else None,
            "manual_p_range": MANUAL_P_RANGE if MANUAL_RANGE_FILTER else None
        }

        run_metadata = {
            "version_id": version_id,
            "system_name": System.name,
            "timestamp": datetime.now().isoformat(),
            "filters": filter_metadata,
            "keys_used": used_keys,
            "sampling_history": sampling_record,
            "hyperparams": {
                "poly_degree": POLY_DEGREE,
                "threshold": thresh,
                "alpha": alpha,
                "unbias": UNBIAS,
                "t_max_train": float(t_max_train),
                "n_tuples": n_tup,
                "n_trajs_per_tuple": n_traj
            }
        }
        
        with open(params_path, "w") as f: json.dump(run_metadata, f, indent=2, cls=NpEncoder)

        # Uploads
        upload_file_to_drive(service, model_path, "sindy_model.joblib", version_folder_id)
        upload_file_to_drive(service, params_path, "sindy_training_params.json", version_folder_id)
        upload_file_to_drive(service, traj_plot_path, "training_trajectories.png", version_folder_id)
        upload_file_to_drive(service, hist_plot_path, "stlsq_history.png", version_folder_id)
        upload_file_to_drive(service, val_plot_path, "validation_trajectories.png", version_folder_id)
        
        # Actualizar Log Maestro
        log_entry = {
            "id": version_id,
            "path": f"v{version_id}/",
            "timestamp": run_metadata["timestamp"],
            "filters": filter_metadata,
            "unbias": UNBIAS,
            "threshold": thresh,
            "alpha": alpha,
            "t_max_train": float(t_max_train),
            "n_tuples": n_tup,
            "n_trajs_per_tuple": n_traj,
            "total_trajectories_used": len(X_list)
        }
        full_log.append(log_entry)
        
        local_log_path = os.path.join(TMP_DIR, log_name)
        with open(local_log_path, "w") as f: json.dump(full_log, f, indent=2, cls=NpEncoder)
            
        upload_file_to_drive(service, local_log_path, log_name, batch_folder_id, existing_file_id=log_file_id)

    if os.path.exists(TMP_DIR): shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("\n🚀 Búsqueda Aleatoria Completada Exitosamente.")

if __name__ == "__main__":
    main()