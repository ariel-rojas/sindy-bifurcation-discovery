#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script CLOUD-NATIVE de BARRIDO TEMPORAL ESTADÍSTICO (Statistical Time Sweep).

OPTIMIZACIÓN (Lazy Loading + Cloud + Subsampling):
- Conecta a Google Drive y sigue la jerarquía: Sistema -> trayectorias.
- Carga bajo demanda solo los archivos seleccionados para el batch.
- Realiza un SUBMUESTREO ALEATORIO de trayectorias por cada configuración.
- Sube todos los resultados directamente a Drive organizados en subcarpetas 
  dinámicas paralelas a los datos:
  Base -> [Sistema] -> BATCH_[Y]_statistical_sweep -> [X]_trajs -> zone_[Z]
- Detecta y recorta automáticamente los tiempos de evaluación (TIME_STEPS) 
  si exceden el tiempo máximo de los datos.
"""

import os
import sys
import time
import json
import random
import tempfile
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import io
import uuid # <-- IMPORTANTE: Para aislar la caché entre corridas
import shutil

from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

# from systems.takens_bogdanov import TakensBogdanov as System
from systems.cubic_symmetric_takens_bogdanov import CubicSymmetricTakensBogdanov as System

from experiments.takens_bogdanov.data_zone_manager import DataZoneManager
from core.io import parse_param_key
from core.drive_auth import get_drive_service, get_target_folder_id

# =============================================================================
# PARÁMETROS GENERALES
# =============================================================================
N_TRAJS_LIST = [10]        # Lista de trayectorias a submuestrear
BATCH_SIZE_LIST = [1]    # Lista de configuraciones (tuplas) por iteración

TARGET_ZONES = [1,2,3,4,5]      
N_BOOTSTRAP_ITERS = 10   
SAVE_ALL_MODELS = False

# SINDy
POLY_DEGREE = 3
THRESHOLD = 0.1
TIME_STEPS = np.arange(0.05, 20.1, 0.1)

# Teoría Desacoplada (Se extrae dinámicamente del sistema)
THEORETICAL_MAP = {i: eq for i, eq in enumerate(System.get_true_coefficients())}

# --- CACHÉ AISLADO Y ÚNICO ---
RUN_ID = str(uuid.uuid4())[:8]
TMP_DIR = os.path.join(tempfile.gettempdir(), f"sindy_sweep_cache_{RUN_ID}")
os.makedirs(TMP_DIR, exist_ok=True)

# =============================================================================
# FUNCIONES DE GOOGLE DRIVE Y CACHÉ (A PRUEBA DE FALLOS)
# =============================================================================
def create_or_get_subfolder(service, parent_id, folder_name):
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if not items:
        metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_id]}
        return service.files().create(body=metadata, fields='id').execute().get('id')
    return items[0]['id']

def build_drive_index(service, folder_id):
    index = {}
    page_token = None
    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive', fields='nextPageToken, files(id, name)',
            pageToken=page_token, pageSize=1000
        ).execute()
        for item in results.get('files', []):
            index[item['name']] = item['id']
        page_token = results.get('nextPageToken', None)
        if not page_token: break
    return index

def download_file_to_cache(service, file_id, file_name):
    local_path = os.path.join(TMP_DIR, file_name)
    if os.path.exists(local_path): return local_path 
    
    for attempt in range(4): # Reintentos anti-microcortes
        try:
            request = service.files().get_media(fileId=file_id)
            fh = io.FileIO(local_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return local_path
        except Exception as e:
            try: fh.close()
            except: pass
            if attempt < 3: time.sleep(2 ** attempt)
            else: raise ConnectionError(f"Fallo crítico al descargar {file_name}: {e}")

def upload_to_drive(service, file_path, file_name, folder_id):
    metadata = {'name': file_name, 'parents': [folder_id]}
    for attempt in range(4): # Reintentos anti-microcortes
        try:
            media = MediaFileUpload(file_path, mimetype='application/octet-stream', resumable=True)
            request = service.files().create(media_body=media, body=metadata)
            response = None
            while response is None:
                _, response = request.next_chunk()
            if os.path.exists(file_path):
                try: os.remove(file_path)
                except: pass
            return True
        except Exception as e:
            if attempt < 3: time.sleep(2 ** attempt)
            else: 
                print(f"Fallo al subir {file_name}: {e}")
                return False

# =============================================================================
# UTILIDADES MATEMÁTICAS Y DE PLOTEO
# =============================================================================
def normalize_term_name(name):
    return " ".join(sorted(name.split()))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_zone_keys(manager, zone_id, batch_size):
    print(f"\n[Info] Identificando claves en Zona {zone_id}...")
    available_keys = manager.get_samples_from_zone(zone_id, n_samples=999999)
    if len(available_keys) < batch_size:
        raise ValueError(f"Zona {zone_id} insuficiente ({len(available_keys)} vs requeridos {batch_size}).")
    print(f"[Info] Se encontraron {len(available_keys)} configuraciones en Zona {zone_id}.")
    return available_keys

def plot_phase_portrait_snapshot(X_batch, t_max, drive_folder_id, service, zone_id, n_groups):
    total_trajs = len(X_batch)
    if total_trajs % n_groups != 0:
        print(f"[Warning Plot] Total trayectorias ({total_trajs}) no es múltiplo de grupos ({n_groups}).")
        return

    trajs_per_group = total_trajs // n_groups
    print(f"   > Generando {n_groups} plots de diagnóstico y subiéndolos a Drive (T={t_max:.1f}s)...")

    for group_idx in range(n_groups):
        start_i = group_idx * trajs_per_group
        end_i = (group_idx + 1) * trajs_per_group
        group_trajs = X_batch[start_i : end_i]
        
        params = group_trajs[0][0, -2:]
        mu1_val, mu2_val = params[0], params[1]

        plt.figure(figsize=(6, 5))
        for traj in group_trajs:
            plt.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.4, lw=0.3)
            plt.scatter(traj[0, 0], traj[0, 1], c='green', s=5, marker='o', alpha=0.6)
            plt.scatter(traj[-1, 0], traj[-1, 1], c='red', s=10, marker='x', alpha=0.8)
        
        plt.title(f"Zona {zone_id} | T={t_max:.1f}s\nGrupo {group_idx+1}: $\mu_1={mu1_val:.4f}, \mu_2={mu2_val:.4f}$")
        plt.xlabel(System.state_names[0])
        plt.ylabel(System.state_names[1])
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        filename = f"portrait_t_{t_max:04.1f}_group_{group_idx+1}.png"
        local_path = os.path.join(TMP_DIR, filename)
        plt.savefig(local_path, dpi=120)
        plt.close()
        
        upload_to_drive(service, local_path, filename, drive_folder_id)

def load_batch_data_from_disk(service, drive_index, selected_keys, n_trajs_to_sample):
    batch_data = []
    sampling_record = {} 
    
    for key in selected_keys:
        try:
            file_name = f"{key}.npz"
            local_path = download_file_to_cache(service, drive_index[file_name], file_name)
            
            with np.load(local_path) as data:
                trajs = data["trajectories"]
                
            param_vals = parse_param_key(key)
            n_sims, n_dim, n_time = trajs.shape
            
            actual_n_trajs = min(n_trajs_to_sample, n_sims)
            selected_indices = random.sample(range(n_sims), actual_n_trajs)
            sampling_record[key] = selected_indices
            
            param_block = np.tile(param_vals, (n_time, 1))
            
            for i in selected_indices:
                traj_T = trajs[i].T
                X_aug = np.hstack((traj_T, param_block))
                batch_data.append(X_aug)
                
            # Limpieza ZERO FOOTPRINT (Borrando caché al instante)
            if os.path.exists(local_path):
                os.remove(local_path)
                
        except Exception as e:
            print(f"[Error] No se pudo cargar key {key}: {e}")
            
    return batch_data, sampling_record

# =============================================================================
# LÓGICA ESTADÍSTICA DE SINDy
# =============================================================================
def run_statistical_analysis(unbias_flag, all_keys, drive_zone_id, service, drive_index, t_full, zone_id, batch_size, n_trajs):
    mode_name = f"Unbias={unbias_flag}"
    print(f"\n{'='*60}\nINICIANDO ANÁLISIS: {mode_name}\n{'='*60}")
    
    # --- RECORTE AUTOMÁTICO DE TIME_STEPS ---
    max_t_available = t_full[-1]
    active_time_steps = np.array([t for t in TIME_STEPS if t <= max_t_available])
    
    if len(active_time_steps) == 0:
        active_time_steps = np.array([max_t_available])
        print(f"Aviso: El vector tiempo es muy corto. Usando {max_t_available}s.")
    elif len(active_time_steps) < len(TIME_STEPS):
        print(f"Aviso: La data de entrada termina en {max_t_available}s. Acortando el barrido.")
    
    # --- LÓGICA DE GUARDADO DE MODELOS ---
    models_folder_id = None
    if SAVE_ALL_MODELS:
        models_folder_id = create_or_get_subfolder(service, drive_zone_id, f"models_unbias_{unbias_flag}")
        print(f"  > Guardado de modelos activado en Drive.")

    raw_data = {0: {}, 1: {}}
    start_time = time.time()
    sampling_history = {}
    
    for n_iter in range(N_BOOTSTRAP_ITERS):
        current_keys = random.sample(all_keys, batch_size)
        
        batch_data, sampled_indices = load_batch_data_from_disk(service, drive_index, current_keys, n_trajs)
        sampling_history[f"iter_{n_iter}"] = sampled_indices
        
        desc = f"Iter {n_iter+1}/{N_BOOTSTRAP_ITERS}"
        
        for t_idx, t_max in enumerate(tqdm(active_time_steps, desc=desc, leave=False)):
            
            valid_idx = np.where(t_full <= t_max)[0]
            limit = valid_idx[-1] + 1
            t_slice = t_full[:limit]
            X_batch_slice = [x[:limit, :] for x in batch_data]

            X_dot_slice = []
            for x_traj in X_batch_slice:
                d_traj = np.gradient(x_traj, t_slice, axis=0)
                X_dot_slice.append(d_traj)
            
            optimizer = ps.STLSQ(threshold=THRESHOLD, unbias=unbias_flag)
            feature_library = ps.PolynomialLibrary(degree=POLY_DEGREE)
            feature_names = System.state_names + System.param_names
            
            model = ps.SINDy(
                optimizer=optimizer,
                feature_library=feature_library,
                differentiation_method=ps.FiniteDifference() 
            )
            
            model.fit(X_batch_slice, x_dot=X_dot_slice, t=t_slice, feature_names=feature_names)
            
            # --- GUARDADO INDIVIDUAL DEL MODELO (SI APLICA) ---
            if SAVE_ALL_MODELS and models_folder_id:
                model_filename = f"model_iter_{n_iter}_t_{t_max:.1f}.joblib"
                model_local_path = os.path.join(TMP_DIR, model_filename)
                joblib.dump(model, model_local_path)
                upload_to_drive(service, model_local_path, model_filename, models_folder_id)
            # --------------------------------------------------

            feats = model.get_feature_names()
            coeffs = model.optimizer.coef_
            
            for target_i in range(2):
                for feat_i, feat_name in enumerate(feats):
                    norm_name = normalize_term_name(feat_name)
                    if norm_name not in raw_data[target_i]:
                        raw_data[target_i][norm_name] = [[] for _ in active_time_steps]
                    raw_data[target_i][norm_name][t_idx].append(coeffs[target_i, feat_i])

    print(f"\n[Procesando] Calculando estadísticas...")
    stats_results = {0: {}, 1: {}}
    
    for target_i in range(2):
        for term, values_over_time in raw_data[target_i].items():
            means, stds = [], []
            for vals_at_t in values_over_time:
                pad_len = N_BOOTSTRAP_ITERS - len(vals_at_t)
                padded_vals = vals_at_t + [0.0] * pad_len
                means.append(np.mean(padded_vals))
                stds.append(np.std(padded_vals))
            stats_results[target_i][term] = {'mean': means, 'std': stds}

    print(f"[Fin] {mode_name} completado en {time.time() - start_time:.2f}s")
    
    plot_statistical_results(stats_results, unbias_flag, drive_zone_id, service, zone_id, active_time_steps, batch_size)
    save_metadata(stats_results, unbias_flag, drive_zone_id, service, zone_id, sampling_history, active_time_steps, batch_size, n_trajs)

def save_metadata(stats, unbias_flag, drive_folder_id, service, zone_id, sampling_history, time_steps, batch_size, n_trajs):
    mode_str = "TRUE" if unbias_flag else "FALSE"
    filename = f"stats_unbias_{mode_str}.json"
    local_path = os.path.join(TMP_DIR, filename)
    
    meta = {
        "config": {
            "zone": zone_id, 
            "iters": N_BOOTSTRAP_ITERS, 
            "batch": batch_size,         
            "times": time_steps, 
            "n_trajs_per_param": n_trajs 
        },
        "sampling_history": sampling_history, 
        "results": stats
    }
    with open(local_path, "w") as f:
        json.dump(meta, f, indent=4, cls=NumpyEncoder)
        
    upload_to_drive(service, local_path, filename, drive_folder_id)
    print(f"[IO] JSON de resultados subido a Drive en Zona {zone_id}")

def plot_statistical_results(stats, unbias_flag, drive_folder_id, service, zone_id, time_steps, batch_size):
    mode_str = "TRUE" if unbias_flag else "FALSE"
    print(f"[Plot] Generando gráfico estadístico...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    targets = ["x'", "y'"]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for target_i in range(2):
        ax = axes[target_i]
        hist_dict = stats[target_i]
        theo_dict = THEORETICAL_MAP[target_i]
        theo_norm = {normalize_term_name(k): v for k, v in theo_dict.items()}
        color_idx = 0
        
        for term, data in hist_dict.items():
            means = np.array(data['mean'])
            stds = np.array(data['std'])
            is_theo = term in theo_norm
            final_mean = means[-1]
            final_std = stds[-1]
            
            if is_theo or np.max(np.abs(means)) > 0.05:
                if is_theo:
                    c = colors[color_idx % 10]; color_idx += 1
                    label = f"{term}: {final_mean:.3f} $\\pm$ {final_std:.3f}"
                    style = '-'; alpha=1.0; width=2.5; fill_alpha=0.2
                else:
                    c = 'gray'
                    label = f"{term} (Espurio)"
                    style = ':'; alpha=0.6; width=1.5; fill_alpha=0.1
                
                ax.plot(time_steps, means, label=label, color=c, ls=style, lw=width, alpha=alpha)
                ax.fill_between(time_steps, means - stds, means + stds, color=c, alpha=fill_alpha)
                
                if is_theo:
                    ax.axhline(theo_norm[term], color=c, ls='--', alpha=0.5, lw=1)
        
        ax.set_title(f"Ecuación {targets[target_i]} (Zona {zone_id})")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Coeficiente")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize='x-small', title="Final (Media $\\pm$ Std)")

    plt.suptitle(f"Estabilidad SINDy (Zona {zone_id}, Unbias={mode_str})\n{N_BOOTSTRAP_ITERS} iters, {batch_size} grupos/iter", fontsize=16)
    plt.tight_layout()
    
    filename = f"stats_plot_zone_{zone_id}_unbias_{mode_str}.png"
    local_path = os.path.join(TMP_DIR, filename)
    plt.savefig(local_path, dpi=150)
    plt.close()
    
    upload_to_drive(service, local_path, filename, drive_folder_id)

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def main():
    try:
        service = get_drive_service()
        base_folder_id = get_target_folder_id()
        
        system_folder_name = System.name.lower().replace("-", "_").replace(" ", "_")
        system_folder_id = create_or_get_subfolder(service, base_folder_id, system_folder_name)
        
        traj_folder_id = create_or_get_subfolder(service, system_folder_id, "trajectories")
        
        drive_index = build_drive_index(service, traj_folder_id)
        
        print("Descargando vector de tiempo (t_eval.npz)...")
        teval_path = download_file_to_cache(service, drive_index["t_eval.npz"], "t_eval.npz")
        with np.load(teval_path) as data:
            t_eval_global = data["t_eval"]
            
        manager = DataZoneManager(system_class=System, service=service, target_folder_id=traj_folder_id)
        
    except Exception as e:
        print(f"Error fatal de conexión/estructura en Drive: {e}")
        return

    for b_size in BATCH_SIZE_LIST:
        batch_folder_name = f"BATCH_{b_size}_statistical_sweep"
        batch_folder_id = create_or_get_subfolder(service, system_folder_id, batch_folder_name)
        
        for n_trajs in N_TRAJS_LIST:
            trajs_folder_name = f"{n_trajs}_trajs"
            sweep_folder_id = create_or_get_subfolder(service, batch_folder_id, trajs_folder_name)
            
            for zone_id in TARGET_ZONES:
                print(f"\n{'#'*60}")
                print(f"PROCESANDO: Batch={b_size} | Trajs={n_trajs} | ZONA={zone_id}")
                print(f"{'#'*60}")
                
                zone_folder_id = create_or_get_subfolder(service, sweep_folder_id, f"zone_{zone_id}")
                
                try:
                    all_keys = get_zone_keys(manager, zone_id, b_size)
                except Exception as e:
                    print(f"Error en Zona {zone_id}: {e}")
                    continue
                    
                run_statistical_analysis(True, all_keys, zone_folder_id, service, drive_index, t_eval_global, zone_id, b_size, n_trajs)
                run_statistical_analysis(False, all_keys, zone_folder_id, service, drive_index, t_eval_global, zone_id, b_size, n_trajs)
    
    # Limpieza Total del TMP Único
    print("\nLimpiando archivos temporales locales...")
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    print("\n=== FINALIZADO TODAS LAS PRUEBAS ESTADÍSTICAS CLOUD-NATIVE ===")

if __name__ == "__main__":
    main()