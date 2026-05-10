#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de RE-GRAFICADO ESTÉTICO Y TEMPORAL.

Este script lee los archivos JSON de resultados estadísticos generados previamente,
aplica filtros de tiempo (MAX_TIME_EVAL) y regenera los gráficos de coeficientes
vs tiempo sin necesidad de volver a correr SINDy.
"""

import os
import sys
import time
import json
import tempfile
import uuid
import shutil
import numpy as np
import matplotlib.pyplot as plt
import io

from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# =============================================================================
# CONFIGURACIÓN DE RUTAS E IMPORTS
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System
# from systems.cubic_symmetric_takens_bogdanov import CubicSymmetricTakensBogdanov as System
from core.drive_auth import get_drive_service, get_target_folder_id

# =============================================================================
# PARÁMETROS DE BÚSQUEDA Y ESTÉTICA
# =============================================================================
# Define qué carpetas quieres procesar
TARGET_BATCHES = [1,2,3]
N_TRAJS_LIST = [1,2,3,4, 5,6, 7, 8, 9,10]
TARGET_ZONES = [1, 2, 3, 4]

# PARÁMETRO CLAVE: Tiempo máximo a mostrar en el gráfico (None para mostrar todo)
MAX_TIME_EVAL = 5

# Teoría Desacoplada
def normalize_term_name(name):
    return " ".join(sorted(name.split()))

THEORETICAL_MAP = {
    i: {normalize_term_name(k): v for k, v in eq.items()} 
    for i, eq in enumerate(System.get_true_coefficients())
}

RUN_ID = str(uuid.uuid4())[:8]
TMP_DIR = os.path.join(tempfile.gettempdir(), f"replot_cache_{RUN_ID}")
os.makedirs(TMP_DIR, exist_ok=True)

# =============================================================================
# FUNCIONES DE GOOGLE DRIVE
# =============================================================================
def get_folder_id_by_name(service, parent_id, folder_name):
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
    items = results.get('files', [])
    return items[0]['id'] if items else None

def get_files_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    return {item['name']: item['id'] for item in results.get('files', [])}

def download_file_to_cache(service, file_id, file_name):
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
            else: raise ConnectionError(f"Fallo al descargar {file_name}: {e}")

def upload_to_drive(service, file_path, file_name, folder_id):
    query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
    existing_file_id = None
    try:
        results = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        items = results.get('files', [])
        if items: existing_file_id = items[0]['id']
    except: pass

    metadata = {'name': file_name, 'parents': [folder_id]}
    for attempt in range(4):
        try:
            media = MediaFileUpload(file_path, mimetype='image/png', resumable=True)
            if existing_file_id:
                request = service.files().update(fileId=existing_file_id, media_body=media)
            else:
                request = service.files().create(media_body=media, body=metadata)
            
            response = None
            while response is None: _, response = request.next_chunk()
            if os.path.exists(file_path): os.remove(file_path)
            return True
        except Exception as e:
            if attempt < 3: time.sleep(2 ** attempt)
            else: return False

# =============================================================================
# FUNCIÓN DE PLOTEO (Adaptada para recortes temporales)
# =============================================================================
def replot_statistical_results(stats, unbias_flag, drive_folder_id, service, zone_id, time_steps, batch_size, n_iters, traj_name):
    mode_str = "TRUE" if unbias_flag else "FALSE"
    
    # --- APLICAR RECORTE TEMPORAL ---
    time_steps = np.array(time_steps)
    if MAX_TIME_EVAL is not None:
        valid_idx = time_steps <= MAX_TIME_EVAL
        time_steps = time_steps[valid_idx]
        
        # Recortar también los arrays dentro del JSON
        for target_i in ["0", "1"]:
            if target_i in stats:
                for term in stats[target_i]:
                    stats[target_i][term]['mean'] = np.array(stats[target_i][term]['mean'])[valid_idx]
                    stats[target_i][term]['std'] = np.array(stats[target_i][term]['std'])[valid_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    targets = ["x'", "y'"]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for target_i_int in range(2):
        target_i = str(target_i_int)
        ax = axes[target_i_int]
        if target_i not in stats: continue
        
        hist_dict = stats[target_i]
        theo_dict = THEORETICAL_MAP[target_i_int]
        color_idx = 0
        
        for term, data in hist_dict.items():
            means = np.array(data['mean'])
            stds = np.array(data['std'])
            if len(means) == 0: continue 
            
            is_theo = term in theo_dict
            final_mean = means[-1]
            final_std = stds[-1]
            
            if is_theo or np.max(np.abs(means)) > 0.05:
                if is_theo:
                    c = colors[color_idx % 10]; color_idx += 1
                    label = f"{term}: {final_mean:.3f} $\\pm$ {final_std:.3f}"
                    style = '-'; alpha = 0.8; width = 2.5; fill_alpha = 0.1
                else:
                    c = 'gray'
                    label = f"{term} (Espurio)"
                    style = ':'; alpha = 0.6; width = 1.5; fill_alpha = 0.1
                
                ax.plot(time_steps, means, label=label, color=c, ls=style, lw=width, alpha=alpha)
                ax.fill_between(time_steps, means - stds, means + stds, color=c, alpha=fill_alpha)
                
                if is_theo:
                    ax.axhline(theo_dict[term], color=c, ls='--', alpha=1, lw=2.5)
        
        ax.set_title(f"Ecuación {targets[target_i_int]} (Zona {zone_id})")
        ax.set_xlabel("Tiempo (s)", fontsize=12)
        ax.set_ylabel("Valor del Coeficiente", fontsize=12)
        
        # --- LÍMITES DE EJES (MOVIDOS AQUÍ) ---
        if MAX_TIME_EVAL is not None:
            ax.set_xlim(0, MAX_TIME_EVAL)
            
        if target_i_int == 0:
            ax.set_ylim(-0.5, 1.5)
        elif target_i_int == 1:
            ax.set_ylim(-1.5, 1.5)
        
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize='x-small', title="Final (Media $\\pm$ Std)")

    time_str = f" (t ≤ {MAX_TIME_EVAL}s)" if MAX_TIME_EVAL else ""
    plt.suptitle(f"Estabilidad SINDy - Zona {zone_id} | {traj_name.replace('_', ' ')}{time_str}\n(Unbias={mode_str} | {n_iters} iters | {batch_size} batch)", fontsize=16)
    plt.tight_layout()
    
    filename = f"stats_plot_zone_{zone_id}_unbias_{mode_str}.png"
    local_path = os.path.join(TMP_DIR, filename)
    plt.savefig(local_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"      > Actualizando {filename} en Drive...")
    upload_to_drive(service, local_path, filename, drive_folder_id)

# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================
def main():
    print("=== INICIANDO RE-GRAFICADO ESTÉTICO/TEMPORAL ===")
    service = get_drive_service()
    base_folder_id = get_target_folder_id()
    
    system_folder_name = System.name.lower().replace("-", "_").replace(" ", "_")
    system_folder_id = get_folder_id_by_name(service, base_folder_id, system_folder_name)
    
    if not system_folder_id:
        print("[Error] Carpeta del sistema no encontrada.")
        return

    for b_size in TARGET_BATCHES:
        batch_folder_name = f"BATCH_{b_size}_statistical_sweep"
        batch_folder_id = get_folder_id_by_name(service, system_folder_id, batch_folder_name)
        
        if not batch_folder_id:
            print(f"[Aviso] No se encontró {batch_folder_name}. Omitiendo.")
            continue
            
        print(f"\n[{batch_folder_name}]")
        
        for n_trajs in N_TRAJS_LIST:
            trajs_folder_name = f"{n_trajs}_trajs"
            trajs_folder_id = get_folder_id_by_name(service, batch_folder_id, trajs_folder_name)
            
            if not trajs_folder_id: continue
                
            for zone_id in TARGET_ZONES:
                zone_folder_name = f"zone_{zone_id}"
                zone_folder_id = get_folder_id_by_name(service, trajs_folder_id, zone_folder_name)
                
                if not zone_folder_id: continue
                
                print(f"  -> Procesando {trajs_folder_name} / {zone_folder_name}...")
                files_in_zone = get_files_in_folder(service, zone_folder_id)
                
                for unbias_flag in [True, False]:
                    mode_str = "TRUE" if unbias_flag else "FALSE"
                    json_filename = f"stats_unbias_{mode_str}.json"
                    
                    if json_filename in files_in_zone:
                        cache_name = f"regen_{b_size}_{n_trajs}_{zone_id}_{json_filename}"
                        local_json = download_file_to_cache(service, files_in_zone[json_filename], cache_name)
                        
                        try:
                            with open(local_json, 'r') as f:
                                data = json.load(f)
                            
                            time_steps = data['config']['times']
                            stats_results = data['results']
                            n_iters = data['config']['iters']
                            
                            replot_statistical_results(
                                stats=stats_results, 
                                unbias_flag=unbias_flag, 
                                drive_folder_id=zone_folder_id, 
                                service=service, 
                                zone_id=zone_id, 
                                time_steps=time_steps, 
                                batch_size=b_size, 
                                n_iters=n_iters,
                                traj_name=trajs_folder_name
                            )
                        except Exception as e:
                            print(f"      [Error] Fallo al procesar {json_filename}: {e}")

    print("\nLimpiando caché local...")
    if os.path.exists(TMP_DIR): shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("\n=== RE-GRAFICADO COMPLETADO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()