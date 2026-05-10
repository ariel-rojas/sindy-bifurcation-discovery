#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de POST-PROCESAMIENTO: Rendimiento Óptimo vs Cantidad de Datos.

Mejoras incluidas:
- Filtro de Tiempo Máximo (MAX_TIME_EVAL) para limitar la búsqueda del error óptimo.
- Consolidación de Error Mínimo vs Cantidad de Trayectorias.
- Histograma de Tiempos Óptimos.
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

# from systems.cubic_symmetric_takens_bogdanov import CubicSymmetricTakensBogdanov as System
from systems.takens_bogdanov import TakensBogdanov as System
from core.drive_auth import get_drive_service, get_target_folder_id

# =============================================================================
# PARÁMETROS DEL SCRIPT
# =============================================================================
TARGET_BATCH = 3
TARGET_ZONES = [1, 2, 3, 4]

# NUEVO PARÁMETRO: Tiempo máximo a evaluar. 
# Ponlo en None para evaluar toda la curva, o un valor float (ej. 10.0)
MAX_TIME_EVAL = 5  

def normalize_term_name(name):
    return " ".join(sorted(name.split()))

THEORETICAL_MAP = {
    i: {normalize_term_name(k): v for k, v in eq.items()} 
    for i, eq in enumerate(System.get_true_coefficients())
}

COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'} 
LINESTYLES = {True: '-', False: '--'} 

RUN_ID = str(uuid.uuid4())[:8]
TMP_DIR = os.path.join(tempfile.gettempdir(), f"optimal_sweep_cache_{RUN_ID}")
os.makedirs(TMP_DIR, exist_ok=True)

# =============================================================================
# FUNCIONES DE GOOGLE DRIVE 
# =============================================================================
def get_folder_id_by_name(service, parent_id, folder_name):
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
    items = results.get('files', [])
    return items[0]['id'] if items else None

def get_subfolders(service, parent_id):
    query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    return results.get('files', [])

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
                
            if os.path.exists(file_path):
                try: os.remove(file_path)
                except: pass
            return True
        except Exception as e:
            if attempt < 3: time.sleep(2 ** attempt)
            else: return False

# =============================================================================
# LÓGICA MATEMÁTICA
# =============================================================================
def calculate_global_relative_error(stats_results):
    time_len = 0
    all_terms = {0: set(THEORETICAL_MAP[0].keys()), 1: set(THEORETICAL_MAP[1].keys())}
    for eq_str in ["0", "1"]:
        if eq_str in stats_results:
            eq_int = int(eq_str)
            all_terms[eq_int].update(stats_results[eq_str].keys())
            for term, data in stats_results[eq_str].items():
                time_len = len(data['mean'])
    
    if time_len == 0: return []

    errors = []
    for t_idx in range(time_len):
        num, den = 0.0, 0.0
        for eq_int in [0, 1]:
            eq_str = str(eq_int)
            for term in all_terms[eq_int]:
                theo_val = THEORETICAL_MAP[eq_int].get(term, 0.0)
                emp_val = 0.0
                if eq_str in stats_results and term in stats_results[eq_str]:
                    emp_val = stats_results[eq_str][term]['mean'][t_idx]
                
                num += (emp_val - theo_val)**2
                den += theo_val**2
                
        error_t = np.sqrt(num) / np.sqrt(den) if den > 0 else np.sqrt(num)
        errors.append(error_t)
        
    return errors

# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================
def main():
    print(f"=== INICIANDO CONSOLIDACIÓN ÓPTIMA (BATCH {TARGET_BATCH}) ===")
    if MAX_TIME_EVAL is not None:
        print(f"[*] Filtro de tiempo activado: Solo se evaluará hasta t = {MAX_TIME_EVAL}s")
        
    service = get_drive_service()
    base_folder_id = get_target_folder_id()
    
    system_folder_name = System.name.lower().replace("-", "_").replace(" ", "_")
    system_folder_id = get_folder_id_by_name(service, base_folder_id, system_folder_name)
    batch_folder_name = f"BATCH_{TARGET_BATCH}_statistical_sweep"
    batch_folder_id = get_folder_id_by_name(service, system_folder_id, batch_folder_name)
    
    trajs_folders_raw = get_subfolders(service, batch_folder_id)
    
    # Filtrar y ordenar las carpetas numéricamente
    trajs_folders = []
    for f in trajs_folders_raw:
        try:
            num = int(f['name'].split('_')[0])
            trajs_folders.append((num, f))
        except ValueError:
            pass
    trajs_folders.sort(key=lambda x: x[0]) 
    
    summary_data = {
        (z, b): {'n_trajs': [], 'min_errors': []} 
        for z in TARGET_ZONES for b in [True, False]
    }
    
    all_optimal_times = []

    print(f"Se procesarán {len(trajs_folders)} carpetas de trayectorias en orden.")

    for n_trajs, traj_folder in trajs_folders:
        traj_name = traj_folder['name']
        traj_id = traj_folder['id']
        print(f" > Extrayendo óptimos de: {traj_name}...")

        zones_folders = get_subfolders(service, traj_id)
        
        for zone_folder in zones_folders:
            zone_name = zone_folder['name']
            if not zone_name.startswith("zone_"): continue
            zone_id = int(zone_name.split("_")[1])
            
            if zone_id not in TARGET_ZONES: continue
            files_in_zone = get_files_in_folder(service, zone_folder['id'])
            
            for unbias_flag in [True, False]:
                mode_str = "TRUE" if unbias_flag else "FALSE"
                json_filename = f"stats_unbias_{mode_str}.json"
                
                if json_filename in files_in_zone:
                    unique_cache_name = f"agg_{traj_name}_{zone_name}_{json_filename}"
                    local_json = download_file_to_cache(service, files_in_zone[json_filename], unique_cache_name)
                    
                    with open(local_json, 'r') as f:
                        data = json.load(f)
                    
                    time_steps = np.array(data['config']['times'])
                    stats_results = data['results']
                    
                    errors = np.array(calculate_global_relative_error(stats_results))
                    
                    if len(errors) > 0:
                        # --- NUEVA LÓGICA DE RECORTE TEMPORAL ---
                        if MAX_TIME_EVAL is not None:
                            # Buscar solo los índices donde el tiempo es menor o igual al máximo
                            valid_indices = np.where(time_steps <= MAX_TIME_EVAL)[0]
                            if len(valid_indices) == 0:
                                print(f"    [Warning] Ningún tiempo es <= {MAX_TIME_EVAL} en {traj_name}/{zone_name}. Se omite.")
                                continue
                            
                            # Extraer el mínimo relativo solo a esa ventana de tiempo
                            valid_errors = errors[valid_indices]
                            min_idx_relative = np.argmin(valid_errors)
                            min_idx = valid_indices[min_idx_relative]
                        else:
                            # Lógica original (toda la curva)
                            min_idx = np.argmin(errors)

                        # Guardar los valores encontrados
                        min_err = errors[min_idx]
                        optimal_time = time_steps[min_idx]
                        
                        summary_data[(zone_id, unbias_flag)]['n_trajs'].append(n_trajs)
                        summary_data[(zone_id, unbias_flag)]['min_errors'].append(min_err)
                        all_optimal_times.append(optimal_time)

    # =========================================================================
    # GENERACIÓN DE GRÁFICOS
    # =========================================================================
    print("\nGenerando gráficos consolidados...")
    time_constraint_str = f" (t ≤ {MAX_TIME_EVAL}s)" if MAX_TIME_EVAL else ""
    
    # --- GRÁFICO 1: Error vs Trayectorias ---
    plt.figure(figsize=(10, 6))
    for (zone_id, unbias_flag), data in summary_data.items():
        if len(data['n_trajs']) == 0: continue
        
        label = f"Zona {zone_id} | {'Unbias' if unbias_flag else 'Biased'}"
        color = COLORS.get(zone_id, 'black')
        linestyle = LINESTYLES[unbias_flag]
        
        plt.plot(data['n_trajs'], data['min_errors'], marker='o', markersize=6, 
                 color=color, linestyle=linestyle, linewidth=2, label=label, alpha=0.8)
            
    plt.title(f"Error Mínimo Global vs Cantidad de Trayectorias{time_constraint_str}\n(Batch Size: {TARGET_BATCH})", fontsize=14)
    plt.xlabel("Número de Trayectorias", fontsize=12)
    plt.ylabel(f"Error Mínimo Global Alcanzado ($L_2$)", fontsize=12)
    plt.yscale('log')
    plt.ylim(0, 1)
    plt.xticks(sorted(list(set([n for d in summary_data.values() for n in d['n_trajs']]))))
    plt.yticks([1e0, 1e-1, 1e-2, 1e-3, 1e-4])
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    plot1_filename = f"optimal_error_vs_trajs_BATCH_{TARGET_BATCH}.png"
    plot1_local = os.path.join(TMP_DIR, plot1_filename)
    plt.savefig(plot1_local, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Subiendo {plot1_filename} a la carpeta principal del Batch...")
    upload_to_drive(service, plot1_local, plot1_filename, batch_folder_id)

    # --- GRÁFICO 2: Histograma de Tiempos Óptimos ---
    if len(all_optimal_times) > 0:
        plt.figure(figsize=(9, 5))
        
        plt.hist(all_optimal_times, bins=20, color='mediumpurple', edgecolor='black', alpha=0.7)
        median_time = np.median(all_optimal_times)
        plt.axvline(median_time, color='red', linestyle='dashed', linewidth=2, label=f'Mediana: {median_time:.2f}s')
        
        plt.title(f"Distribución de Tiempos de Evaluación Óptimos{time_constraint_str}\n(Agregado de todas las Zonas y Trayectorias | Batch: {TARGET_BATCH})", fontsize=14)
        plt.xlabel("Tiempo Óptimo (s)", fontsize=12)
        plt.ylabel("Frecuencia", fontsize=12)
        
        if MAX_TIME_EVAL:
            plt.xlim(0, MAX_TIME_EVAL * 1.05) # Ajustar el eje X para que coincida con el recorte
            
        plt.grid(True, axis='y', ls="--", alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        plot2_filename = f"optimal_times_histogram_BATCH_{TARGET_BATCH}.png"
        plot2_local = os.path.join(TMP_DIR, plot2_filename)
        plt.savefig(plot2_local, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Subiendo {plot2_filename} a la carpeta principal del Batch...")
        upload_to_drive(service, plot2_local, plot2_filename, batch_folder_id)

    print("\nLimpiando caché local...")
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        
    print("\n=== CONSOLIDACIÓN FINALIZADA EXITOSAMENTE ===")

if __name__ == "__main__":
    main()