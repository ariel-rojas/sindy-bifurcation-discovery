#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de POST-PROCESAMIENTO: Comparación de Trade-off (Tuplas vs Trayectorias).

Evalúa y compara el Error Relativo Global de 3 configuraciones equivalentes (6 trayectorias totales):
1. 2 tuplas con 3 trayectorias (BATCH_2 -> 3_trajs)
2. 3 tuplas con 2 trayectorias (BATCH_3 -> 2_trajs)
3. 1 tupla con 6 trayectorias  (BATCH_1 -> 6_trajs)

Los gráficos se guardan localmente en la carpeta base.
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

from googleapiclient.http import MediaIoBaseDownload

# =============================================================================
# CONFIGURACIÓN DE RUTAS E IMPORTS 
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

# from systems.takens_bogdanov import TakensBogdanov as System
from systems.cubic_symmetric_takens_bogdanov import CubicSymmetricTakensBogdanov as System
from core.drive_auth import get_drive_service, get_target_folder_id

# =============================================================================
# PARÁMETROS DEL SCRIPT
# =============================================================================
TARGET_ZONES = [1, 2, 3, 4,5]
MAX_TIME_EVAL = 5  

# Configuraciones a comparar: (Batch_ID, Nombre_Carpeta_Traj, Etiqueta, Color)
CONFIGURATIONS = [
    (1, "6_trajs", "1 Tupla x 6 Trajs", '#d62728'),  # Rojo
    (2, "3_trajs", "2 Tuplas x 3 Trajs", '#1f77b4'),  # Azul
    (3, "2_trajs", "3 Tuplas x 2 Trajs", '#2ca02c')   # Verde
]

LINESTYLES = {True: '-', False: '--'} 

RUN_ID = str(uuid.uuid4())[:8]
TMP_DIR = os.path.join(tempfile.gettempdir(), f"tradeoff_cache_{RUN_ID}")
os.makedirs(TMP_DIR, exist_ok=True)

def normalize_term_name(name):
    return " ".join(sorted(name.split()))

THEORETICAL_MAP = {
    i: {normalize_term_name(k): v for k, v in eq.items()} 
    for i, eq in enumerate(System.get_true_coefficients())
}

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
            while not done:
                _, done = downloader.next_chunk()
            return local_path
        except Exception as e:
            try: fh.close()
            except: pass
            if attempt < 3: time.sleep(2 ** attempt)
            else: raise ConnectionError(f"Fallo al descargar {file_name}: {e}")

# =============================================================================
# LÓGICA MATEMÁTICA
# =============================================================================
def get_time_len_and_terms(stats_results):
    time_len = 0
    all_terms = {0: set(THEORETICAL_MAP[0].keys()), 1: set(THEORETICAL_MAP[1].keys())}
    for eq_str in ["0", "1"]:
        eq_int = int(eq_str)
        if eq_str in stats_results:
            all_terms[eq_int].update(stats_results[eq_str].keys())
            for term, data in stats_results[eq_str].items():
                time_len = len(data['mean'])
    return time_len, all_terms

def calculate_global_error(stats_results):
    time_len, all_terms = get_time_len_and_terms(stats_results)
    if time_len == 0: return []

    errors = []
    for t_idx in range(time_len):
        num_err, den = 0.0, 0.0
        
        for eq_int in [0, 1]:
            eq_str = str(eq_int)
            for term in all_terms[eq_int]:
                theo_val = THEORETICAL_MAP[eq_int].get(term, 0.0)
                
                emp_mean = 0.0
                if eq_str in stats_results and term in stats_results[eq_str]:
                    emp_mean = stats_results[eq_str][term]['mean'][t_idx]
                
                num_err += (emp_mean - theo_val)**2
                den += theo_val**2
                
        err_t = np.sqrt(num_err) / np.sqrt(den) if den > 0 else np.sqrt(num_err)
        errors.append(err_t)
        
    return errors

# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================
def main():
    print("=== INICIANDO COMPARACIÓN DE CONFIGURACIONES (TRADEOFF) ===")
    service = get_drive_service()
    base_folder_id = get_target_folder_id()
    
    system_folder_name = System.name.lower().replace("-", "_").replace(" ", "_")
    system_folder_id = get_folder_id_by_name(service, base_folder_id, system_folder_name)
    
    # Estructura: plot_data[zone_id][config_label][unbias_flag] = (times, errors, color)
    plot_data = {z: {} for z in TARGET_ZONES}

    # 1. Recolección de Datos
    for batch, traj_name, label, color in CONFIGURATIONS:
        print(f"\n[*] Buscando datos para: {label} (Batch: {batch}, Carpeta: {traj_name})")
        
        batch_folder_id = get_folder_id_by_name(service, system_folder_id, f"BATCH_{batch}_statistical_sweep")
        if not batch_folder_id:
            print(f"  [!] No se encontró BATCH_{batch}_statistical_sweep. Saltando.")
            continue
            
        traj_folder_id = get_folder_id_by_name(service, batch_folder_id, traj_name)
        if not traj_folder_id:
            print(f"  [!] No se encontró la carpeta '{traj_name}'. Saltando.")
            continue
            
        for zone_id in TARGET_ZONES:
            zone_folder_id = get_folder_id_by_name(service, traj_folder_id, f"zone_{zone_id}")
            if not zone_folder_id: continue
                
            files_in_zone = get_files_in_folder(service, zone_folder_id)
            plot_data[zone_id][label] = {}
            
            for unbias_flag in [True, False]:
                mode_str = "TRUE" if unbias_flag else "FALSE"
                json_filename = f"stats_unbias_{mode_str}.json"
                
                if json_filename in files_in_zone:
                    unique_cache_name = f"{label.replace(' ', '_')}_zone_{zone_id}_{json_filename}"
                    local_json = download_file_to_cache(service, files_in_zone[json_filename], unique_cache_name)
                    
                    with open(local_json, 'r') as f:
                        data = json.load(f)
                    
                    time_steps = np.array(data['config']['times'])
                    stats_results = data['results']
                    errors = np.array(calculate_global_error(stats_results))
                    
                    if MAX_TIME_EVAL is not None:
                        valid_idx = time_steps <= MAX_TIME_EVAL
                        time_steps = time_steps[valid_idx]
                        errors = errors[valid_idx]
                        
                    plot_data[zone_id][label][unbias_flag] = (time_steps, errors, color)
                    print(f"  > Descargado y procesado: Zona {zone_id} | Unbias: {unbias_flag}")

    # 2. Generación de Gráficos Locales
    print("\n=== GENERANDO GRÁFICOS ===")
    time_str = f" (t ≤ {MAX_TIME_EVAL}s)" if MAX_TIME_EVAL else ""
    
    for zone_id, configs_data in plot_data.items():
        if not configs_data: continue
        
        plt.figure(figsize=(12, 7))
        
        for label, flags_data in configs_data.items():
            for unbias_flag, (times, errors, color) in flags_data.items():
                line_label = f"{label} | {'Unbias' if unbias_flag else 'Biased'}"
                linestyle = LINESTYLES[unbias_flag]
                
                plt.plot(times, errors, color=color, linestyle=linestyle, linewidth=2, label=line_label, alpha=0.85)
                
        plt.title(f"Comparación de Error Relativo Global - ZONA {zone_id}{time_str}", fontsize=15)
        plt.xlabel("Tiempo de Evaluación (s)", fontsize=12)
        plt.ylabel("Error Relativo Global ($L_2$)", fontsize=12)
        
        plt.yscale('log')
        # Puedes ajustar los límites si lo necesitas:
        plt.ylim((5e-4, 5e2)) 
        
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        
        # Guardado en la carpeta base local
        filename = f"comparacion_tradeoff_zona_{zone_id}.png"
        save_path = os.path.join(CURRENT_DIR, filename)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  [✓] Gráfico guardado: {save_path}")

    print("\nLimpiando caché temporal...")
    if os.path.exists(TMP_DIR): shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("=== PROCESO COMPLETADO ===")

if __name__ == "__main__":
    main()