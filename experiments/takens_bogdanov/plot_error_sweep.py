#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de POST-PROCESAMIENTO: Barrido de Error, Ancho y Ángulos Óptimos.

Mejoras incluidas:
- Filtro de Tiempo Máximo (MAX_TIME_EVAL) para limitar el gráfico.
- Cálculo del Ancho Global (Norma L2 de las desviaciones estándar).
- Cálculo del Ángulo crudo convertido a grados correctamente.
- Generación de 4 gráficos independientes por configuración.
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
from core.drive_auth import get_drive_service, get_target_folder_id

# =============================================================================
# PARÁMETROS DEL SCRIPT
# =============================================================================
TARGET_BATCH = 3
TARGET_ZONES = [1, 2, 3, 4]

# NUEVO PARÁMETRO: Tiempo máximo a evaluar. 
# Ponlo en None para evaluar toda la curva, o un valor float (ej. 15.0)
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
TMP_DIR = os.path.join(tempfile.gettempdir(), f"error_sweep_cache_{RUN_ID}")
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
            while not done:
                _, done = downloader.next_chunk()
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
            while response is None:
                _, response = request.next_chunk()
                
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

def calculate_global_metrics(stats_results):
    """Calcula el Error Global (mean) y el Ancho Global (std)"""
    time_len, all_terms = get_time_len_and_terms(stats_results)
    if time_len == 0: return [], []

    errors, widths = [], []
    for t_idx in range(time_len):
        num_err, num_width, den = 0.0, 0.0, 0.0
        
        for eq_int in [0, 1]:
            eq_str = str(eq_int)
            for term in all_terms[eq_int]:
                theo_val = THEORETICAL_MAP[eq_int].get(term, 0.0)
                
                emp_mean, emp_std = 0.0, 0.0
                if eq_str in stats_results and term in stats_results[eq_str]:
                    emp_mean = stats_results[eq_str][term]['mean'][t_idx]
                    emp_std = stats_results[eq_str][term]['std'][t_idx]
                
                num_err += (emp_mean - theo_val)**2
                num_width += emp_std**2
                den += theo_val**2
                
        err_t = np.sqrt(num_err) / np.sqrt(den) if den > 0 else np.sqrt(num_err)
        width_t = np.sqrt(num_width) / np.sqrt(den) if den > 0 else np.sqrt(num_width)
        
        errors.append(err_t)
        widths.append(width_t)
        
    return errors, widths

def calculate_curve_angles(time_steps, values):
    """
    Calcula el ángulo crudo de la curva sin normalizar: theta = arctan(dy/dt).
    """
    t = np.array(time_steps)
    v = np.array(values)
    
    angles = []
    # Derivada discreta hacia adelante
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        dv = v[i+1] - v[i]
        
        # arctan2(y, x) nos da el ángulo exacto en el plano real. 
        # Convertido a grados para coincidir con la etiqueta del eje Y.
        angle_rad = np.arctan2(dv, dt)
        angles.append(np.degrees(angle_rad)) 
        
    # Duplicar el último valor para mantener la longitud del vector temporal
    if len(angles) > 0:
        angles.append(angles[-1])
        
    return np.array(angles)
# =============================================================================
# UTILIDAD DE PLOTEO MODULAR
# =============================================================================
def generate_and_upload_plot(service, traj_name, traj_id, plot_data, data_key, title, ylabel, filename, use_log=False, y_lims=None):
    plt.figure(figsize=(10, 6))
    
    for (zone_id, unbias_flag), data in plot_data.items():
        label = f"Zona {zone_id} | {'Unbias' if unbias_flag else 'Biased'}"
        color = COLORS.get(zone_id, 'black')
        linestyle = LINESTYLES[unbias_flag]
        
        plt.plot(data['times'], data[data_key], color=color, linestyle=linestyle, linewidth=2, label=label, alpha=0.8)
        
    plt.title(f"{title}\n({traj_name.replace('_', ' ')} | Batch Size: {TARGET_BATCH})", fontsize=14)
    plt.xlabel("Tiempo de Evaluación (s)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    if use_log:
        plt.yscale('log')
    if y_lims:
        plt.ylim(y_lims)
        
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    local_path = os.path.join(TMP_DIR, filename)
    plt.savefig(local_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  > Subiendo {filename}...")
    upload_to_drive(service, local_path, filename, traj_id)


# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================
def main():
    print(f"=== INICIANDO ANÁLISIS DE ERROR Y ÁNGULOS (BATCH {TARGET_BATCH}) ===")
    if MAX_TIME_EVAL is not None:
        print(f"[*] Filtro de tiempo activado: Gráficos recortados hasta t = {MAX_TIME_EVAL}s")
        
    service = get_drive_service()
    base_folder_id = get_target_folder_id()
    
    system_folder_name = System.name.lower().replace("-", "_").replace(" ", "_")
    system_folder_id = get_folder_id_by_name(service, base_folder_id, system_folder_name)
    batch_folder_id = get_folder_id_by_name(service, system_folder_id, f"BATCH_{TARGET_BATCH}_statistical_sweep")
    
    trajs_folders = get_subfolders(service, batch_folder_id)

    for traj_folder in trajs_folders:
        traj_name = traj_folder['name'] 
        traj_id = traj_folder['id']
        print(f"\n{'='*60}\nProcesando: {traj_name}\n{'='*60}")

        zones_folders = get_subfolders(service, traj_id)
        plot_data = {}
        
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
                    unique_cache_name = f"{traj_name}_{zone_name}_{json_filename}"
                    local_json = download_file_to_cache(service, files_in_zone[json_filename], unique_cache_name)
                    
                    with open(local_json, 'r') as f:
                        data = json.load(f)
                    
                    time_steps = np.array(data['config']['times'])
                    stats_results = data['results']
                    
                    errors, widths = calculate_global_metrics(stats_results)
                    errors = np.array(errors)
                    widths = np.array(widths)
                    
                    # --- NUEVO: RECORTE TEMPORAL ---
                    if MAX_TIME_EVAL is not None:
                        valid_idx = time_steps <= MAX_TIME_EVAL
                        time_steps = time_steps[valid_idx]
                        errors = errors[valid_idx]
                        widths = widths[valid_idx]
                    
                    # Se calculan los ángulos sobre los arrays ya recortados
                    angle_errors = calculate_curve_angles(time_steps, errors)
                    angle_widths = calculate_curve_angles(time_steps, widths)
                    
                    plot_data[(zone_id, unbias_flag)] = {
                        'times': time_steps,
                        'errors': errors,
                        'widths': widths,
                        'angle_errors': angle_errors,
                        'angle_widths': angle_widths
                    }

        if plot_data:
            print(f"Generando 4 gráficos consolidados para {traj_name}...")
            time_str = f" (t ≤ {MAX_TIME_EVAL}s)" if MAX_TIME_EVAL else ""
            
            # 1. Gráfico de Error
            generate_and_upload_plot(
                service, traj_name, traj_id, plot_data, 
                data_key='errors', 
                title=f"Error Relativo Global vs Tiempo{time_str}", 
                ylabel="Error Relativo Global ($L_2$)", 
                filename=f"plot_1_global_error_{traj_name}.png", 
                use_log=True, y_lims=(5e-4, 5e2)
            )
            
            # # 2. Gráfico del Ángulo del Error
            # generate_and_upload_plot(
            #     service, traj_name, traj_id, plot_data, 
            #     data_key='angle_errors', 
            #     title=f"Ángulo Crudo del Error vs Tiempo{time_str}\n($\\theta = \\arctan(\\Delta E / \\Delta t)$)", 
            #     ylabel="Ángulo (grados)", 
            #     filename=f"plot_2_angle_error_{traj_name}.png", 
            #     use_log=False, y_lims=None 
            # )

            # # 3. Gráfico de Ancho
            # generate_and_upload_plot(
            #     service, traj_name, traj_id, plot_data, 
            #     data_key='widths', 
            #     title=f"Ancho Estadístico Global (STD) vs Tiempo{time_str}", 
            #     ylabel="Ancho Global ($L_2$ de las Desviaciones)", 
            #     filename=f"plot_3_global_width_{traj_name}.png", 
            #     use_log=True, y_lims=(5e-4, 5e2)
            # )

            # # 4. Gráfico del Ángulo del Ancho
            # generate_and_upload_plot(
            #     service, traj_name, traj_id, plot_data, 
            #     data_key='angle_widths', 
            #     title=f"Ángulo Crudo del Ancho vs Tiempo{time_str}\n($\\theta = \\arctan(\\Delta W / \\Delta t)$)", 
            #     ylabel="Ángulo (grados)", 
            #     filename=f"plot_4_angle_width_{traj_name}.png", 
            #     use_log=False, y_lims=None 
            # )

    print("\nLimpiando caché local...")
    if os.path.exists(TMP_DIR): shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("\n=== POST-PROCESAMIENTO FINALIZADO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()