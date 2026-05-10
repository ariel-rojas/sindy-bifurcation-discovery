#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de Análisis Estadístico para SINDy (rs_batch).
----------------------------------------------------
Conecta a Google Drive, escanea la carpeta 'rs_batch' y descarga todos los 
modelos SINDy estables generados por el Random Search.
Calcula el promedio y la desviación estándar de cada término de la ecuación
para evaluar la robustez del descubrimiento frente a variaciones de hiperparámetros.
"""

import os
import sys
import io
import json
import tempfile
import shutil
import numpy as np
import joblib
import matplotlib.pyplot as plt
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
import pysindy as ps

# =============================================================================
# CONFIGURACIÓN DE RUTAS E IMPORTACIONES
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from systems.syrinx import SyrinxModel as System
from core.drive_auth import get_drive_service, get_target_folder_id

# Fix retrocompatibilidad de Joblib
if 'pysindy.pysindy' not in sys.modules: sys.modules['pysindy.pysindy'] = ps
if 'pysindy.utils.axes' not in sys.modules: sys.modules['pysindy.utils.axes'] = ps.utils

# Nombres en Drive
SYSTEM_FOLDER_NAME = System.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
MODELS_FOLDER_NAME = "sindy_models"
BATCH_FOLDER_NAME = "rs_batch"

TMP_DIR = os.path.join(tempfile.gettempdir(), f"sindy_stats_cache_{SYSTEM_FOLDER_NAME}")

# =============================================================================
# UTILIDADES DE GOOGLE DRIVE
# =============================================================================
def get_drive_index_full(service, folder_id, mime_type=None):
    """Obtiene todos los archivos/carpetas de un directorio manejando paginación."""
    idx = {}
    page_token = None
    query = f"'{folder_id}' in parents and trashed=false"
    if mime_type:
        query += f" and mimeType='{mime_type}'"
        
    while True:
        results = service.files().list(
            q=query, spaces='drive', fields='nextPageToken, files(id, name)',
            pageToken=page_token, pageSize=1000
        ).execute()
        for item in results.get('files', []):
            idx[item['name']] = item['id']
        page_token = results.get('nextPageToken', None)
        if not page_token: break
    return idx

def download_file_to_cache(service, file_id, file_name):
    """Descarga al caché temporal usando el nombre de archivo local seguro."""
    local_path = os.path.join(TMP_DIR, file_name)
    if os.path.exists(local_path): return local_path 
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(local_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: _, done = downloader.next_chunk()
    return local_path

# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================
def main():
    print(f"--- Análisis Estadístico de Modelos SINDy ({System.name}) ---")
    if os.path.exists(TMP_DIR): shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    
    # 1. Conexión a Drive con parche AuthorizedSession
    try:
        raw_service = get_drive_service()
        if not hasattr(raw_service, 'files'):
            creds = getattr(raw_service, 'credentials', raw_service)
            service = build('drive', 'v3', credentials=creds)
        else:
            service = raw_service

        base_id = get_target_folder_id()
        
        # Navegación jerárquica
        sys_query = service.files().list(q=f"name='{SYSTEM_FOLDER_NAME}' and '{base_id}' in parents and trashed=false", fields='files(id)').execute()
        if not sys_query.get('files'): raise FileNotFoundError("Carpeta del sistema no encontrada.")
        sys_id = sys_query.get('files')[0]['id']
        
        models_query = service.files().list(q=f"name='{MODELS_FOLDER_NAME}' and '{sys_id}' in parents and trashed=false", fields='files(id)').execute()
        if not models_query.get('files'): raise FileNotFoundError("Carpeta de modelos no encontrada.")
        models_id = models_query.get('files')[0]['id']
        
        batch_query = service.files().list(q=f"name='{BATCH_FOLDER_NAME}' and '{models_id}' in parents and trashed=false", fields='files(id)').execute()
        if not batch_query.get('files'): raise FileNotFoundError(f"Carpeta '{BATCH_FOLDER_NAME}' no encontrada.")
        batch_id = batch_query.get('files')[0]['id']
        
    except Exception as e:
        print(f"❌ Error crítico accediendo a la Nube: {e}")
        return

    # 2. Obtener todas las versiones (carpetas v1, v2, v3...)
    print("Sincronizando versiones de modelos...")
    folder_mime = 'application/vnd.google-apps.folder'
    versions_idx = get_drive_index_full(service, batch_id, mime_type=folder_mime)
    
    versions = {k: v for k, v in versions_idx.items() if k.startswith('v')}
    print(f"Se encontraron {len(versions)} modelos guardados.")
    
    if not versions: return

    # 3. Descarga y agregación de coeficientes
    # Estructura: stats[target_name][feature_name] = [lista_de_valores]
    stats = {target: {} for target in System.state_names}
    processed_count = 0

    print("Descargando y analizando ecuaciones...")
    for ver_name, ver_id in versions.items():
        files_idx = get_drive_index_full(service, ver_id)
        
        if "sindy_model.joblib" not in files_idx:
            continue
            
        try:
            local_model_name = f"{ver_name}_sindy_model.joblib"
            model_path = download_file_to_cache(service, files_idx["sindy_model.joblib"], local_model_name)
            
            model = joblib.load(model_path)
            coefs = model.coefficients()
            features = model.get_feature_names()
            
            for i, target in enumerate(System.state_names):
                for j, feat in enumerate(features):
                    # Registramos el valor numérico en nuestro diccionario
                    val = coefs[i, j]
                    if feat not in stats[target]:
                        stats[target][feat] = []
                    stats[target][feat].append(val)
            
            processed_count += 1
            sys.stdout.write(f"\rProgreso: {processed_count}/{len(versions)} modelos analizados.")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"\nError procesando {ver_name}: {e}")

    print("\n")
    if processed_count == 0:
        print("No se pudo procesar ningún modelo válido.")
        return

    # 4. Cálculo de Estadísticas y Reporte
    print("="*75)
    print(f"📊 REPORTE ESTADÍSTICO DE COEFICIENTES (Basado en {processed_count} modelos estables)")
    print("="*75)

    # Solo mostrar los términos que en al menos un modelo no fueron cero
    THRESHOLD_DISPLAY = 1e-6 

    for target in System.state_names:
        print(f"\n▶ Dinámica de ({target})':")
        print(f"  {'TÉRMINO':<20} | {'MEDIA':>10} | {'DESV. ESTÁNDAR (σ)':>15} | {'ROBUSTEZ'}")
        print("-" * 75)
        
        # Ordenar features por el valor absoluto de su media para ver los dominantes primero
        sorted_feats = sorted(stats[target].keys(), key=lambda f: abs(np.mean(stats[target][f])), reverse=True)
        
        for feat in sorted_feats:
            vals = np.array(stats[target][feat])
            
            if np.all(np.abs(vals) < THRESHOLD_DISPLAY):
                continue # Omitir términos que todos los modelos mataron
                
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            
            # Clasificación cualitativa de robustez (Coeficiente de Variación)
            if abs(mean_val) < 1e-9:
                robustness = "N/A"
            else:
                cv = abs(std_val / mean_val)
                if cv < 0.1: robustness = "⭐⭐⭐ Muy Alta"
                elif cv < 0.5: robustness = "⭐⭐ Alta"
                elif cv < 1.0: robustness = "⭐ Media"
                else: robustness = "⚠️ Dudosa"

            print(f"  {feat:<20} | {mean_val:>10.4f} | {std_val:>15.4f} | {robustness}")

    print("\n" + "="*75)
    print("Nota de Robustez: σ bajas significan que SINDy descubre ese término de")
    print("forma consistente sin importar los recortes temporales o hiperparámetros.")

    if os.path.exists(TMP_DIR): shutil.rmtree(TMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()