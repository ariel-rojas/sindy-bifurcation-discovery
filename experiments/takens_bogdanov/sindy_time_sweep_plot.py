#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script CLOUD-NATIVE: Visualización de Barrido Temporal SINDy.
=============================================================
Recorre las carpetas 'batch_*', descarga el JSON de resultados más 
reciente de cada una, consolida los datos y genera el gráfico 
estilizado (Scatter + Bandas de Error) de la convergencia del modelo.
"""

import os
import sys
import io
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.http import MediaIoBaseDownload

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System
from core.drive_auth import get_drive_service, get_target_folder_id
from experiments.takens_bogdanov.data_zone_manager import _build_drive_service

# --- CONFIGURACIÓN DEL PLOTEO ---
TARGET_REGION = "base"

# "ALL" para graficar todas, o lista manual ej: ["ALL_ZONES (Mezcla)", "Z5 - Var Params"]
TARGET_CONFIGS_TO_PLOT = "ALL"
# TARGET_CONFIGS_TO_PLOT = ["ALL_ZONES (Mezcla)"]
# TARGET_CONFIGS_TO_PLOT = ["Z1 - Var Params", "Z2 - Var Params", "Z3 - Var Params", "Z4 - Var Params", "Z5 - Var Params"]
# TARGET_CONFIGS_TO_PLOT = ["Z1 - Fixed Params", "Z2 - Fixed Params", "Z3 - Fixed Params", "Z4 - Fixed Params", "Z5 - Fixed Params"]
# =============================================================================
# UTILIDADES DE DRIVE (CRAWLER)
# =============================================================================
def _get_folder_id(service, parent_id, name):
    query = f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    res = service.files().list(q=query, fields='files(id)').execute().get('files', [])
    if not res: return None
    return res[0]['id']

def _list_folders(service, parent_id):
    query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    return service.files().list(q=query, fields='files(id, name)').execute().get('files', [])

def download_json_in_memory(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done: _, done = downloader.next_chunk()
    fh.seek(0)
    return json.load(fh)

def collect_all_results(service):
    """Explora todas las carpetas batch_*, descarga el JSON más reciente y los une."""
    system_folder = System.name.lower().replace("-", "_").replace(" ", "_")
    base_id = get_target_folder_id()
    
    sys_id = _get_folder_id(service, base_id, system_folder)
    region_id = _get_folder_id(service, sys_id, TARGET_REGION)
    models_id = _get_folder_id(service, region_id, "sindy_models")
    
    if not models_id:
        raise FileNotFoundError("No se encontró la carpeta 'sindy_models' en la región especificada.")

    all_records = []
    print("🔍 Recolectando resultados desde Drive...")
    
    batches = _list_folders(service, models_id)
    for batch in batches:
        # Buscar todos los time_sweep_*.json en esta carpeta batch
        q_files = f"name contains 'time_sweep_' and name contains '.json' and '{batch['id']}' in parents and trashed=false"
        files = service.files().list(q=q_files, fields='files(id, name)').execute().get('files', [])
        
        if not files: continue
        
        # Tomar el más reciente por orden alfabético (ya que tienen timestamp)
        latest_file = sorted(files, key=lambda x: x['name'], reverse=True)[0]
        
        print(f" -> Procesando: {batch['name']} | Archivo: {latest_file['name']}")
        try:
            data = download_json_in_memory(service, latest_file['id'])
            config_label = data.get("config_label", "Unknown Config")
            
            for record in data.get("results", []):
                # Inyectar el label de la configuración al registro
                record["Config"] = config_label
                all_records.append(record)
        except Exception as e:
            print(f"    ⚠️ Error leyendo JSON: {e}")

    return pd.DataFrame(all_records)

# =============================================================================
# LÓGICA DE PLOTEO (ESTILO PAPER)
# =============================================================================
def plot_results(df):
    if df.empty: 
        print("❌ El DataFrame está vacío. No hay datos para graficar.")
        return
        
    if TARGET_CONFIGS_TO_PLOT != "ALL":
        df = df[df['Config'].isin(TARGET_CONFIGS_TO_PLOT)]
        if df.empty:
            print(f"❌ Ninguna de las configuraciones solicitadas existe en los datos.")
            return

    plt.figure(figsize=(11, 7))
    sns.set_theme(style="ticks") 
    
    # Paleta de colores distintiva
    colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12', '#8e44ad', '#d35400', '#c0392b', '#16a085', '#34495e', '#7f8c8d', '#bdc3c7']
    configs = df['Config'].unique()
    
    for idx, config_name in enumerate(configs):
        df_conf = df[df['Config'] == config_name].copy()
        
        # Filtrar posibles NaNs por iteraciones fallidas
        df_conf = df_conf.dropna(subset=['mse_global'])
        if df_conf.empty: continue
        
        c = colors[idx % len(colors)]
        
        # Escala logarítmica (agregando un epsilon diminuto para evitar log(0))
        log_mse = np.log10(df_conf['mse_global'])
        
        # 1. SCATTER (Transparente para ver densidad)
        plt.scatter(df_conf['t_max'], log_mse, 
                    color=c, alpha=0.15, s=15, marker='o', edgecolors='none')
        
        # 2. BANDA DE DESVIACIÓN Y LÍNEA MEDIA
        grouped = pd.DataFrame({'t_max': df_conf['t_max'], 'log_mse': log_mse}).groupby('t_max')['log_mse']
        t_vals = grouped.mean().index
        mean_vals = grouped.mean().values
        std_vals = grouped.std().values
        
        # Línea Sólida Media
        plt.plot(t_vals, mean_vals, color=c, lw=2.5, label=config_name)
        
        # Banda Sombreada Std Dev
        plt.fill_between(t_vals, mean_vals - std_vals, mean_vals + std_vals, 
                         color=c, alpha=0.15)
                         
        # Líneas de Contorno de la Banda
        plt.plot(t_vals, mean_vals + std_vals, color=c, linestyle='--', lw=1.0, alpha=0.7)
        plt.plot(t_vals, mean_vals - std_vals, color=c, linestyle='--', lw=1.0, alpha=0.7)
        
    plt.xlabel(r"Tiempo de simulación inyectado a SINDy ($t_{max}$) [s]", fontsize=13)
    plt.ylabel(r"$\log_{10}(\mathrm{MSE})$ Global de Inferencia", fontsize=13)
    
    title_suffix = "Todas las configs" if TARGET_CONFIGS_TO_PLOT == "ALL" else "Selección Específica"
    plt.title(f"Convergencia Temporal de SINDy (Región: {TARGET_REGION} | {title_suffix})", fontsize=15, fontweight='bold', pad=15)
    
    # Acomodar la leyenda si son muchas configuraciones
    if len(configs) > 5:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, edgecolor='black', fontsize=10)
    else:
        plt.legend(loc='best', frameon=True, edgecolor='black', fontsize=11)
        
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim(df['t_max'].min(), df['t_max'].max())
    
    plt.tight_layout()
    output_path = os.path.join(CURRENT_DIR, f"sindy_time_sweep_plot_{TARGET_REGION}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Gráfico generado y guardado en: {output_path}")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("--- Visualizador de Barrido Temporal SINDy ---")
    try:
        raw_service = get_drive_service()
        service = _build_drive_service(raw_service)
        
        df = collect_all_results(service)
        
        print("🎨 Generando Gráfico...")
        plot_results(df)
        
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")

if __name__ == "__main__":
    main()