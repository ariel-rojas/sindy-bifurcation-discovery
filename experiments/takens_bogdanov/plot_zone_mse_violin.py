#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script CLOUD-NATIVE para Visualización de Errores de Ensamble (SINDy).
=====================================================================
Descarga los resultados de las ejecuciones más recientes para las 11
configuraciones y genera un Violin Plot personalizado en Matplotlib puro.
"""

import os
import sys
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from googleapiclient.http import MediaIoBaseDownload

# =============================================================================
# CONFIGURACIÓN DE RUTAS E IMPORTACIONES
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System
from core.drive_auth import get_drive_service, get_target_folder_id
from experiments.takens_bogdanov.data_zone_manager import _build_drive_service

# Métrica a visualizar: 'mae_global' o 'mse_global'
METRIC_TO_PLOT = 'mae_global' 

# Paleta de colores estricta por zona
ZONES_COLORS = {
    "Todas": "#95a5a6",  # Gris (Para el modo all_zones)
    "1": "#3498db",      # Azul
    "2": "#9b59b6",      # Violeta
    "3": "#f1c40f",      # Amarillo
    "4": "#2ecc71",      # Verde
    "5": "#e74c3c"       # Rojo
}

# Configuración del Eje Y
Y_CATEGORIES = {
    "one_zone_fixed_params": ("Parámetros Fijos", 1),
    "one_zone_var_params": ("Parámetros Variables", 2),
    "all_zones": ("Mezcla (Todas las zonas)", 3)
}

# =============================================================================
# CLASE RECOLECTORA DE DATOS
# =============================================================================
class EnsembleDataCollector:
    def __init__(self):
        self.system_folder = System.name.lower().replace("-", "_").replace(" ", "_")
        
        raw_service = get_drive_service()
        self.service = _build_drive_service(raw_service)
        
        base_id = get_target_folder_id()
        self.sys_id = self._get_folder_id(base_id, self.system_folder)
        self.models_id = self._get_folder_id(self.sys_id, "sindy_models")

    def _get_folder_id(self, parent_id, name):
        query = f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        res = self.service.files().list(q=query, fields='files(id)').execute().get('files', [])
        if not res:
            raise FileNotFoundError(f"Carpeta '{name}' no encontrada en Drive.")
        return res[0]['id']

    def _list_subfolders(self, parent_id):
        query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        return self.service.files().list(q=query, fields='files(id, name)').execute().get('files', [])

    def _download_json(self, file_id):
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return json.load(fh)

    def fetch_latest_results(self):
        print("🔍 Explorando carpetas de modelos en Drive...")
        latest_runs = {} 
        
        batches = self._list_subfolders(self.models_id)
        for batch in batches:
            print(f" -> Revisando: {batch['name']}")
            runs = self._list_subfolders(batch['id'])
            
            for run in runs:
                query = f"name='ensemble_metadata.json' and '{run['id']}' in parents and trashed=false"
                files = self.service.files().list(q=query, fields='files(id)').execute().get('files', [])
                
                if files:
                    try:
                        data = self._download_json(files[0]['id'])
                        mode = data['config']['sampling_mode']
                        zone = str(data['config'].get('target_zone', 'Todas')) if mode != 'all_zones' else 'Todas'
                        timestamp = data['timestamp']
                        
                        config_key = (mode, zone)
                        
                        if config_key not in latest_runs or timestamp > latest_runs[config_key]['timestamp']:
                            latest_runs[config_key] = data
                    except Exception as e:
                        print(f"⚠️ Error procesando JSON en {run['name']}: {e}")

        return latest_runs

# =============================================================================
# GENERACIÓN DEL GRÁFICO (MATPLOTLIB PURO)
# =============================================================================
def build_dataframe(latest_runs):
    records = []
    for (mode, zone), data in latest_runs.items():
        for log in data.get('iteration_logs', []):
            records.append({
                'MAE Global': log['mae_global'],
                'MSE Global': log['mse_global'],
                'Modo': mode,
                'Zona': zone
            })
    return pd.DataFrame(records)

def plot_custom_violin(df, metric):
    if df.empty:
        print("❌ No hay datos para graficar.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Mapeo de nombres para el eje X y la columna del DataFrame
    if metric == "mae_global":
        col_name = "MAE Global"
        metric_label = "MAE Global (Error Absoluto Medio)"
    else:
        col_name = "MSE Global"
        metric_label = "MSE Global (Error Cuadrático Medio)"
    
    y_ticks = []
    y_labels = []

    # Iterar sobre las 3 categorías principales del Eje Y
    for mode_key, (y_label, y_center) in Y_CATEGORIES.items():
        mode_df = df[df['Modo'] == mode_key]
        if mode_df.empty: 
            continue
            
        y_ticks.append(y_center)
        y_labels.append(y_label)

        # Determinar cuántas zonas existen en este modo para calcular los offsets espaciales
        zones = sorted(mode_df['Zona'].unique())
        n_zones = len(zones)
        
        # Ancho total disponible para los violines en este nivel Y
        total_width = 0.8 
        
        if n_zones == 1:
            offsets = [0]
            v_width = 0.5
        else:
            offsets = np.linspace(-total_width/2, total_width/2, n_zones)
            v_width = (total_width / n_zones) * 0.9

        # Dibujar cada violín
        for i, zone in enumerate(zones):
            # AQUÍ ESTABA EL ERROR: Cambiamos [metric] por [col_name]
            zone_data = mode_df[mode_df['Zona'] == zone][col_name].dropna().values
            if len(zone_data) == 0: continue
            
            pos = y_center + offsets[i]
            
            # Matplotlib puro: máxima personalización
            parts = ax.violinplot(
                zone_data, 
                positions=[pos], 
                vert=False, 
                widths=v_width,
                showmeans=False, 
                showextrema=True, 
                showmedians=True,
                quantiles=[0.25, 0.75] # Para mostrar los rangos intercuartílicos
            )

            color = ZONES_COLORS.get(zone, "#333333")

            # 1. Estilizar el "cuerpo" del violín
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
                pc.set_linewidth(1)

            # 2. Estilizar la MEDIANA (Línea sólida gruesa)
            if 'cmedians' in parts:
                parts['cmedians'].set_color('black')
                parts['cmedians'].set_linewidth(3.0)

            # 3. Estilizar los CUARTILES 25% y 75% (Líneas punteadas)
            if 'cquantiles' in parts:
                parts['cquantiles'].set_color('black')
                parts['cquantiles'].set_linewidth(1.8)
                parts['cquantiles'].set_linestyle('--')

            # 4. Estilizar los "bigotes" (Mínimos y Máximos)
            for part_name in ['cbars', 'cmins', 'cmaxes']:
                if part_name in parts:
                    parts[part_name].set_color('black')
                    parts[part_name].set_linewidth(1.2)

    # --- Estética Global del Gráfico ---
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=13, fontweight='bold')
    ax.set_xlabel(metric_label, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Distribución del Error ({col_name.upper()}) por Zona Dinámica y Muestreo\nSistema: {System.name}", 
                 fontsize=15, fontweight='bold', pad=15)
    
    # Cuadrícula para facilitar lectura
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    # --- Leyenda Personalizada (El "cuadrito") ---
    present_zones = sorted(df['Zona'].unique())
    legend_patches = []
    for z in present_zones:
        label = f"Zona {z}" if z != "Todas" else "Todas (Mezcla)"
        patch = mpatches.Patch(color=ZONES_COLORS.get(z, "#000"), label=label, alpha=0.7)
        legend_patches.append(patch)
        
    ax.legend(handles=legend_patches, title="Zonas", title_fontsize='11', 
              fontsize='10', loc='upper right', frameon=True, edgecolor='black')

    plt.tight_layout()

    # Guardar y mostrar
    output_path = os.path.join(CURRENT_DIR, f"violin_plot_{metric}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Gráfico guardado exitosamente en: {output_path}")
    
    plt.show()

# =============================================================================
# ORQUESTADOR
# =============================================================================
def main():
    print("--- Generador de Violin Plots de Ensamble SINDy ---")
    
    try:
        collector = EnsembleDataCollector()
        latest_runs = collector.fetch_latest_results()
        
        if not latest_runs:
            print("❌ No se encontraron datos válidos en Drive.")
            return

        print(f"✅ Se obtuvieron datos de {len(latest_runs)} configuraciones distintas.")
        
        df = build_dataframe(latest_runs)
        
        # Puedes cambiar METRIC_TO_PLOT en las constantes globales para MSE
        plot_custom_violin(df, metric=METRIC_TO_PLOT)
        
    except Exception as e:
        import traceback
        print(f"\n❌ Error durante la ejecución: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()