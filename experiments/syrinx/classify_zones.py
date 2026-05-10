#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clasificador Geométrico de Regiones para SyrinxModel (Versión Dimensional).
Filtra los datos sucios de Drive y clasifica el espacio de parámetros físicos
evaluando la posición respecto a las curvas teóricas (Saddle-Node y Hopf).
"""

import os
import sys
import tempfile
import json
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

# =============================================================================
# CONFIGURACIÓN DE RUTAS E IMPORTACIONES
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from experiments.syrinx.config import EXPERIMENT
from systems.syrinx import SyrinxModel as System
from core.io import parse_param_key
from core.drive_auth import get_drive_service, get_target_folder_id

# Compatibilidad exacta con "The Bastion" para encontrar la carpeta en Drive
SYSTEM_FOLDER_NAME = System.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
CACHE_DIR = os.path.join(tempfile.gettempdir(), f"syrinx_classifier_cache_{SYSTEM_FOLDER_NAME}")
os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# DEFINICIÓN DE ZONAS (GEOMÉTRICAS)
# =============================================================================
ZONE_COLORS = {
    "Zona 1 (Izquierda de Hopf)": "#2ecc71",       # Verde
    "Zona 2 (Entre Hopf y SN)": "#e74c3c",         # Rojo
    "Zona 3 (Derecha de SN)": "#9b59b6",           # Morado
    "Zona 0 (Sin fronteras teóricas)": "#95a5a6",  # Gris
}

def get_metadata_from_drive(service):
    """Descarga la metadata consolidada de The Bastion desde Google Drive."""
    base_folder_id = get_target_folder_id()
    
    # 1. Buscar carpeta del sistema
    res = service.files().list(q=f"'{base_folder_id}' in parents and name='{SYSTEM_FOLDER_NAME}' and trashed=false", fields="files(id)").execute()
    if not res.get("files"): raise FileNotFoundError(f"Carpeta del sistema '{SYSTEM_FOLDER_NAME}' no encontrada.")
    sys_id = res.get("files")[0]["id"]
    
    # 2. Buscar carpeta de trayectorias
    res = service.files().list(q=f"'{sys_id}' in parents and name='trajectories' and trashed=false", fields="files(id)").execute()
    if not res.get("files"): raise FileNotFoundError("Carpeta 'trajectories' no encontrada.")
    traj_id = res.get("files")[0]["id"]
    
    # 3. Buscar archivo JSON de metadata
    res = service.files().list(q=f"'{traj_id}' in parents and trashed=false", fields="files(id, name)").execute()
    files = res.get("files", [])
    
    meta_file = next((f for f in files if f["name"].startswith("grid_metadata_") and f["name"].endswith(".json")), None)
    if not meta_file: raise FileNotFoundError("Metadata JSON no encontrada en Drive.")
    
    local_path = os.path.join(CACHE_DIR, meta_file["name"])
    request = service.files().get_media(fileId=meta_file["id"])
    with io.FileIO(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: _, done = downloader.next_chunk()
        
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_boundary_interpolator(curve_data, x_key, y_key):
    """Crea una función que, dado el eje Y (kappa1), devuelve la frontera en el eje X (psub)."""
    x_vals, y_vals = curve_data[x_key], curve_data[y_key]
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_v, y_v = x_vals[valid], y_vals[valid]
    
    if len(y_v) == 0: return lambda y: np.nan
    
    # Ordenar estrictamente ascendente por el eje Y para np.interp
    sort_idx = np.argsort(y_v)
    y_sorted, x_sorted = y_v[sort_idx], x_v[sort_idx]
    
    # Remover duplicados en Y
    y_uniq, unq_idx = np.unique(y_sorted, return_index=True)
    x_uniq = x_sorted[unq_idx]
    
    return lambda y_input: float(np.interp(y_input, y_uniq, x_uniq, left=np.nan, right=np.nan))

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Conectando a Google Drive...")
    try:
        service = get_drive_service()
        metadata = get_metadata_from_drive(service)
    except Exception as e:
        print(f"Error de conexión o lectura: {e}")
        return
        
    sys_info = metadata.get("_system_info", {})
    phys_base = sys_info.get("physical_base_params", {})
    
    if not phys_base:
        print("Advertencia: No se encontraron parámetros base en la metadata. Se usarán los de por defecto.")
        phys_base = System.default_params

    # 1. Calculamos las Curvas Teóricas Físicas
    print("\nCalculando Curvas Teóricas para construir Fronteras Geométricas...")
    CURVE_POINTS = 50000  # Resolución de la curva teórica
    X_MIN_SEARCH, X_MAX_SEARCH = 0.001, 0.5  # Rango de x_eq para buscar las raíces
    
    curves = System.get_physical_bifurcation_curves(
        physical_params=phys_base, x_min=X_MIN_SEARCH, x_max=X_MAX_SEARCH, n_points=CURVE_POINTS
    )
    
    # Definición explícita de ejes
    x_param, y_param = "psub", "kappa1"

    # Construir funciones interpoladoras
    hopf_data = curves.get("Hopf")
    sn_data = curves.get("Saddle-Node")
    
    hopf_func = build_boundary_interpolator(hopf_data, x_param, y_param) if hopf_data else lambda y: np.nan
    sn_func = build_boundary_interpolator(sn_data, x_param, y_param) if sn_data else lambda y: np.nan

    # 2. Filtrado de Datos
    # Según la convención, param_arr es [kappa1, psub], por lo que p0 es kappa1 y p1 es psub
    p0_min, p0_max = EXPERIMENT.sweep.physical_ranges[0] # Rango kappa1
    p1_min, p1_max = EXPERIMENT.sweep.physical_ranges[1] # Rango psub

    param_keys = [k for k in metadata.keys() if not k.startswith("_")]
    classified_points = []
    
    print(f"Analizando y Filtrando {len(param_keys)} puntos totales de la grilla...")
    
    for key in tqdm(param_keys, desc="Clasificando (Geometría)", unit="pt"):
        vals = parse_param_key(key)
        if len(vals) < 2: continue
        
        # vals[0] = kappa1 (eje Y), vals[1] = psub (eje X)
        p0_val, p1_val = float(vals[0]), float(vals[1]) 
        
        # Filtro estricto: Ignorar simulaciones fósiles fuera del rango actual
        if not (p0_min <= p0_val <= p0_max and p1_min <= p1_val <= p1_max):
            continue
            
        # CLASIFICACIÓN PURAMENTE GEOMÉTRICA (evaluando psub en función de kappa1)
        p_hopf = hopf_func(p0_val)
        p_sn = sn_func(p0_val)
        
        if np.isnan(p_hopf) and np.isnan(p_sn):
            zona = "Zona 0 (Sin fronteras teóricas)"
        elif np.isnan(p_hopf):
            zona = "Zona 1 (Izquierda de Hopf)" if p1_val <= p_sn else "Zona 3 (Derecha de SN)"
        elif np.isnan(p_sn):
            zona = "Zona 1 (Izquierda de Hopf)" if p1_val <= p_hopf else "Zona 3 (Derecha de SN)"
        else:
            left_b = min(p_hopf, p_sn)
            right_b = max(p_hopf, p_sn)
            if p1_val <= left_b: zona = "Zona 1 (Izquierda de Hopf)"
            elif p1_val <= right_b: zona = "Zona 2 (Entre Hopf y SN)"
            else: zona = "Zona 3 (Derecha de SN)"
            
        classified_points.append((p0_val, p1_val, zona))

    # =========================================================================
    # VISUALIZACIÓN VECTORIZADA Y CONTEO
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    
    print("\nPreparando gráfico...")
    puntos_por_zona = {zona: {"x": [], "y": []} for zona in ZONE_COLORS.keys()}
    for p0, p1, zona in classified_points:
        puntos_por_zona[zona]["x"].append(p1) # X es psub
        puntos_por_zona[zona]["y"].append(p0) # Y es kappa1
        
    # --- REPORTE DE CONTEO EN CONSOLA ---
    print("\n" + "="*40)
    print("📊 REPORTE DE PUNTOS POR ZONA")
    print("="*40)
    for zona, coords in puntos_por_zona.items():
        conteo = len(coords["x"])
        print(f"[{conteo:^6}] -> {zona}")
    print("="*40 + "\n")
        
    for zona, coords in puntos_por_zona.items():
        if coords["x"]: 
            ax.scatter(coords["x"], coords["y"], color=ZONE_COLORS[zona], s=80, marker="s", edgecolors="none", alpha=0.8)
            
    # Dibujar Curvas Teóricas Físicas
    for name, data in curves.items():
        ax.plot(data[x_param], data[y_param], color="black" if "Saddle" in name else "blue", 
                linestyle=data["linestyle"], lw=2.5, label=f"Teórica: {name}", zorder=5)
    
    # Dibujar punto(s) Takens-Bogdanov
    print("Buscando codimensión-2 (Takens-Bogdanov)...")
    tb_points = System.find_takens_bogdanov_points(physical_params=phys_base, x_range=(X_MIN_SEARCH, X_MAX_SEARCH))
    
    for i, point in enumerate(tb_points):
        x_tb = point["P"]   # psub
        y_tb = point["k1"]  # kappa1
        ax.scatter(x_tb, y_tb, s=150, color="black", marker="*", zorder=10, label="Takens-Bogdanov" if i==0 else None)

    # Etiquetas Dimensionales
    lbl_1 = r"Rigidez Lineal de los Labios, $\kappa_1$ (din/cm)"
    lbl_2 = r"Presión Subglótica, $p_{sub}$ (din/cm$^2$)"
    
    ax.set_xlabel(lbl_2)
    ax.set_ylabel(lbl_1)
    ax.set_title("Zonas Dinámicas (Clasificación Geométrica Dimensional)")
    ax.set_xlim(p1_min, p1_max)
    ax.set_ylim(p0_min, p0_max)
    
    # --- ACTUALIZACIÓN DE LA LEYENDA CON CONTEO ---
    legend_patches = [
        mpatches.Patch(color=color, label=f"{label} ({len(puntos_por_zona[label]['x'])} pts)") 
        for label, color in ZONE_COLORS.items()
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_patches + handles, loc="best", fontsize="small")
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()