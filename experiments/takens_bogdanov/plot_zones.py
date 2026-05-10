#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualizador de Distribución de Zonas CLOUD-NATIVE.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.takens_bogdanov.data_zone_manager import DataZoneManager
from systems.takens_bogdanov import TakensBogdanov as System
from core.io import parse_param_key
from core.drive_auth import get_drive_service

def main():
    print("--- Iniciando Visualizador de Zonas Cloud ---")
    
    try:
        service = get_drive_service()
        # El manager se encarga de descargar la metadata y clasificar
        manager = DataZoneManager(system_class=System, service=service)
    except Exception as e:
        print(f"❌ Error de conexión o inicialización: {e}")
        return

    # Configuración de Estética (Zonas 4 y 5 invertidas según tu pedido)
    zones_config = {
        1: {"label": "Zona 1 (Nodo/Foco Est)", "color": "#1f77b4"}, 
        2: {"label": "Zona 2 (Silla-Nodo)",    "color": "#ff7f0e"}, 
        3: {"label": "Zona 3 (3 Puntos Fijos)", "color": "#2ca02c"}, 
        4: {"label": "Zona 4 (Ciclo Límite)",   "color": "#9467bd"}, # Violeta
        5: {"label": "Zona 5 (Escape)",         "color": "#d62728"}, # Rojo
    }

    plt.figure(figsize=(12, 8))
    
    # 1. Dibujar Curvas de Referencia
    print("Recuperando curvas de referencia...")
    # Usamos la homoclina detectada que el manager ya descargó en self.meta
    hc_data = manager.meta.get("_detected_homoclinic_curve")
    
    curves = System().get_bifurcation_curves(detected_homoclinic=hc_data)
    for name, (cx, cy, color, style) in curves.items():
        plt.plot(cx, cy, color='black', linestyle=style, linewidth=2, alpha=0.8, label=f"Teoría: {name}")

    # 2. Graficar Puntos por Zona
    print("Graficando puntos por zona...")
    total_points = 0
    
    # IMPORTANTE: Usamos directamente manager.zone_map para evitar el AttributeError
    # si el método get_samples_from_zone no está presente por algún motivo.
    for zone_id, props in zones_config.items():
        keys = manager.zone_map.get(zone_id, [])
        
        if not keys:
            continue
            
        coords = np.array([parse_param_key(k) for k in keys])
        
        plt.scatter(
            coords[:, 0], coords[:, 1], 
            c=props["color"], 
            label=f"{props['label']} (n={len(keys)})",
            s=12, 
            alpha=0.5, 
            edgecolors='none'
        )
        total_points += len(keys)

    # 3. Estética Final
    plt.title(f"Distribución Cloud: {System.name} (Total: {total_points})", fontsize=14)
    plt.xlabel(fr"Parámetro ${System.param_names[0]}$")
    plt.ylabel(fr"Parámetro ${System.param_names[1]}$")
    
    p_ranges = System.param_ranges
    plt.xlim(p_ranges[0][0] - 0.01, p_ranges[0][1] + 0.01)
    plt.ylim(p_ranges[1][0] - 0.01, p_ranges[1][1] + 0.01)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    
    plt.tight_layout()
    
    output_name = f"zones_map_cloud_{manager.system_folder_name}.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"✅ Mapa generado con éxito: {output_name}")
    plt.show()

if __name__ == "__main__":
    main()