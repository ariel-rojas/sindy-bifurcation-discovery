#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de Prueba Rápida (Sanity Check)
======================================
Lee los datos originales subidos a Google Drive y printea las propiedades 
temporales y las dimensiones de las trayectorias directamente desde la RAM.
"""

import os
import sys
import io
import numpy as np
from googleapiclient.http import MediaIoBaseDownload

# --- CONFIGURACIÓN DE RUTAS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Puedes cambiar el sistema aquí si estás usando el Cúbico
from systems.takens_bogdanov import TakensBogdanov as System
# from systems.cubic_symmetric_takens_bogdanov import CubicSymmetricTakensBogdanov as System

from core.drive_auth import get_drive_service, get_target_folder_id
from experiments.takens_bogdanov.data_zone_manager import _build_drive_service

def download_to_memory(service, file_id):
    """Descarga un archivo desde Drive directamente a la memoria RAM."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def main():
    print(f"\n--- Prueba de Lectura de Trayectorias Originales ({System.name}) ---")
    
    try:
        # Autenticación
        raw_service = get_drive_service()
        service = _build_drive_service(raw_service)
        base_id = get_target_folder_id()
        
        # Navegación en Drive
        system_folder = System.name.lower().replace("-", "_").replace(" ", "_")
        print(f"Buscando carpeta del sistema: '{system_folder}'...")
        
        sys_query = service.files().list(q=f"name='{system_folder}' and '{base_id}' in parents and trashed=false", fields='files(id)').execute()
        sys_id = sys_query.get('files')[0]['id']
        
        traj_query = service.files().list(q=f"name='trajectories' and '{sys_id}' in parents and trashed=false", fields='files(id)').execute()
        traj_id = traj_query.get('files')[0]['id']
        
        # Listar archivos
        print("Obteniendo índice de archivos...\n")
        files = service.files().list(q=f"'{traj_id}' in parents and name contains '.npz' and trashed=false", fields='files(id, name)', pageSize=10).execute().get('files', [])
        
        if not files:
            print("❌ No se encontraron archivos .npz en la carpeta trajectories.")
            return

        # Buscar t_eval.npz y un archivo de trayectoria aleatorio
        t_eval_file = next((f for f in files if f['name'] == 't_eval.npz'), None)
        data_file = next((f for f in files if f['name'] != 't_eval.npz'), None)
        
        # 1. Analizar t_eval
        if t_eval_file:
            print(f"⏳ Descargando y analizando {t_eval_file['name']}...")
            t_bytes = download_to_memory(service, t_eval_file['id'])
            with np.load(t_bytes) as data:
                t_eval = data['t_eval']
                dt = t_eval[1] - t_eval[0]
                t_final = t_eval[-1]
                print(f"  ✅ dt (Paso temporal): {dt:.4f} s")
                print(f"  ✅ T final: {t_final:.4f} s")
                print(f"  ✅ Pasos de integración totales: {len(t_eval)}\n")
        else:
            print("⚠️ No se encontró 't_eval.npz'.")

        # 2. Analizar un archivo de datos
        if data_file:
            print(f"📈 Descargando y analizando datos ({data_file['name']})...")
            d_bytes = download_to_memory(service, data_file['id'])
            with np.load(d_bytes) as data:
                trajs = data['trajectories']
                n_sims = trajs.shape[0]
                n_dims = trajs.shape[1]
                n_time = trajs.shape[2]
                
                print(f"  ✅ Condiciones iniciales (N_sims): {n_sims}")
                print(f"  ✅ Variables de estado (N_dims): {n_dims}")
                print(f"  ✅ Longitud temporal real guardada: {n_time}")
                print(f"  ✅ Shape completo del tensor: {trajs.shape}")
                
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")

if __name__ == '__main__':
    main()