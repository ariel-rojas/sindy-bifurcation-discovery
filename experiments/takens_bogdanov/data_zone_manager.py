#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import tempfile
import json
import io
import numpy as np
import matplotlib.pyplot as plt
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.discovery import build

# =============================================================================
# CONFIGURACIÓN DE RUTAS e IO
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.io import parse_param_key
from core.drive_auth import get_drive_service, get_target_folder_id
from systems.takens_bogdanov import TakensBogdanov as DefaultSystem

def _build_drive_service(raw):
    """Convierte AuthorizedSession en un Drive Service clásico."""
    if hasattr(raw, "files"):
        return raw  
    creds = getattr(raw, "credentials", raw)
    return build("drive", "v3", credentials=creds)

# =============================================================================
# CLASE PRINCIPAL DE CLASIFICACIÓN
# =============================================================================
class DataZoneManager:
    def __init__(self, system_class=DefaultSystem, service=None, target_folder_id=None, region="base"):
        self.system = system_class
        self.region = region
        self.system.set_region(self.region) # Configurar la región activa
        
        self.system_folder_name = self.system.name.lower().replace("-", "_").replace(" ", "_")
        self.target_folder_name = "trajectories"
        
        # Soporte para las 5 zonas del sistema
        self.zone_map = {z_id: [] for z_id in self.system.zone_names.keys()}
        self.unclassified = []
        
        # Conexión compatible
        raw_service = service or get_drive_service()
        self.service = _build_drive_service(raw_service)
        self.target_folder_id = target_folder_id or self._get_drive_folder_id()
        
        # Carga de Metadata específica para la región
        self.meta = self._load_metadata_from_drive()
        
        self._scan_and_classify_data()

    def _get_drive_folder_id(self):
        """Busca la jerarquía Sistema -> Región -> trajectories en Drive."""
        base_folder_id = get_target_folder_id()
        
        # 1. Sistema
        sys_results = self.service.files().list(
            q=f"name='{self.system_folder_name}' and '{base_folder_id}' in parents and trashed=false",
            fields='files(id)'
        ).execute()
        if not sys_results.get('files'):
            raise FileNotFoundError(f"No existe la carpeta del sistema '{self.system_folder_name}'")
        system_folder_id = sys_results.get('files')[0]['id']
        
        # 2. Región (Novedad Fase 2)
        reg_results = self.service.files().list(
            q=f"name='{self.region}' and '{system_folder_id}' in parents and trashed=false",
            fields='files(id)'
        ).execute()
        if not reg_results.get('files'):
            raise FileNotFoundError(f"No existe la subcarpeta de la región '{self.region}'")
        region_folder_id = reg_results.get('files')[0]['id']
        
        # 3. Trayectorias
        traj_results = self.service.files().list(
            q=f"name='{self.target_folder_name}' and '{region_folder_id}' in parents and trashed=false",
            fields='files(id)'
        ).execute()
        if not traj_results.get('files'):
            raise FileNotFoundError(f"La subcarpeta '{self.target_folder_name}' no existe en Drive.")
            
        return traj_results.get('files')[0]['id']

    def _load_metadata_from_drive(self):
        """Descarga el JSON de metadata desde la carpeta de trayectorias en Drive."""
        meta_name = f"grid_metadata_{self.system_folder_name}_{self.region}.json"
        print(f"Buscando metadata: {meta_name}...")
        
        results = self.service.files().list(
            q=f"name='{meta_name}' and '{self.target_folder_id}' in parents and trashed=false",
            fields='files(id)'
        ).execute()
        
        files = results.get('files', [])
        if not files:
            print("⚠️ No se encontró archivo de metadata. Se usará clasificación analítica base.")
            return {}
        
        file_id = files[0]['id']
        content = self.service.files().get_media(fileId=file_id).execute()
        return json.loads(content.decode('utf-8'))

    def _scan_and_classify_data(self):
        """Sincroniza archivos y clasifica usando la homoclina de self.meta."""
        print(f"Sincronizando zonas desde Drive (Región: {self.region})...")
        all_keys = []
        page_token = None
        
        while True:
            results = self.service.files().list(
                q=f"'{self.target_folder_id}' in parents and name contains '.npz' and trashed=false",
                fields='nextPageToken, files(name)',
                pageToken=page_token, 
                pageSize=1000
            ).execute()
            
            for item in results.get('files', []):
                name = item['name']
                if name not in ['t_eval.npz'] and not name.startswith('grid_metadata'):
                    all_keys.append(name.replace('.npz', ''))
                    
            page_token = results.get('nextPageToken', None)
            if not page_token: break

        # Recuperar curva homoclina de la metadata cargada
        hc_curve = self.meta.get("_detected_homoclinic_curve", None)

        for key in all_keys:
            try:
                params = parse_param_key(key)
                zone = self.system.classify_point(params)
                
                # Refinamiento Z4 vs Z5
                if zone == 4 and hc_curve is not None:
                    if self._is_inside_homoclinic(params, hc_curve):
                        zone = 4 
                    else:
                        zone = 5 

                if zone in self.zone_map:
                    self.zone_map[zone].append(key)
                else:
                    self.unclassified.append(key)
            except Exception:
                continue

        print(f"Clasificación completada: {sum(len(v) for v in self.zone_map.values())} puntos.")

    def _is_inside_homoclinic(self, point, hc_curve):
        mu1_val, mu2_val = point
        hc_mu1 = np.array(hc_curve[0])
        hc_mu2 = np.array(hc_curve[1])

        if mu1_val < hc_mu1.min() or mu1_val > hc_mu1.max():
            return False

        mu2_frontier = np.interp(mu1_val, hc_mu1, hc_mu2)
        return mu2_val > mu2_frontier

    def plot_zone_distribution(self, save_to_drive=False, show_plot=True):
        plt.figure(figsize=(10, 8))
        
        if hasattr(self.system, 'get_bifurcation_curves'):
            hc = self.meta.get("_detected_homoclinic_curve", None)
            curves = self.system().get_bifurcation_curves(detected_homoclinic=hc)
            for name, (cx, cy, color, style) in curves.items():
                plt.plot(cx, cy, color='black', ls=style, lw=1.5, alpha=0.7, label=f"Teoría: {name}")

        cmap = plt.cm.get_cmap('tab10')
        for i, (zone_id, zone_desc) in enumerate(self.system.zone_names.items()):
            keys = self.zone_map.get(zone_id, [])
            if not keys: continue
            
            coords = np.array([parse_param_key(k) for k in keys])
            plt.scatter(coords[:, 0], coords[:, 1], color=cmap(i), 
                        label=f"Zona {zone_id}: {zone_desc}", s=20, alpha=0.7)

        plt.title(f"Mapa de Zonas Dinámicas: {self.system.name} (Región: {self.region})")
        plt.xlabel(fr"${self.system.param_names[0]}$")
        plt.ylabel(fr"${self.system.param_names[1]}$")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Ajustar los ejes al param_ranges de la región activa
        plt.xlim(self.system.param_ranges[0])
        plt.ylim(self.system.param_ranges[1])
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_to_drive:
            self._save_plot_to_drive()

        if show_plot: plt.show()
        else: plt.close()

    def _save_plot_to_drive(self):
        filename = f"zones_map_{self.system_folder_name}_{self.region}.png"
        local_path = os.path.join(tempfile.gettempdir(), filename)
        plt.savefig(local_path, dpi=150)
        with open(local_path, "rb") as f:
            media = MediaIoBaseUpload(f, mimetype='image/png')
            self.service.files().create(
                body={'name': filename, 'parents': [self.target_folder_id]},
                media_body=media
            ).execute()
        os.remove(local_path)
        print(f"✅ Mapa subido a Drive: {filename}")

if __name__ == "__main__":
    # Puedes cambiar "base" por "far_z1", "far_z2", o "far_z5" para testear
    manager = DataZoneManager(region="base")
    manager.plot_zone_distribution(save_to_drive=True)