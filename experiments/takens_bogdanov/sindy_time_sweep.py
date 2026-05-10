#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script CLOUD-NATIVE: Cómputo de Barrido Temporal SINDy con Checkpoints.
=======================================================================
Entrena ensambles SINDy variando el tiempo máximo de integración (t_max).
Guarda los resultados GRADUALMENTE en archivos JSON independientes dentro 
de la carpeta correspondiente a cada configuración de muestreo.
"""

import os
import sys
import io
import json
import random
import tempfile
import numpy as np
import pysindy as ps
from enum import Enum
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System
from core.io import parse_param_key
from core.drive_auth import get_drive_service, get_target_folder_id
from experiments.takens_bogdanov.data_zone_manager import DataZoneManager, _build_drive_service

# --- SELECCIÓN DE REGIÓN ---
TARGET_REGION = "base" 
System.set_region(TARGET_REGION)

# =============================================================================
# ENUMS Y CONFIGURACIÓN DEL EXPERIMENTO
# =============================================================================
class SamplingMode(Enum):
    ALL_ZONES = "all_zones"
    ONE_ZONE_VAR_PARAMS = "one_zone_var_params"
    ONE_ZONE_FIXED_PARAMS = "one_zone_fixed_params"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

class ExperimentConfig:
    def __init__(self):
        # Barrido Temporal
        initial_t_swep_range = np.linspace(0, 2, 21)[1:]
        self.t_sweep_range = np.concatenate((initial_t_swep_range, np.linspace(2.0, 10.0, 10)))
        self.n_iterations = 10         
        
        # Muestreo
        self.sampling_mode = SamplingMode.ALL_ZONES
        self.target_zone = 5           
        self.n_trajs_total = 5         
        self.train_region = TARGET_REGION
        
        # Hiperparámetros SINDy
        self.poly_degree = 3
        self.threshold = 0.1         
        self.alpha = 0.05          
        self.unbias = True             
        
        self.system_folder = System.name.lower().replace("-", "_").replace(" ", "_")
        self.tmp_dir = os.path.join(tempfile.gettempdir(), f"sindy_sweep_{self.system_folder}_{self.train_region}")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def get_config_name(self):
        if self.sampling_mode == SamplingMode.ALL_ZONES:
            return "ALL_ZONES (Mezcla)"
        else:
            tipo = "Var Params" if self.sampling_mode == SamplingMode.ONE_ZONE_VAR_PARAMS else "Fixed Params"
            return f"Z{self.target_zone} - {tipo}"
            
    def get_folder_name(self):
        """Nombre normalizado para la carpeta en Google Drive."""
        if self.sampling_mode == SamplingMode.ALL_ZONES:
            return f"batch_{self.sampling_mode.value}"
        else:
            return f"batch_{self.sampling_mode.value}_zone_{self.target_zone}"

    def to_dict(self):
        return {
            "n_iterations": self.n_iterations,
            "sampling_mode": self.sampling_mode.value,
            "target_zone": self.target_zone,
            "n_trajs_total": self.n_trajs_total,
            "train_region": self.train_region,
            "sindy_hyperparams": {
                "poly_degree": self.poly_degree,
                "threshold": self.threshold,
                "alpha": self.alpha,
                "unbias": self.unbias
            }
        }

# =============================================================================
# GESTIÓN DE I/O Y DATOS (NUBE)
# =============================================================================
class DriveHelper:
    """Clase utilitaria para interactuar con Google Drive modularmente."""
    def __init__(self, service):
        self.service = service

    def get_or_create_folder(self, parent_id, name):
        query = f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        res = self.service.files().list(q=query, fields='files(id)').execute().get('files', [])
        if res: return res[0]['id']
        meta = {'name': name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_id]}
        return self.service.files().create(body=meta, fields='id').execute().get('id')

    def download_file(self, file_id, local_path):
        if os.path.exists(local_path): return local_path 
        request = self.service.files().get_media(fileId=file_id)
        fh = io.FileIO(local_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: _, done = downloader.next_chunk()
        return local_path

    def update_or_create_json(self, data_dict, file_name, folder_id, existing_file_id=None):
        """Guarda un JSON en memoria y lo sube/actualiza en Drive."""
        raw = json.dumps(data_dict, indent=2, cls=NpEncoder).encode("utf-8")
        fh = io.BytesIO(raw)
        media = MediaIoBaseUpload(fh, mimetype="application/json", resumable=True)
        
        if existing_file_id:
            self.service.files().update(fileId=existing_file_id, media_body=media).execute()
            return existing_file_id
        else:
            meta = {'name': file_name, 'parents': [folder_id]}
            res = self.service.files().create(media_body=media, body=meta, fields='id').execute()
            return res.get('id')


class SindyDataFetcher:
    """Encargado de descargar y preparar trayectorias truncadas."""
    def __init__(self, config, drive_helper, manager):
        self.cfg = config
        self.drive = drive_helper
        self.manager = manager
        
        base_id = get_target_folder_id()
        self.sys_id = self.drive.get_or_create_folder(base_id, self.cfg.system_folder)
        self.region_id = self.drive.get_or_create_folder(self.sys_id, self.cfg.train_region)
        self.traj_id = self.drive.get_or_create_folder(self.region_id, "trajectories")
        self.models_id = self.drive.get_or_create_folder(self.region_id, "sindy_models")
        
        self.drive_index = {}
        page_token = None
        while True:
            res = self.drive.service.files().list(
                q=f"'{self.traj_id}' in parents and trashed=false",
                fields='nextPageToken, files(id, name)',
                pageToken=page_token
            ).execute()
            for i in res.get('files', []): 
                self.drive_index[i['name']] = i['id']
            page_token = res.get('nextPageToken', None)
            if not page_token: break

        teval_path = os.path.join(self.cfg.tmp_dir, "t_eval.npz")
        self.drive.download_file(self.drive_index["t_eval.npz"], teval_path)
        with np.load(teval_path) as data:
            self.t_full = data["t_eval"]

    def fetch_iteration_data(self, current_t_max):
        plan = self._get_sampling_plan()
        X_list, t_list = [], []
        
        limit_idx = len(self.t_full)
        valid_indices = np.where(self.t_full <= current_t_max)[0]
        if len(valid_indices) > 0: 
            limit_idx = valid_indices[-1] + 1

        for key, n_trajs in plan.items():
            fname = f"{key}.npz"
            if fname not in self.drive_index: continue
            
            local_path = os.path.join(self.cfg.tmp_dir, fname)
            self.drive.download_file(self.drive_index[fname], local_path)
            
            with np.load(local_path) as data:
                raw_trajs = data["trajectories"]
            
            current_limit = min(limit_idx, raw_trajs.shape[2])
            trajs = raw_trajs[:, :, :current_limit]
            current_t = self.t_full[:current_limit]
            
            n_sims = trajs.shape[0]
            actual_n = min(n_trajs, n_sims)
            selected_idx = random.sample(range(n_sims), actual_n)
            
            param_vals = parse_param_key(key)
            param_block = np.tile(param_vals, (current_limit, 1))
            
            for i in selected_idx:
                X_aug = np.hstack((trajs[i].T, param_block))
                X_list.append(X_aug)
                t_list.append(current_t)
                
        return X_list, t_list

    def _get_sampling_plan(self):
        plan = defaultdict(int)
        zm = self.manager.zone_map

        if self.cfg.sampling_mode == SamplingMode.ALL_ZONES:
            zones_available = [z for z in zm.keys() if len(zm[z]) > 0]
            zones_to_sample = random.sample(zones_available, min(self.cfg.n_trajs_total, len(zones_available)))
            for z in zones_to_sample: plan[random.choice(zm[z])] = 1
        elif self.cfg.sampling_mode == SamplingMode.ONE_ZONE_VAR_PARAMS:
            keys_in_zone = zm.get(self.cfg.target_zone, [])
            keys = random.sample(keys_in_zone, min(self.cfg.n_trajs_total, len(keys_in_zone)))
            for k in keys: plan[k] = 1
        elif self.cfg.sampling_mode == SamplingMode.ONE_ZONE_FIXED_PARAMS:
            keys_in_zone = zm.get(self.cfg.target_zone, [])
            plan[random.choice(keys_in_zone)] = self.cfg.n_trajs_total
            
        return plan

# =============================================================================
# EVALUACIÓN
# =============================================================================
class SindyEvaluator:
    @staticmethod
    def normalize_feature_name(name):
        return " ".join(sorted(name.split(" ")))

    @classmethod
    def evaluate_coefficients(cls, model, true_coefs_list):
        pred_matrix = model.coefficients()
        norm_features = [cls.normalize_feature_name(f) for f in model.get_feature_names()]
        mae_sum, mse_sum, mae_count = 0.0, 0.0, 0
        
        for i, true_eq_dict in enumerate(true_coefs_list):
            norm_true_dict = {cls.normalize_feature_name(k): v for k, v in true_eq_dict.items()}
            for j, f_name in enumerate(norm_features):
                pred_val = pred_matrix[i, j]
                true_val = norm_true_dict.get(f_name, 0.0)
                
                if abs(pred_val) > 1e-10 or abs(true_val) > 1e-10:
                    err = abs(pred_val - true_val)
                    mse_sum += err ** 2
                    mae_count += 1
                    
        return mse_sum / mae_count if mae_count > 0 else float('inf')

# =============================================================================
# ORQUESTADOR PRINCIPAL (Maneja el Guardado Gradual)
# =============================================================================
class TimeSweepRunner:
    def __init__(self, config):
        self.cfg = config
        raw_service = get_drive_service()
        self.service = _build_drive_service(raw_service)
        self.drive = DriveHelper(self.service)
        
        self.manager = DataZoneManager(system_class=System, service=self.service, region=self.cfg.train_region)
        self.fetcher = SindyDataFetcher(self.cfg, self.drive, self.manager)
        
        # Preparación de la carpeta Batch
        batch_folder_name = self.cfg.get_folder_name()
        self.batch_id = self.drive.get_or_create_folder(self.fetcher.models_id, batch_folder_name)

    def run(self):
        print(f"\n🚀 Iniciando: {self.cfg.get_config_name()}")
        
        feature_names = System.state_names + System.param_names
        true_coefs = System.get_true_coefficients()
        
        results_records = []
        
        # Archivo que iremos sobreescribiendo gradualmente
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"time_sweep_{timestamp}.json"
        cloud_file_id = None 
        
        # Diccionario maestro a guardar
        payload = {
            "config": self.cfg.to_dict(),
            "config_label": self.cfg.get_config_name(),
            "results": []
        }

        # Loop Principal
        for t_target in tqdm(self.cfg.t_sweep_range, desc="Barrido Temporal"):
            for i in range(self.cfg.n_iterations):
                X_list, t_list = self.fetcher.fetch_iteration_data(t_target)
                if not X_list: continue
                
                try:
                    X_dot_list = [np.gradient(X, t, axis=0) for X, t in zip(X_list, t_list)]
                    
                    optimizer = ps.STLSQ(threshold=self.cfg.threshold, alpha=self.cfg.alpha, unbias=self.cfg.unbias)
                    library = ps.PolynomialLibrary(degree=self.cfg.poly_degree)
                    model = ps.SINDy(optimizer=optimizer, feature_library=library)
                    
                    model.fit(X_list, x_dot=X_dot_list, t=t_list, feature_names=feature_names)
                    mse_val = SindyEvaluator.evaluate_coefficients(model, true_coefs)
                except Exception as e:
                    print(f"\n⚠️ Error en t={t_target}, iter={i}: {e}")
                    mse_val = np.nan
                
                record = {
                    "t_max": float(t_target),
                    "iteration": i + 1,
                    "mse_global": float(mse_val)
                }
                results_records.append(record)
            
            # --- GUARDADO GRADUAL (CHECKPOINT) ---
            payload["results"] = results_records
            cloud_file_id = self.drive.update_or_create_json(payload, file_name, self.batch_id, cloud_file_id)

        print(f"✅ Resultados guardados en Drive: {self.cfg.get_folder_name()}/{file_name}")


if __name__ == "__main__":
    RUN_ALL_CONFIGS = True 
    
    if RUN_ALL_CONFIGS:
        print("\n" + "="*60)
        print("MODO AUTOMÁTICO: Evaluando las 11 configuraciones")
        print("="*60)
        
        configs_to_run = []
        c_all = ExperimentConfig()
        c_all.sampling_mode = SamplingMode.ALL_ZONES
        configs_to_run.append(c_all)
        
        for z in range(1, 6):
            c_var = ExperimentConfig()
            c_var.sampling_mode = SamplingMode.ONE_ZONE_VAR_PARAMS
            c_var.target_zone = z
            configs_to_run.append(c_var)
            
        for z in range(1, 6):
            c_fixed = ExperimentConfig()
            c_fixed.sampling_mode = SamplingMode.ONE_ZONE_FIXED_PARAMS
            c_fixed.target_zone = z
            configs_to_run.append(c_fixed)
            
        for idx, config in enumerate(configs_to_run):
            try:
                runner = TimeSweepRunner(config)
                runner.run()
            except Exception as e:
                print(f"❌ Error crítico en config {config.get_config_name()}: {e}")
                
    else:
        # Ejecución Manual
        config = ExperimentConfig()
        config.sampling_mode = SamplingMode.ALL_ZONES 
        config.n_iterations = 10      
        runner = TimeSweepRunner(config)
        runner.run()

    # Limpieza final de caché de descargas
    try:
        import shutil
        shutil.rmtree(config.tmp_dir, ignore_errors=True)
        print("🧹 Caché temporal limpiado.")
    except: pass