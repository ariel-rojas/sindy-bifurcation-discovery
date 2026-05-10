#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script CLOUD-NATIVE para Entrenamiento por Ensamble de SINDy.
=============================================================
Diseñado para la evaluación sistemática del descubrimiento de ecuaciones en 
sistemas no lineales (ej. Takens-Bogdanov).

Modos de Muestreo (Sampling Modes):
1. ALL_ZONES: 1 trayectoria por zona (N zonas = N trayectorias).
2. ONE_ZONE_VAR_PARAMS: N archivos distintos de una misma zona, 1 trayectoria por archivo.
3. ONE_ZONE_FIXED_PARAMS: 1 solo archivo de una zona, N trayectorias de ese archivo.
"""

import os
import sys
import io
import json
import random
import tempfile
import numpy as np
import pysindy as ps
import joblib
from enum import Enum
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# =============================================================================
# CONFIGURACIÓN DE RUTAS E IMPORTACIONES
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System
from core.io import parse_param_key
from core.drive_auth import get_drive_service, get_target_folder_id
from experiments.takens_bogdanov.data_zone_manager import DataZoneManager, _build_drive_service

# --- SELECCIÓN DE REGIÓN DE ENTRENAMIENTO (Fase 4) ---
TARGET_REGION = "far_z5" # Opciones: "base", "far_z1", "far_z2", "far_z5"
System.set_region(TARGET_REGION)

# =============================================================================
# ENUMS Y CONFIGURACIÓN
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
    """Contenedor centralizado de hiperparámetros para reproducibilidad."""
    def __init__(self):
        self.n_iterations = 20         
        self.sampling_mode = SamplingMode.ALL_ZONES
        self.target_zone = 5           
        self.n_trajs_total = 5         
        self.t_max = 5              
        self.train_region = TARGET_REGION
        
        # Hiperparámetros de SINDy
        self.poly_degree = 3
        self.threshold = 0.1         
        self.alpha = 0.05          
        self.unbias = True             
        
        self.system_folder = System.name.lower().replace("-", "_").replace(" ", "_")
        self.tmp_dir = os.path.join(tempfile.gettempdir(), f"sindy_ensemble_{self.system_folder}_{self.train_region}")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def to_dict(self):
        return {
            "n_iterations": self.n_iterations,
            "sampling_mode": self.sampling_mode.value,
            "target_zone": self.target_zone,
            "n_trajs_total": self.n_trajs_total,
            "t_max": self.t_max,
            "train_region": self.train_region,
            "sindy_hyperparams": {
                "poly_degree": self.poly_degree,
                "threshold": self.threshold,
                "alpha": self.alpha,
                "unbias": self.unbias
            }
        }

# =============================================================================
# MÓDULO DE GESTIÓN DE DATOS (CLOUD-NATIVE)
# =============================================================================
class SindyDataFetcher:
    def __init__(self, config, service, manager):
        self.cfg = config
        self.service = service
        self.manager = manager
        
        base_id = get_target_folder_id()
        self.sys_id = self._get_folder(base_id, self.cfg.system_folder)
        # Búsqueda segmentada por región (Fase 4)
        self.region_id = self._get_folder(self.sys_id, self.cfg.train_region)
        self.traj_id = self._get_folder(self.region_id, "trajectories")
        self.models_id = self._get_folder(self.region_id, "sindy_models")
        
        self.drive_index = {}
        page_token = None
        print(f"Sincronizando índice de trayectorias (Región: {self.cfg.train_region})...")
        while True:
            res = self.service.files().list(
                q=f"'{self.traj_id}' in parents and trashed=false",
                fields='nextPageToken, files(id, name)',
                pageToken=page_token
            ).execute()
            for i in res.get('files', []): 
                self.drive_index[i['name']] = i['id']
            page_token = res.get('nextPageToken', None)
            if not page_token: break

        teval_path = self._download_file("t_eval.npz", self.drive_index["t_eval.npz"])
        with np.load(teval_path) as data:
            self.t_full = data["t_eval"]

    def _get_folder(self, parent_id, name):
        query = f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        res = self.service.files().list(q=query, fields='files(id)').execute().get('files', [])
        if res: return res[0]['id']
        meta = {'name': name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_id]}
        return self.service.files().create(body=meta, fields='id').execute().get('id')

    def _download_file(self, name, file_id):
        path = os.path.join(self.cfg.tmp_dir, name)
        if os.path.exists(path): return path
        request = self.service.files().get_media(fileId=file_id)
        fh = io.FileIO(path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: _, done = downloader.next_chunk()
        return path

    def _get_sampling_plan(self):
        plan = defaultdict(int)
        zm = self.manager.zone_map

        if self.cfg.sampling_mode == SamplingMode.ALL_ZONES:
            zones_available = [z for z in zm.keys() if len(zm[z]) > 0]
            zones_to_sample = random.sample(zones_available, min(self.cfg.n_trajs_total, len(zones_available)))
            for z in zones_to_sample:
                key = random.choice(zm[z])
                plan[key] = 1
                
        elif self.cfg.sampling_mode == SamplingMode.ONE_ZONE_VAR_PARAMS:
            keys_in_zone = zm.get(self.cfg.target_zone, [])
            if not keys_in_zone: raise ValueError(f"No hay datos en la Zona {self.cfg.target_zone}")
            keys = random.sample(keys_in_zone, min(self.cfg.n_trajs_total, len(keys_in_zone)))
            for k in keys:
                plan[k] = 1
                
        elif self.cfg.sampling_mode == SamplingMode.ONE_ZONE_FIXED_PARAMS:
            keys_in_zone = zm.get(self.cfg.target_zone, [])
            if not keys_in_zone: raise ValueError(f"No hay datos en la Zona {self.cfg.target_zone}")
            key = random.choice(keys_in_zone)
            plan[key] = self.cfg.n_trajs_total
            
        return plan

    def fetch_iteration_data(self):
        plan = self._get_sampling_plan()
        X_list = []
        t_list = [] 
        
        limit_idx = len(self.t_full)
        if self.cfg.t_max is not None:
            valid_indices = np.where(self.t_full <= self.cfg.t_max)[0]
            if len(valid_indices) > 0: limit_idx = valid_indices[-1] + 1

        sampling_record = {}
        for key, n_trajs in plan.items():
            fname = f"{key}.npz"
            if fname not in self.drive_index: continue
            
            path = self._download_file(fname, self.drive_index[fname])
            with np.load(path) as data:
                raw_trajs = data["trajectories"]
            
            current_limit = min(limit_idx, raw_trajs.shape[2])
            trajs = raw_trajs[:, :, :current_limit]
            current_t = self.t_full[:current_limit]
            
            n_sims = trajs.shape[0]
            actual_n = min(n_trajs, n_sims)
            selected_idx = random.sample(range(n_sims), actual_n)
            sampling_record[key] = selected_idx
            
            param_vals = parse_param_key(key)
            param_block = np.tile(param_vals, (current_limit, 1))
            
            for i in selected_idx:
                X_aug = np.hstack((trajs[i].T, param_block))
                X_list.append(X_aug)
                t_list.append(current_t)
                
            os.remove(path)
            
        return X_list, t_list, sampling_record

# =============================================================================
# MÓDULO DE EVALUACIÓN
# =============================================================================
class SindyEvaluator:
    @staticmethod
    def normalize_feature_name(name):
        parts = name.split(" ")
        return " ".join(sorted(parts))

    @classmethod
    def evaluate_coefficients(cls, model, true_coefs_list):
        pred_matrix = model.coefficients()
        feature_names = model.get_feature_names()
        
        norm_features = [cls.normalize_feature_name(f) for f in feature_names]
        
        stats = {
            "mae_global": 0.0,
            "mse_global": 0.0,
            "f1_score": 0.0,
            "equations": []
        }
        
        total_tp, total_fp, total_fn = 0, 0, 0
        mae_sum, mse_sum, mae_count = 0.0, 0.0, 0
        
        for i, true_eq_dict in enumerate(true_coefs_list):
            eq_stats = {"tp": 0, "fp": 0, "fn": 0, "errors": {}}
            
            norm_true_dict = {cls.normalize_feature_name(k): v for k, v in true_eq_dict.items()}
            
            for j, f_name in enumerate(norm_features):
                pred_val = pred_matrix[i, j]
                true_val = norm_true_dict.get(f_name, 0.0)
                
                is_pred_nonzero = abs(pred_val) > 1e-6
                is_true_nonzero = abs(true_val) > 1e-6
                
                if is_pred_nonzero and is_true_nonzero:
                    eq_stats["tp"] += 1
                    err = abs(pred_val - true_val)
                    sq_err = err ** 2
                    eq_stats["errors"][f_name] = {"pred": pred_val, "true": true_val, "mae": err, "mse": sq_err}
                    mae_sum += err
                    mse_sum += sq_err
                    mae_count += 1
                elif is_pred_nonzero and not is_true_nonzero:
                    eq_stats["fp"] += 1
                    err = abs(pred_val)
                    sq_err = err ** 2
                    eq_stats["errors"][f_name] = {"pred": pred_val, "true": 0.0, "mae": err, "mse": sq_err}
                    mae_sum += err
                    mse_sum += sq_err
                    mae_count += 1
                elif not is_pred_nonzero and is_true_nonzero:
                    eq_stats["fn"] += 1
                    
            total_tp += eq_stats["tp"]
            total_fp += eq_stats["fp"]
            total_fn += eq_stats["fn"]
            stats["equations"].append(eq_stats)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        
        stats["f1_score"] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        stats["mae_global"] = mae_sum / mae_count if mae_count > 0 else float('inf')
        stats["mse_global"] = mse_sum / mae_count if mae_count > 0 else float('inf')
        
        return stats

# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================
class EnsembleSindyTrainer:
    def __init__(self, config):
        self.cfg = config
        
        raw_service = get_drive_service()
        self.service = _build_drive_service(raw_service)
        self.manager = DataZoneManager(system_class=System, service=self.service, region=self.cfg.train_region)
        self.fetcher = SindyDataFetcher(self.cfg, self.service, self.manager)

    def run_ensemble(self):
        print(f"\n🚀 Iniciando Ensamble SINDy ({self.cfg.n_iterations} iteraciones)")
        print(f"Modo: {self.cfg.sampling_mode.value} | Zona: {self.cfg.target_zone} | Trayectorias/Modelo: {self.cfg.n_trajs_total}")
        
        feature_names = System.state_names + System.param_names
        true_coefs = System.get_true_coefficients()
        
        ensemble_results = []
        all_coef_matrices = []

        for i in tqdm(range(self.cfg.n_iterations), desc="Entrenando Ensamble"):
            X_list, t_list, samp_rec = self.fetcher.fetch_iteration_data()
            if not X_list: continue
            
            X_dot_list = [np.gradient(X, t, axis=0) for X, t in zip(X_list, t_list)]
            
            optimizer = ps.STLSQ(threshold=self.cfg.threshold, alpha=self.cfg.alpha, unbias=self.cfg.unbias)
            library = ps.PolynomialLibrary(degree=self.cfg.poly_degree)
            
            model = ps.SINDy(optimizer=optimizer, feature_library=library, differentiation_method=ps.FiniteDifference())
            model.fit(X_list, x_dot=X_dot_list, t=t_list, feature_names=feature_names)
            
            eval_stats = SindyEvaluator.evaluate_coefficients(model, true_coefs)
            all_coef_matrices.append(model.coefficients())
            
            ensemble_results.append({
                "iteration": i + 1,
                "sampling_record": samp_rec,
                "mae_global": eval_stats["mae_global"],
                "mse_global": eval_stats["mse_global"],
                "f1_score": eval_stats["f1_score"],
                "coefficients": model.coefficients()
            })

        self._save_ensemble_results(ensemble_results, all_coef_matrices, model.get_feature_names(), true_coefs)

    def _save_ensemble_results(self, iter_results, matrices, f_names, true_coefs_list):
        print("\n📊 Computando estadísticas del Ensamble...")
        stack = np.array(matrices) 
        
        mean_mae = np.mean([r["mae_global"] for r in iter_results])
        mean_mse = np.mean([r["mse_global"] for r in iter_results])
        mean_f1 = np.mean([r["f1_score"] for r in iter_results])
        
        print(f"Resultados Globales -> MAE Medio: {mean_mae:.4f} | MSE Medio: {mean_mse:.4f} | F1-Score Medio: {mean_f1:.4f}")

        coef_summary = {}
        for eq_idx, eq_name in enumerate(["x'", "y'"]):
            coef_summary[eq_name] = {}
            norm_true_dict = {SindyEvaluator.normalize_feature_name(k): v for k, v in true_coefs_list[eq_idx].items()}

            for f_idx, f_name in enumerate(f_names):
                norm_f_name = SindyEvaluator.normalize_feature_name(f_name)
                true_val = norm_true_dict.get(norm_f_name, 0.0)
                
                term_preds = stack[:, eq_idx, f_idx]
                term_abs_errs = np.abs(term_preds - true_val)
                term_sq_errs = term_abs_errs ** 2

                if abs(true_val) > 1e-6 or np.mean(np.abs(term_preds)) > 1e-5:
                    coef_summary[eq_name][f_name] = {
                        "true_value": true_val,
                        "predictions_stats": {
                            "mean": float(np.mean(term_preds)),
                            "std": float(np.std(term_preds)),
                            "median": float(np.median(term_preds)),
                            "q25": float(np.percentile(term_preds, 25)),
                            "q75": float(np.percentile(term_preds, 75)),
                            "min": float(np.min(term_preds)),
                            "max": float(np.max(term_preds))
                        },
                        "error_stats": {
                            "mean_mae": float(np.mean(term_abs_errs)),
                            "std_mae": float(np.std(term_abs_errs)),
                            "mean_mse": float(np.mean(term_sq_errs)),
                            "std_mse": float(np.std(term_sq_errs))
                        }
                    }

        final_metadata = {
            "timestamp": datetime.now().isoformat(),
            "config": self.cfg.to_dict(),
            "ensemble_metrics": {
                "mean_mae": mean_mae,
                "mean_mse": mean_mse,
                "mean_f1_score": mean_f1
            },
            "discovered_coefficients": coef_summary,
            "iteration_logs": iter_results
        }
        
        if self.cfg.sampling_mode == SamplingMode.ALL_ZONES:
            batch_folder = f"batch_{self.cfg.sampling_mode.value}"
        else:
            batch_folder = f"batch_{self.cfg.sampling_mode.value}_zone_{self.cfg.target_zone}"
            
        b_id = self.fetcher._get_folder(self.fetcher.models_id, batch_folder)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        v_id = self.fetcher._get_folder(b_id, f"run_{timestamp_str}")
        
        meta_path = os.path.join(self.cfg.tmp_dir, "ensemble_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(final_metadata, f, indent=2, cls=NpEncoder)
            
        self._upload_to_drive(meta_path, "ensemble_metadata.json", v_id)
        print(f"✅ Metadata del ensamble guardada en la carpeta de Drive: {batch_folder}/run_{timestamp_str}")

    def _upload_to_drive(self, local_path, name, folder_id):
        with open(local_path, "rb") as fd:
            media = MediaIoBaseUpload(fd, mimetype='application/json', resumable=True)
            self.service.files().create(media_body=media, body={'name': name, 'parents': [folder_id]}).execute()

if __name__ == "__main__":
    # --- Bandera Maestra ---
    RUN_ALL_CONFIGS = False 
    
    if RUN_ALL_CONFIGS:
        print("\n" + "="*60)
        print("MODO AUTOMÁTICO: Ejecutando las 11 configuraciones")
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
            print(f"\n\n{'='*50}")
            print(f"Ejecutando Configuración {idx+1} de 11...")
            print(f"Modo: {config.sampling_mode.value} | Zona: {config.target_zone if config.sampling_mode != SamplingMode.ALL_ZONES else 'Todas'}")
            print(f"{'='*50}")
            try:
                trainer = EnsembleSindyTrainer(config)
                trainer.run_ensemble()
            except Exception as e:
                print(f"❌ Error en la configuración {idx+1}: {e}. Saltando a la siguiente...")
                
    else:
        # --- Ejecución Manual / Individual ---
        config = ExperimentConfig()
        config.sampling_mode = SamplingMode.ALL_ZONES 
        config.n_iterations = 10      
        config.n_trajs_total = 5      
        config.target_zone = 1        
        config.t_max = 5
        
        trainer = EnsembleSindyTrainer(config)
        trainer.run_ensemble()