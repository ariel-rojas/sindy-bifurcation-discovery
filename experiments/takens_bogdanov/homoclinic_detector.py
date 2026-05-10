
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detector de Bifurcación Homoclina (Shooting de Variedad Inestable)
==================================================================
Calcula la curva homoclina de manera matemáticamente rigurosa integrando 
la variedad inestable del nodo silla. Luego usa esta frontera analítica 
para aplicar un relleno lógico sobre la grilla discreta de Drive.
"""

import io
import os
import sys
import json
import numpy as np
from numba import jit
from scipy.optimize import brentq

# =============================================================================
# CONFIGURACIÓN DE RUTAS Y METADATA
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from systems.takens_bogdanov import TakensBogdanov as System
from core.io import make_param_key, parse_param_key
from core.utils import generate_param_grid
from core.drive_auth import get_drive_service, get_target_folder_id

DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
GRID_DENSITY = 100
N_TRAJ_PER_AXIS = 10
VF_RESOLUTION = 100
T_SPAN = [0.0, 10]
N_STEPS = 1000

def build_system_info():
    return {
        "system_name": System.name,
        "state_names": list(System.state_names),
        "param_names": list(System.param_names),
        "sweep_ranges": [list(r) for r in System.param_ranges],
        "grid_density": GRID_DENSITY,
        "n_traj_per_axis": N_TRAJ_PER_AXIS,
        "vf_resolution": VF_RESOLUTION,
        "t_span": list(T_SPAN),
        "n_steps": N_STEPS,
    }

# =============================================================================
# FUNCIONES DE DRIVE Y JSON
# =============================================================================
def get_drive_index_full(session, folder_id):
    idx = {}
    token = None
    while True:
        params = {"q": f"'{folder_id}' in parents and trashed=false", "fields": "nextPageToken, files(id, name)", "pageSize": 1000}
        if token: params["pageToken"] = token
        response = session.get(DRIVE_FILES_URL, params=params)
        response.raise_for_status()
        body = response.json()
        for f in body.get("files", []): idx[f["name"]] = f["id"]
        token = body.get("nextPageToken")
        if not token: break
    return idx

def download_bytes(session, file_id):
    url = f"{DRIVE_FILES_URL}/{file_id}?alt=media"
    response = session.get(url)
    response.raise_for_status()
    return response.content

def find_subfolder(session, parent_id, name, fuzzy=False):
    folders = []
    token = None
    while True:
        params = {"q": (f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"), "fields": "nextPageToken, files(id, name, modifiedTime)", "pageSize": 1000}
        if token: params["pageToken"] = token
        response = session.get(DRIVE_FILES_URL, params=params)
        response.raise_for_status()
        body = response.json()
        folders.extend(body.get("files", []))
        token = body.get("nextPageToken")
        if not token: break
    candidates = [f for f in folders if (name.lower() in f["name"].lower() if fuzzy else f["name"] == name)]
    if not candidates: raise FileNotFoundError(f"No se encontró la subcarpeta '{name}'.")
    return candidates[0]["id"] if not fuzzy else sorted(candidates, key=lambda x: x.get("modifiedTime", ""), reverse=True)[0]["id"]

def upload_dict_as_json_to_drive(session, data_dict, file_name, folder_id, drive_service_classic):
    from googleapiclient.http import MediaIoBaseUpload
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
    results = drive_service_classic.files().list(q=query, fields="files(id)").execute()
    for item in results.get("files", []):
        try: drive_service_classic.files().delete(fileId=item["id"]).execute()
        except: pass

    json_bytes = json.dumps(data_dict, indent=2, cls=NumpyEncoder).encode('utf-8')
    fh = io.BytesIO(json_bytes)
    media = MediaIoBaseUpload(fh, mimetype='application/json', resumable=True)
    metadata = {'name': file_name, 'parents': [folder_id]}
    print(f"Subiendo {file_name} a Drive...")
    drive_service_classic.files().create(media_body=media, body=metadata).execute()


# =============================================================================
# NÚCLEO FÍSICO: SEGUIMIENTO DE VARIEDAD INESTABLE
# =============================================================================
@jit(nopython=True, cache=True)
def _tb_homoclinic_residual_kernel(mu1, mu2, x_saddle, dt, t_max):
    """
    Integra a lo largo del autovector inestable de la silla.
    Retorna positivo si escapa (Zona 4), negativo si espirala hacia adentro (Zona 5).
    """
    # 1. Jacobiano en la silla (y=0)
    det = mu2 - 2.0 * x_saddle + 3.0 * x_saddle**2
    tr = -(x_saddle**2 + x_saddle)
    
    # 2. Autovalor inestable (λ_u > 0)
    lambda_u = (tr + np.sqrt(tr**2 - 4.0 * det)) / 2.0
    
    # 3. Condición inicial: Micro-paso a la izquierda sobre el autovector (1, λ_u)
    eps = 1e-4
    x = x_saddle - eps
    y = -eps * lambda_u
    
    n_steps = int(t_max / dt)
    max_x = x

    for _ in range(n_steps):
        k1x = y
        k1y = -mu1 - mu2*x + x*x - x*x*x - (x*x + x)*y
        xa, ya = x + 0.5 * dt * k1x, y + 0.5 * dt * k1y
        
        k2x = ya
        k2y = -mu1 - mu2*xa + xa*xa - xa*xa*xa - (xa*xa + xa)*ya
        xb, yb = x + 0.5 * dt * k2x, y + 0.5 * dt * k2y
        
        k3x = yb
        k3y = -mu1 - mu2*xb + xb*xb - xb*xb*xb - (xb*xb + xb)*yb
        xc, yc = x + dt * k3x, y + dt * k3y
        
        k4x = yc
        k4y = -mu1 - mu2*xc + xc*xc - xc*xc*xc - (xc*xc + xc)*yc

        x = x + (dt / 6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x)
        y = y + (dt / 6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y)

        if x > max_x:
            max_x = x
        
        # Si cruza el saddle hacia la derecha, escapó (no hay homoclina)
        if max_x > x_saddle:
            return max_x - x_saddle

    return max_x - x_saddle

def _shooting_residual(mu1, mu2, dt=0.02, t_max=20000.0):
    coeffs = np.array([1.0, -1.0, float(mu2), float(mu1)])
    roots = np.roots(coeffs)
    real_roots = sorted(r.real for r in roots if abs(r.imag) < 1e-9)

    if len(real_roots) != 3: return np.nan
    x_saddle = float(real_roots[1])

    det = float(mu2) - 2.0 * x_saddle + 3.0 * x_saddle**2
    if det >= 0.0: return np.nan

    # t_max aumentado a 20000 para lidiar con el ralentizamiento asintótico
    return float(_tb_homoclinic_residual_kernel(
        float(mu1), float(mu2), x_saddle, float(dt), float(t_max)
    ))

def compute_homoclinic_curve(n_points=200):
    # En Takens-Bogdanov normalizado, la curva termina exactamente en mu2 = 0.25 (intersección con Hopf)
    mu2_grid = np.linspace(0.002, 0.2495, n_points)
    mu1_results, mu2_results = [0.0], [0.0]
    
    last_mu1, last_sn_right = None, None

    print("Calculando curva homoclina analítica profunda...")
    for mu2 in mu2_grid:
        disc = np.sqrt(4.0 - 12.0 * mu2)
        t_right = (2.0 + disc) / 6.0
        mu1_sn_right = 2.0 * t_right**3 - t_right**2
        
        if mu1_sn_right <= 1e-7: continue
        upper = mu1_sn_right * 0.999
        f = lambda m: _shooting_residual(m, mu2)
        bracket = None

        if last_mu1 is not None and last_sn_right is not None:
            guess = (last_mu1 / last_sn_right) * mu1_sn_right
            guess = max(1e-7, min(upper, guess))
            for delta_frac in (0.02, 0.1, 0.3, 0.8):
                delta = delta_frac * upper
                a, b = max(1e-7, guess - delta), min(upper, guess + delta)
                if a >= b: continue
                f_a, f_b = f(a), f(b)
                if np.isfinite(f_a) and np.isfinite(f_b) and f_a < 0.0 < f_b:
                    bracket = (a, b)
                    break

        if bracket is None:
            # Escaneo denso si el heurístico falla cerca de la cresta
            mu1_scan = np.linspace(1e-7, upper, 250)
            residuals = np.array([f(m) for m in mu1_scan])
            last_neg_idx = None
            for i, r in enumerate(residuals):
                if not np.isfinite(r): continue
                if r < 0.0:
                    last_neg_idx = i
                elif r > 0.0 and last_neg_idx is not None:
                    bracket = (mu1_scan[last_neg_idx], mu1_scan[i])
                    break

        if bracket is not None:
            try:
                mu1_h = brentq(f, bracket[0], bracket[1], xtol=1e-8, rtol=1e-9, maxiter=100)
                mu1_results.append(mu1_h)
                mu2_results.append(mu2)
                last_mu1, last_sn_right = mu1_h, mu1_sn_right
            except: pass

    idx = np.argsort(mu2_results)
    return np.asarray(mu1_results, dtype=np.float64)[idx], np.asarray(mu2_results, dtype=np.float64)[idx]


# =============================================================================
# FLUJO PRINCIPAL Y RELLENO LÓGICO
# =============================================================================
def main():
    print("--- Iniciando Detector Homoclino (Unstable Manifold + Metadata) ---")
    session = get_drive_service()
    from googleapiclient.discovery import build
    drive_classic = build("drive", "v3", credentials=getattr(session, "credentials", session))
    
    base_id = get_target_folder_id()
    SYSTEM_FOLDER_NAME = System.name.lower().replace("-", "_").replace(" ", "_")
    
    try:
        sys_id = find_subfolder(session, base_id, SYSTEM_FOLDER_NAME, fuzzy=True)
        traj_id = find_subfolder(session, sys_id, "trajectories")
    except Exception as e:
        print(f"Error accediendo a Drive: {e}")
        return

    print("Obteniendo índice de archivos en Drive...")
    idx = get_drive_index_full(session, traj_id)
    meta_name = f"grid_metadata_{SYSTEM_FOLDER_NAME}.json"
    metadata = {}

    if meta_name in idx:
        print("✅ Metadatos encontrados en Drive. Descargando...")
        meta_bytes = download_bytes(session, idx[meta_name])
        metadata = json.loads(meta_bytes.decode("utf-8"))
    else:
        print("⚠️ Metadatos no encontrados. Reconstruyendo desde cero...")
        metadata["_system_info"] = build_system_info()
        all_params = generate_param_grid(System.param_ranges, GRID_DENSITY)
        for p in all_params:
            key = make_param_key(p)
            if f"{key}.npz" in idx:
                fixed_points = System.calculate_fixed_points(p)
                metadata[key] = {"num_fixed_points": int(fixed_points.shape[0] if fixed_points is not None else 0)}

    # 1. Calcular la curva continua impecable
    mu1_hc, mu2_hc = compute_homoclinic_curve()
    metadata["_detected_homoclinic_curve"] = [mu1_hc.tolist(), mu2_hc.tolist()]

    # 2. Relleno Lógico sobre la grilla discreta de la metadata
    print("Aplicando relleno lógico a la grilla basado en la frontera...")
    classified_5, classified_4 = 0, 0

    for key, info in metadata.items():
        if key.startswith("_"): continue
        
        params = parse_param_key(key)
        mu1, mu2 = params[0], params[1]

        # Solo discriminamos los puntos dentro de la cuña teórica
        if System.classify_point(params) == 4:
            # Si mu2 supera el rango (0.25), sabemos que la homoclina colapsó con Hopf. Todo es escape.
            if mu2 > mu2_hc[-1]:
                mu1_bound = -1.0
            else:
                mu1_bound = float(np.interp(mu2, mu2_hc, mu1_hc))

            if mu1 <= mu1_bound:
                metadata[key]["refined_zone"] = 5
                classified_5 += 1
            else:
                metadata[key]["refined_zone"] = 4
                classified_4 += 1

    print(f"Distribución en la cuña -> Zona 5 (ciclo): {classified_5} | Zona 4 (escape): {classified_4}")
    upload_dict_as_json_to_drive(session, metadata, meta_name, traj_id, drive_classic)
    print("✅ Proceso exitoso. Ejecuta tu InteractiveViewer para ver la magia.")

if __name__ == "__main__":
    main()