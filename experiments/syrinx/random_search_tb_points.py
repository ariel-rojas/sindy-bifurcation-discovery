#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import time

# ============================================================
# IMPORTACIÓN DEL MODELO SYRINX
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from systems.syrinx import SyrinxModel

# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================
OUTPUT_DIR = "resultados"
IMG_DIR = os.path.join(OUTPUT_DIR, "imagenes")
JSON_PATH = os.path.join(OUTPUT_DIR, "casos_validos.jsonl")

os.makedirs(IMG_DIR, exist_ok=True)

# Rango fisiológico aceptable de presión subsiríngea [din/cm^2]
PSUB_MIN = 5000
PSUB_MAX = 30000

# ============================================================
# 2. RANGOS DE PARÁMETROS (muestreo log-uniforme)
# ============================================================
param_ranges = {
    "m":       {"min": 1e-4,  "max": 1e-2},
    "tau":     {"min": 1e-6,  "max": 5e-4},
    "a01":     {"min": 5e-2,  "max": 2e-1},
    "alab":    {"min": 1e-4,  "max": 1e-3},
    "kappa2":  {"min": 100.0, "max": 1000.0},
    "gamma1":  {"min": 1e-3,  "max": 1e-1},
    "c":       {"min": 1e-2,  "max": 1e2},
    "delta_a": {"min": -0.1,  "max": -0.001},
    "f0":      {"min": 0.001, "max": 0.1},
}

# ============================================================
# 3. MUESTREO ALEATORIO
# ============================================================
def sample_log(min_val, max_val):
    """
    Muestreo log-uniforme preservando signo.
    """
    sign = np.sign(min_val)
    val = np.exp(np.random.uniform(np.log(abs(min_val)), np.log(abs(max_val))))
    return sign * val

def sample_params():
    """
    Genera un conjunto aleatorio de parámetros fisiológicos.
    """
    return {k: sample_log(v["min"], v["max"]) for k, v in param_ranges.items()}

# ============================================================
# 4. MODELO: PUNTOS FIJOS
# ============================================================
def compute_critical(params):
    """
    Calcula valores críticos donde:
    - término lineal ≈ 0
    - término independiente ≈ 0
    """
    k1_star = params["f0"] / params["a01"]
    psub_star = -(params["f0"] * params["a01"]) / (params["alab"] * params["delta_a"])
    return k1_star, psub_star

def get_real_roots(params, k1, psub):
    """
    Calcula raíces reales del polinomio cuártico efectivo.
    Elimina la raíz espuria x = -a01.
    """
    k2 = params["kappa2"]
    a01 = params["a01"]
    alab = params["alab"]
    delta_a = params["delta_a"]
    f0 = params["f0"]

    C = alab * psub * delta_a

    coeffs = [
        -k2,
        -k2 * a01,
        -k1,
        (f0 - k1 * a01),
        (f0 * a01 + C)
    ]

    roots = np.roots(coeffs)
    roots = roots[np.isreal(roots)].real
    roots = roots[np.abs(roots + a01) > 1e-6]

    return roots

# ============================================================
# 5. PRE-FILTRO RÁPIDO
# ============================================================
def quick_reject(params, samples=30):
    """
    Descarta rápidamente casos donde aparecen 0 puntos fijos.
    """
    k1_star, psub_star = compute_critical(params)

    for _ in range(samples):
        k = np.random.uniform(k1_star*0.7, k1_star*1.3)
        p = np.random.uniform(psub_star*0.7, psub_star*1.3)

        if len(get_real_roots(params, k, p)) == 0:
            return True

    return False

# ============================================================
# 6. ANÁLISIS COMPLETO EN GRILLA
# ============================================================
def analyze_case(params, grid_N=150):
    """
    Construye mapa Z(k1, psub) con número de puntos fijos.
    Descarta si aparece algún punto con 0 soluciones.
    """
    k1_star, psub_star = compute_critical(params)

    k1_vals = np.linspace(k1_star*0.85, k1_star*1.1, grid_N)
    p_vals = np.linspace(psub_star*0.9994, psub_star*1.0006, grid_N)
    # Filtro biológico: rango de presión
    if (p_vals.min() < PSUB_MIN) or (p_vals.max() > PSUB_MAX):
        return None

    Z = np.zeros((grid_N, grid_N))

    for i, k in enumerate(k1_vals):
        for j, p in enumerate(p_vals):

            roots = get_real_roots(params, k, p)

            if len(roots) == 0:
                return None

            Z[i, j] = len(roots)

    return Z, k1_vals, p_vals

# ============================================================
# 7. GUARDADO DE RESULTADOS
# ============================================================
def save_case(case_id, params, Z, k1_vals, p_vals, curves):
    """
    Guarda:
    - imagen del mapa con las curvas de bifurcación superpuestas
    - metadata en JSONL
    """
    timestamp = time.time()

    # Nueva métrica: fracción con 4 puntos fijos
    frac_4 = float(np.sum(Z == 4) / Z.size)

    img_path = os.path.join(IMG_DIR, f"case_{case_id}.png")

    plt.figure(figsize=(8,6))
    
    # Mapa de calor de los puntos fijos
    plt.imshow(
        Z,
        extent=[p_vals[0], p_vals[-1], k1_vals[0], k1_vals[-1]],
        origin='lower',
        aspect='auto'
    )
    plt.colorbar(label="Número de puntos fijos")
    
    # Graficar curvas de bifurcación
    sn = curves["Saddle-Node"]
    hopf = curves["Hopf"]
    
    plt.plot(sn["psub"], sn["kappa1"], color=sn["color"], linestyle=sn["linestyle"], lw=2, label="Saddle-Node")
    plt.plot(hopf["psub"], hopf["kappa1"], color=hopf["color"], linestyle=hopf["linestyle"], lw=2, label="Hopf")
    
    # Restringir la ventana gráfica estrictamente al bounding box para evitar zoom out
    plt.xlim(p_vals[0], p_vals[-1])
    plt.ylim(k1_vals[0], k1_vals[-1])
    
    plt.legend(loc="best", fontsize="small")
    plt.xlabel("psub")
    plt.ylabel("k1")
    plt.title(f"Case {case_id} | frac(4)={frac_4:.2f}")
    plt.savefig(img_path, dpi=120)
    plt.close()

    record = {
        "case_id": case_id,
        "timestamp": timestamp,
        "params": params,
        "k1_range": [float(k1_vals[0]), float(k1_vals[-1])],
        "psub_range": [float(p_vals[0]), float(p_vals[-1])],
        "fraction_4_fixed_points": frac_4,
        "image_path": img_path
    }

    with open(JSON_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

# ============================================================
# 8. LOOP PRINCIPAL
# ============================================================
if __name__ == "__main__":
    max_iter = 50000
    case_id = 51

    for i in range(max_iter):

        print(f"Iteración {i+1} | casos encontrados: {case_id}")

        params = sample_params()

        if quick_reject(params):
            continue

        print("  → pasa pre-check")

        result = analyze_case(params)

        if result is None:
            continue

        Z, k1_vals, p_vals = result
        
        # --- NUEVA LÓGICA: Cálculo y filtrado por Curvas de Bifurcación ---
        # Ampliamos generosamente el rango de 'x' para asegurar interceptar las curvas
        curves = SyrinxModel.get_physical_bifurcation_curves(params, x_min=0.001, x_max=0.5, n_points=20000)
        
        p_min, p_max = p_vals.min(), p_vals.max()
        k1_min, k1_max = k1_vals.min(), k1_vals.max()
        
        sn_p, sn_k = curves["Saddle-Node"]["psub"], curves["Saddle-Node"]["kappa1"]
        hopf_p, hopf_k = curves["Hopf"]["psub"], curves["Hopf"]["kappa1"]
        
        # Comprobar intersección topológica: ¿Existe algún punto de la curva dentro de la grilla?
        sn_in_box = np.any((sn_p >= p_min) & (sn_p <= p_max) & (sn_k >= k1_min) & (sn_k <= k1_max))
        hopf_in_box = np.any((hopf_p >= p_min) & (hopf_p <= p_max) & (hopf_k >= k1_min) & (hopf_k <= k1_max))
        
        if not (sn_in_box and hopf_in_box):
            print("  ❌ descartado: falta curva SN o Hopf en el rango de análisis")
            continue

        print("  ✅ caso válido (biológicamente plausible y con ambas bifurcaciones)")

        save_case(case_id, params, Z, k1_vals, p_vals, curves)

        case_id += 1