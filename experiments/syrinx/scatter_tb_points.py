#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualizador Interactivo de Random Search (Versión Dimensional).
Genera un scatter plot de los puntos Takens-Bogdanov encontrados (psub vs kappa1)
con funcionalidad de hover para identificar rápidamente el ID del caso.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import mplcursors

# =============================================================================
# CARGA DE DATOS
# =============================================================================
# Ruta al JSON de resultados
json_path = Path("tb_random_search") / "hits.json"

try:
    with open(json_path, "r", encoding="utf-8") as f:
        hits = json.load(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en {json_path}")
    exit()

if not hits:
    print("El archivo hits.json está vacío. No hay casos con TB.")
    exit()

# =============================================================================
# EXTRACCIÓN DIMENSIONAL
# =============================================================================
kappa1_vals = []
psub_vals = []
ids = []

for case in hits:
    tb = case.get("tb", {})
    
    # Buscamos directamente las llaves físicas generadas por el nuevo pipeline
    if "kappa1" in tb and "psub" in tb:
        kappa1_vals.append(tb["kappa1"])
        psub_vals.append(tb["psub"])
        ids.append(case["id"])

if not ids:
    print("No se encontraron puntos válidos con 'kappa1' y 'psub' en el JSON.")
    exit()

# =============================================================================
# RENDERIZADO DEL GRÁFICO
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 6))

# Invertimos el orden original para mantener psub en X y kappa1 en Y
scatter = ax.scatter(psub_vals, kappa1_vals, s=40, alpha=0.7, color="tab:purple", edgecolors="white", linewidth=0.5)

ax.set_xlabel(r"Presión Subglótica, $p_{sub}$ (din/cm$^2$)", fontsize=12)
ax.set_ylabel(r"Rigidez Lineal, $\kappa_1$ (din/cm)", fontsize=12)
ax.set_title("Exploración Monte Carlo: Puntos Takens-Bogdanov", fontsize=14)
ax.set_yscale("log")
# psub suele tener rangos amplios (ej. 5000 a 30000), una escala lineal o logarítmica
# funciona bien dependiendo de la densidad. Lo dejamos lineal por defecto para apreciar 
# mejor la distribución física.
ax.grid(True, alpha=0.3, linestyle="--")

# =============================================================================
# HOVER INTERACTIVO
# =============================================================================
cursor = mplcursors.cursor(scatter, hover=True)

@cursor.connect("add")
def on_add(sel):
    i = sel.index
    # Formateo limpio del tooltip
    sel.annotation.set_text(
        f"ID: {ids[i]}\n"
        f"psub:   {psub_vals[i]:.1f}\n"
        f"kappa1: {kappa1_vals[i]:.3f}"
    )
    # Estética del cuadro de hover
    sel.annotation.get_bbox_patch().set(boxstyle="round,pad=0.5", fc="white", alpha=0.9, ec="gray")

plt.tight_layout()
plt.show()