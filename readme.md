# SINDy-Bifurcation: Descubrimiento de Sistemas Dinámicos Paramétricos

Este proyecto implementa un pipeline modular y de alto rendimiento para descubrir ecuaciones diferenciales no lineales dependientes de parámetros utilizando **SINDy** (Sparse Identification of Nonlinear Dynamics).

Está diseñado para estudiar bifurcaciones complejas (como Takens-Bogdanov) generando datos sintéticos masivos, entrenando modelos de IA simbólica y validando los resultados mediante simulaciones paralelas comparativas.

## 🚀 Características Principales

* **Arquitectura Modular:** Separación estricta entre la definición física del sistema (`systems/`) y la maquinaria numérica (`core/`).
* **Alto Rendimiento:** Integradores numéricos (RK4) y funciones de campo vectorial compilados en tiempo de ejecución con **Numba JIT**.
* **Paralelismo Eficiente:** Uso de `ProcessPoolExecutor` para cómputo (CPU) y `ThreadPoolExecutor` para escritura en disco (I/O), permitiendo simulaciones masivas sin saturar la memoria.
* **Streaming HDF5:** Los datos se escriben directamente a disco en formato HDF5 comprimido, soportando gigabytes de trayectorias.
* **Agnóstico de la Dimensión:** Soporta sistemas de $N$ variables de estado y $M$ parámetros sin cambiar el código base.
* **Optimización Avanzada:** Scripts dedicados para búsqueda de hiperparámetros (Grid Search) y ajuste fino (Hill Climbing).
* **Visualización Interactiva:** Herramientas para explorar diagramas de bifurcación y comparar visualmente el "Ground Truth" vs. el modelo descubierto.

## 📂 Estructura del Proyecto

```text
.
├── core/                   # Maquinaria numérica y utilidades (Agnóstico)
│   ├── __init__.py
│   ├── integrators.py      # Integrador RK4 genérico N-dimensional (Numba)
│   ├── io.py               # Gestión de HDF5 y claves de parámetros
│   └── utils.py            # Generación de grillas de parámetros
│
├── systems/                # Definiciones de Ecuaciones Diferenciales
│   ├── __init__.py
│   ├── base.py             # Clase base abstracta (Contrato)
│   └── takens_bogdanov.py  # Implementación específica (física del problema)
│
├── output/                 # Carpeta de SALIDA (se crea automáticamente)
│   ├── trajectory_data.hdf5   # Datos de entrenamiento (Ground Truth)
│   ├── sindy_model.joblib     # Modelo entrenado guardado
│   ├── sindy_simulations.hdf5 # Datos de validación (Simulados por SINDy)
│   ├── optimization_results/  # Resultados de búsqueda de hiperparámetros
│   └── *.json                 # Metadatos y logs
│
├── scripts/                # Scripts ejecutables
│   ├── run_precompute.py       # 1. Generación masiva de datos
│   ├── run_interactive.py      # 2. Exploración visual de datos
│   ├── run_discovery.py        # 3. Entrenamiento del modelo SINDy (Single Run)
│   ├── run_optimization.py     # 4. Búsqueda de mejores modelos (Grid Search)
│   ├── run_fine_tuning.py      # 5. Refinamiento local (Hill Climbing)
│   ├── run_validation.py       # 6. Simulación del modelo aprendido
│   └── run_comparison.py       # 7. Comparación final (GT vs SINDy)
└── README.md
```

## 🛠️ Requisitos

El proyecto requiere Python 3.8+ y las siguientes librerías:

```bash
pip install numpy scipy matplotlib h5py numba pysindy tqdm joblib
```

## ⚡ Flujo de Trabajo (Quick Start)

### 1. Definir el Sistema
El sistema por defecto es **Takens-Bogdanov**. Para cambiarlo o añadir uno nuevo (ej. Van der Pol), crea un archivo en `systems/` heredando de `BaseSystem` y actualiza la importación en los scripts `scripts/run_*.py`:
```python
from systems.takens_bogdanov import TakensBogdanov as System
```

### 2. Generar Datos (Precompute)
Simula el sistema real en una grilla de parámetros.
```bash
python scripts/run_precompute.py
```
* **Salida:** `output/trajectory_data.hdf5` (Trayectorias reales) y `output/grid_metadata.json`.

### 3. Explorar Datos (Opcional)
Abre un visor interactivo para ver el mapa de bifurcación y las trayectorias generadas.
```bash
python scripts/run_interactive.py
```

### 4. Entrenar Modelo (Discovery)
Usa PySINDy para encontrar las ecuaciones gobernantes, tratando a los parámetros como variables.
```bash
python scripts/run_discovery.py
```
* **Salida:** `output/sindy_model.joblib` (El modelo IA) y `output/sindy_training_params.json`.

### 5. Validar Modelo
Simula trayectorias nuevas usando **únicamente** las ecuaciones descubiertas por SINDy (reconstruidas en Numba).
```bash
python scripts/run_validation.py
```
* **Salida:** `output/sindy_simulations.hdf5` (Trayectorias simuladas por la IA).

### 6. Comparar Resultados
Muestra una interfaz gráfica lado a lado: Realidad vs. Modelo SINDy.
```bash
python scripts/run_comparison.py
```

## 🧪 Optimización Avanzada

Si deseas encontrar el mejor modelo posible en lugar de entrenar uno solo:

1.  **Ejecuta la Optimización:**
    Busca los mejores hiperparámetros y combinaciones de datos.
    ```bash
    python scripts/run_optimization.py
    ```
    *Salida:* `output/optimization_results/top_models/`


## 📝 Notas sobre Numba
La primera vez que se ejecuta un script, Numba compilará las funciones (JIT). Esto puede tomar unos segundos (warm-up). Las ejecuciones subsiguientes serán extremadamente rápidas gracias al caché.

---

## 🧪 Guía Paso a Paso Detallada

### Paso 0: Preparación del Entorno
Verifica que la estructura de carpetas sea correcta:
1. Carpeta `scripts/`: contiene todos los archivos `run_*.py`.
2. Carpeta `core/`: `__init__.py`, `integrators.py`, `io.py`, `utils.py`.
3. Carpeta `systems/`: `__init__.py`, `base.py`, `takens_bogdanov.py`.

*Nota: Si faltan archivos `__init__.py`, créalos vacíos para habilitar los módulos.*

### Paso 1: Generación de Datos Masivos (Ground Truth)
Simula las ecuaciones diferenciales reales en una grilla de parámetros.
* **Comando:** `python scripts/run_precompute.py`
* **Qué hace:** Lee la configuración de `systems/takens_bogdanov.py`, genera una grilla de parámetros (mu1, mu2), ejecuta miles de trayectorias en paralelo y guarda todo en un HDF5 comprimido.

### Paso 2: Exploración Visual (Sanity Check)
Permite verificar visualmente que los datos generados tengan sentido.
* **Comando:** `python scripts/run_interactive.py`
* **Qué hace:** Abre un visor interactivo con un mapa de calor del número de puntos fijos. Al hacer clic en el mapa, muestra el retrato de fase correspondiente.

### Paso 3: Entrenamiento (SINDy)
Entrena el modelo simbólico para descubrir las ecuaciones gobernantes.
* **Comando:** `python scripts/run_discovery.py`
* **Qué hace:** Carga una muestra de trayectorias, ajusta un modelo SINDy para estimar las derivadas y guarda el modelo serializado.

### Paso 4: Validación Numérica
Evalúa el modelo descubierto simulando trayectorias nuevas.
* **Comando:** `python scripts/run_validation.py`
* **Qué hace:** Reconstruye las ecuaciones descubiertas, las compila con Numba y simula trayectorias para parámetros *no usados* en el entrenamiento.

### Paso 5: Comparación Final
Compara visualmente la dinámica real vs. la aprendida.
* **Comando:** `python scripts/run_comparison.py`
* **Qué hace:** Abre una interfaz con tres paneles: Mapa de cobertura, Dinámica real (Ground Truth) y Dinámica aprendida (SINDy).
* **Criterio de éxito:** Si el modelo es bueno, los paneles central y derecho deben ser prácticamente indistinguibles.

---

## Contribución
Para agregar un nuevo sistema dinámico (ej. Van der Pol):

1. Copia `systems/takens_bogdanov.py` → `systems/van_der_pol.py`.
2. Modifica la clase para heredar de `BaseSystem`.
3. Define:
   - `state_names`
   - `param_names`
   - la función JIT `ode_func`
   - `get_true_coefficients` (para validación)
4. Actualiza los scripts `scripts/run_*.py` reemplazando la importación por:
   ```python
   from systems.van_der_pol import VanDerPol as System
   ```