# core/config.py
import os

# 1. Definir la Raíz del Proyecto (Un nivel arriba de 'core/')
# __file__ es la ruta de este archivo config.py
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CORE_DIR)

# 2. Definir Directorios Principales
# Cambia "output" por el nombre que quieras usar por defecto
DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Asegurar que la carpeta base exista
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# 3. Rutas a Archivos Específicos (Constantes)
HDF5_FILE = os.path.join(DATA_OUTPUT_DIR, "trajectory_data.hdf5")
METADATA_FILE = os.path.join(DATA_OUTPUT_DIR, "grid_metadata.json")

# Carpetas de Resultados SINDy
OPT_RESULT_DIR = os.path.join(DATA_OUTPUT_DIR, "optimization_results")
TOP_MODELS_DIR = os.path.join(OPT_DIR, "top_models")

# Archivos de Modelos (Paths por defecto/Globales)
BEST_MODEL_PATH = os.path.join(OPT_RESULT_DIR, "sindy_model.joblib")
TRAINING_PARAMS_PATH = os.path.join(OPT_RESULT_DIR, "sindy_training_params.json")
SINDY_SIM_HDF5 = os.path.join(DATA_OUTPUT_DIR, "sindy_simulations.hdf5")

# Configuraciones Numéricas Globales (Opcional)
DEFAULT_BATCH_SIZE = 5