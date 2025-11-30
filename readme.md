# SINDy-Bifurcation: Parametric Dynamical Systems Discovery

This project implements a high-performance, modular pipeline for discovering parameter-dependent nonlinear differential equations using **SINDy** (Sparse Identification of Nonlinear Dynamics).

It is designed to study complex bifurcations (such as the Takens-Bogdanov bifurcation) by generating massive synthetic datasets, training symbolic AI models, and validating results through comparative parallel simulations.

## 🚀 Key Features

* **Modular Architecture:** Strict separation between the physical system definition (`systems/`) and the numerical machinery (`core/`).
* **High Performance:** Numerical integrators (RK4) and vector field functions compiled at runtime using **Numba JIT**.
* **Efficient Parallelism:** Utilizes `ProcessPoolExecutor` for computation (CPU) and `ThreadPoolExecutor` for disk writing (I/O), enabling massive simulations without memory saturation.
* **Streaming HDF5:** Data is written directly to disk in compressed HDF5 format, supporting gigabytes of trajectories.
* **Dimension Agnostic:** Supports systems with $N$ state variables and $M$ parameters without changing the core codebase.
* **Advanced Optimization:** Dedicated scripts for hyperparameter search (Grid Search) and fine-tuning (Hill Climbing).
* **Interactive Visualization:** Tools to explore bifurcation diagrams and visually compare the "Ground Truth" vs. the discovered model.

## 📂 Project Structure

```text
.
├── core/                   # Numerical machinery and utilities (Agnostic)
│   ├── __init__.py
│   ├── integrators.py      # Generic N-dimensional RK4 integrator (Numba)
│   ├── io.py               # HDF5 management and parameter keys
│   └── utils.py            # Parameter grid generation
│
├── systems/                # Differential Equation Definitions
│   ├── __init__.py
│   ├── base.py             # Abstract base class (Contract)
│   └── takens_bogdanov.py  # Specific implementation (physics of the problem)
│
├── output/                 # OUTPUT folder (created automatically)
│   ├── trajectory_data.hdf5   # Training data (Ground Truth)
│   ├── sindy_model.joblib     # Trained model saved
│   ├── sindy_simulations.hdf5 # Validation data (Simulated by SINDy)
│   ├── optimization_results/  # Hyperparameter search results
│   └── *.json                 # Metadata and logs
│
├── scripts/                # Executable scripts
│   ├── run_precompute.py       # 1. Massive data generation
│   ├── run_interactive.py      # 2. Visual data exploration
│   ├── run_discovery.py        # 3. SINDy model training (Single Run)
│   ├── run_optimization.py     # 4. Best model search (Grid Search)
│   ├── run_fine_tuning.py      # 5. Local refinement (Hill Climbing)
│   ├── run_validation.py       # 6. Learned model simulation
│   └── run_comparison.py       # 7. Final comparison (GT vs SINDy)
└── README.md
```

## 🛠️ Requirements

The project requires Python 3.8+ and the following libraries:

```bash
pip install requirements.txt
```

## ⚡ Workflow (Quick Start)

### 1. Define the System
The default system is **Takens-Bogdanov**. To change it or add a new one (e.g., Van der Pol), create a file in `systems/` inheriting from `BaseSystem` and update the import in the `scripts/run_*.py` scripts:
```python
from systems.takens_bogdanov import TakensBogdanov as System
```

### 2. Generate Data (Precompute)
Simulate the real system across a parameter grid.
```bash
python scripts/run_precompute.py
```
* **Output:** `output/trajectory_data.hdf5` (Real trajectories) and `output/grid_metadata.json`.

### 3. Explore Data (Optional)
Open an interactive viewer to see the bifurcation map and generated trajectories.
```bash
python scripts/run_interactive.py
```

### 4. Train Model (Discovery)
Use PySINDy to find the governing equations, treating parameters as variables.
```bash
python scripts/run_discovery.py
```
* **Output:** `output/sindy_model.joblib` (The AI model) and `output/sindy_training_params.json`.

### 5. Validate Model
Simulate new trajectories using **only** the equations discovered by SINDy (reconstructed in Numba).
```bash
python scripts/run_validation.py
```
* **Output:** `output/sindy_simulations.hdf5` (Trajectories simulated by the AI).

### 6. Compare Results
Display a side-by-side graphical interface: Reality vs. SINDy Model.
```bash
python scripts/run_comparison.py
```

## 🧪 Advanced Optimization

If you wish to find the best possible model instead of training just one:

1.  **Run Optimization:**
    Search for the best hyperparameters and data combinations (Top 5).
    ```bash
    python scripts/run_optimization.py
    ```
    *Output:* `output/optimization_results/top_models/`

2.  **Refine the Champion:**
    Take the best found model and try to improve it by perturbing data or parameters.
    ```bash
    python scripts/run_fine_tuning.py
    ```

## 📝 Notes on Numba
The first time a script is executed, Numba will compile the functions (JIT). This may take a few seconds (warm-up). Subsequent executions will be extremely fast thanks to caching.

---

## 🧪 Detailed Step-by-Step Guide

### Step 0: Environment Preparation
Verify that the folder structure is correct:
1. `scripts/` folder: contains all `run_*.py` files.
2. `core/` folder: `__init__.py`, `integrators.py`, `io.py`, `utils.py`.
3. `systems/` folder: `__init__.py`, `base.py`, `takens_bogdanov.py`.

*Note: If `__init__.py` files are missing, create them empty to enable modules.*

### Step 1: Massive Data Generation (Ground Truth)
Simulates the real differential equations on a parameter grid.
* **Command:** `python scripts/run_precompute.py`
* **What it does:** Reads the configuration from `systems/takens_bogdanov.py`, generates a parameter grid (mu1, mu2), executes thousands of trajectories in parallel, and saves everything to a compressed HDF5 file.

### Step 2: Visual Exploration (Sanity Check)
Allows visual verification that the generated data makes sense.
* **Command:** `python scripts/run_interactive.py`
* **What it does:** Opens an interactive viewer with a heatmap of the number of fixed points. Clicking on the map displays the corresponding phase portrait.

### Step 3: Training (SINDy)
Trains the symbolic model to discover the governing equations.
* **Command:** `python scripts/run_discovery.py`
* **What it does:** Loads a sample of trajectories, fits a SINDy model to estimate derivatives, and saves the serialized model.

### Step 4: Numerical Validation
Evaluates the discovered model by simulating new trajectories.
* **Command:** `python scripts/run_validation.py`
* **What it does:** Reconstructs the discovered equations, compiles them with Numba, and simulates trajectories for parameters *not used* in training.

### Step 5: Final Comparison
Visually compares the real vs. learned dynamics.
* **Command:** `python scripts/run_comparison.py`
* **What it does:** Opens an interface with three panels: Coverage map, Real dynamics (Ground Truth), and Learned dynamics (SINDy).
* **Success Criteria:** If the model is good, the center and right panels should be practically indistinguishable.

---

## Contribution
To add a new dynamical system (e.g., Van der Pol):

1. Copy `systems/takens_bogdanov.py` → `systems/van_der_pol.py`.
2. Modify the class to inherit from `BaseSystem`.
3. Define:
   - `state_names`
   - `param_names`
   - The JIT function `ode_func`
   - `get_true_coefficients` (for validation)
4. Update the `scripts/run_*.py` scripts by replacing the import with:
   ```python
   from systems.van_der_pol import VanDerPol as System
   ```