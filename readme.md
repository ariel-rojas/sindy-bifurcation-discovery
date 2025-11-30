# SINDy-Bifurcation: Parametric Dynamical Systems Discovery

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research_prototype-orange)
![Scientific Stack](https://img.shields.io/badge/stack-NumPy_|_Numba_|_PySINDy-8A2BE2)

A high-performance, modular pipeline for discovering parameter-dependent nonlinear differential equations using **SINDy** (Sparse Identification of Nonlinear Dynamics).

This project focuses on the data-driven discovery of complex bifurcations, specifically extended forms of the **Takens-Bogdanov** bifurcation. It generates massive synthetic datasets, trains symbolic AI models to recover the governing equations, and validates the results through comparative parallel simulations.

## 🔬 Scientific Context

The core objective is to reconstruct the dynamics of a parameterized nonlinear system from time-series data. The specific system used as Ground Truth in this repository is an unfolded Takens-Bogdanov form with higher-order terms:

$$
\begin{aligned}
\dot{x} &= y \\
\dot{y} &= -\mu_1 - \mu_2 x + x^2 - x^3 - xy - x^2y
\end{aligned}
$$

Where $x, y$ are the state variables and $\mu_1, \mu_2$ are the bifurcation parameters.

**The Challenge:** Standard SINDy assumes constant coefficients. To capture bifurcations, this pipeline treats parameters ($\mu$) as variables. This allows the sparse regression to automatically discover the functional dependency between system dynamics and external control parameters.

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
│   ├── optimization_results/  # Hyperparameter search results & Top Models
│   ├── v1/                    # Versioned discovery runs
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

## 🛠️ Installation

The project requires Python 3.8+. It is recommended to use a virtual environment.

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Workflow (Quick Start)

### 1. Define the System
The default system is **Takens-Bogdanov**. To change it or add a new one (e.g., Van der Pol), create a file in `systems/` inheriting from `BaseSystem` and update the import in the scripts:
```python
# scripts/run_precompute.py
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
> **Visualizing the Phase Space:** Click on the heatmap (bifurcation diagram) to visualize the phase portrait for specific $(\mu_1, \mu_2)$ values. Click on a trajectory line to see time-series details.

![Interactive Viewer](https://via.placeholder.com/800x400?text=Insert+run_interactive+GIF+Here)

### 4. Train Model (Discovery)
Use PySINDy to find the governing equations, treating parameters as variables.
```bash
python scripts/run_discovery.py
```
* **Output:** Creates a versioned folder inside `output/` (e.g., `output/5/13/`) containing `sindy_model.joblib` and `sindy_training_params.json`.

### 5. Validate Model
Simulate new trajectories using **only** the equations discovered by SINDy (reconstructed and compiled via Numba).
```bash
python scripts/run_validation.py
```
* **Output:** `output/.../sindy_simulations.hdf5`.

### 6. Compare Results
Display a side-by-side graphical interface: Reality vs. SINDy Model.
```bash
python scripts/run_comparison.py
```
> **Validation:** Click on the blue points (Validation set) in the coverage map. The Center panel shows Ground Truth, the Right panel shows SINDy's simulation. They should be topologically identical.

![Comparison Viewer](https://via.placeholder.com/800x400?text=Insert+run_comparison+Screenshot+Here)

---

## 📊 Example Results

*Typical output from `run_discovery.py` comparing True vs. Identified coefficients:*

| Term | True Coefficient | Identified Coefficient | Error |
| :--- | :---: | :---: | :---: |
| $\dot{x}: y$ | 1.000 | 0.9998 | 0.02% |
| $\dot{y}: \mu_1$ | -1.000 | -0.9985 | 0.15% |
| $\dot{y}: x \mu_2$ | -1.000 | -1.0012 | 0.12% |
| $\dot{y}: x^2$ | 1.000 | 0.9991 | 0.09% |
| $\dot{y}: x^2 y$ | -1.000 | -0.9950 | 0.50% |

---

## 🧪 Advanced Optimization

If you wish to find the best possible model instead of training just one:

1.  **Run Optimization:**
    Search for the best hyperparameters (Threshold, Ridge Alpha) and data combinations.
    ```bash
    python scripts/run_optimization.py
    ```
    *Output:* Stores the Top 5 best models in `output/optimization_results/top_models/` and updates the historic log.

2.  **Refine the Champion:**
    Take the best found model and perform local optimization (Hill Climbing) on the training data selection.
    ```bash
    python scripts/run_fine_tuning.py
    ```

## 📝 Performance Notes
* **JIT Compilation:** The first time a script is executed, Numba will compile the functions. This implies a warm-up time of 1-3 seconds. Subsequent executions are near C-speed.
* **HDF5 Compression:** Data is stored using `gzip` compression to save disk space while allowing chunked access.

---

## Contribution

To add a new dynamical system (e.g., Van der Pol):

1.  Copy `systems/takens_bogdanov.py` → `systems/van_der_pol.py`.
2.  Modify the class to inherit from `BaseSystem`.
3.  Define:
    * `state_names` and `param_names`
    * The JIT function `ode_func`
    * `get_true_coefficients` (for validation)
4.  Update the `scripts/run_*.py` scripts to import your new class.