# systems/takens_bogdanov.py
import numpy as np
from numba import jit, prange
from .base import BaseSystem

# =============================================================================
# FUNCIONES JIT NIVEL MÓDULO (GLOBALES)
# Al estar aquí afuera, 'multiprocessing' puede encontrarlas y serializarlas sin error.
# =============================================================================

@jit(nopython=True, cache=True)
def _tb_ode_func(t, state_arr, param_arr):
    """Función ODE pura para Takens-Bogdanov."""
    # Desempaquetado eficiente (float32)
    x = state_arr[0]
    y = state_arr[1]
    
    mu1 = param_arr[0]
    mu2 = param_arr[1]
    
    # Ecuaciones originales
    x_dot = y
    y_dot = -mu1 - mu2 * x + x * x - x * x * x - (x * x + x) * y
    
    return np.array([x_dot, y_dot], dtype=np.float32)

@jit(nopython=True, parallel=True, cache=True)
def _tb_vf_func(x_vals, y_vals, param_arr):
    """Función de Campo Vectorial paralela."""
    nx = len(x_vals)
    ny = len(y_vals)
    U = np.zeros((ny, nx), dtype=np.float32)
    V = np.zeros((ny, nx), dtype=np.float32)
    
    mu1 = param_arr[0]
    mu2 = param_arr[1]
    
    for i in prange(ny):
        for j in range(nx):
            x = x_vals[j]
            y = y_vals[i]
            
            x_dot = y
            y_dot = -mu1 - mu2 * x + x * x - x * x * x - (x * x + x) * y
            
            U[i, j] = x_dot
            V[i, j] = y_dot
    return U, V

@jit(nopython=True, cache=True)
def _tb_build_features(x, y, param_arr):
    """Constructor de features polinómicas (Grado 3)."""
    x = np.float32(x)
    y = np.float32(y)
    mu1 = np.float32(param_arr[0])
    mu2 = np.float32(param_arr[1])
    
    # Pre-cálculos
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    m1_2 = mu1 * mu1
    m1_3 = m1_2 * mu1
    m2_2 = mu2 * mu2
    m2_3 = m2_2 * mu2
    
    return np.array([
        1.0,                                # 1
        x, y, mu1, mu2,                     # Grado 1
        x2, x*y, x*mu1, x*mu2, y2, y*mu1, y*mu2, m1_2, mu1*mu2, m2_2, # Grado 2
        x3, x2*y, x2*mu1, x2*mu2, x*y2, x*y*mu1, x*y*mu2, x*m1_2, x*mu1*mu2, x*m2_2,
        y3, y2*mu1, y2*mu2, y*m1_2, y*mu1*mu2, y*m2_2,
        m1_3, m1_2*mu2, mu1*m2_2, m2_3
    ], dtype=np.float32)


# =============================================================================
# CLASE DEL SISTEMA
# =============================================================================

class TakensBogdanov(BaseSystem):
    """
    Implementación del sistema Takens-Bogdanov normal form.
    """
    
    # --- 1. Metadatos ---
    name = "Takens-Bogdanov"
    state_names = ["x", "y"]
    param_names = ["mu1", "mu2"]
    
    # Rangos de parámetros
    param_ranges = [(-0.1, 0.25), (-0.1, 0.4)]
    
    # Límites del espacio de fase
    state_limits = [(-1.0, 1.0), (-0.5, 0.5)]

    # --- 2. Métodos que devuelven las funciones globales ---

    @staticmethod
    def get_ode_jit():
        """Devuelve la función x' = f(t, x, params) compilada."""
        return _tb_ode_func

    @staticmethod
    def get_vector_field_jit():
        """Devuelve la función paralela para calcular campo vectorial U, V."""
        return _tb_vf_func

    @staticmethod
    def get_numba_features_func(degree=3):
        """Devuelve el constructor de features."""
        if degree != 3:
             raise ValueError("Por ahora solo está implementado grado 3 manual para TB.")
        return _tb_build_features

    # --- 3. Utilidades Específicas ---

    @staticmethod
    def calculate_fixed_points(param_arr):
        """
        Calcula los puntos fijos analíticos (raíces).
        """
        mu1 = param_arr[0]
        mu2 = param_arr[1]
        
        # Polinomio: -x^3 + x^2 - mu2*x - mu1 = 0
        coeffs = np.array([-1.0, 1.0, -float(mu2), -float(mu1)], dtype=np.float64)
        roots = np.roots(coeffs)
        
        fixed_x = roots[np.isreal(roots)].real.astype(np.float32)
        
        if fixed_x.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        results = np.empty((fixed_x.shape[0], 3), dtype=np.float32)
        for i, x0 in enumerate(fixed_x):
            traza = -x0 * x0 - x0
            det = mu2 + np.float32(3.0) * x0 * x0 - np.float32(2.0) * x0
            
            results[i, 0] = x0
            results[i, 1] = traza
            results[i, 2] = det
            
        return results

    def get_bifurcation_curves(self):
        """Curvas analíticas para graficar."""
        t = np.linspace(-1.5, 1.5, 400)
        mu1_sn = 2 * t**3 - t**2
        mu2_sn = -3 * t**2 + 2 * t
        
        y_range = self.param_ranges[1]
        mu2_h = np.linspace(0, y_range[1], 100)
        mu1_h = np.zeros_like(mu2_h)
        
        return {
            "Saddle-Node": (mu1_sn, mu2_sn, "red", "-"),
            "Hopf": (mu1_h, mu2_h, "cyan", "--")
        }

    @staticmethod
    def get_true_coefficients():
        """
        Coeficientes teóricos para Takens-Bogdanov normal form:
        x' = y
        y' = -mu1 - mu2*x + x^2 - x^3 - x*y - x^2*y
        
        Nombres basados en feature_names=["x", "y", "mu1", "mu2"] y grado 3.
        PySINDy usa espacios para multiplicaciones por defecto.
        """
        # Ecuación 0: x'
        eq_x = {'y': 1.0}
        
        # Ecuación 1: y'
        # Términos: -1*mu1, -1*x*mu2, +1*x^2, -1*x^3, -1*x*y, -1*x^2*y
        eq_y = {
            'mu1': -1.0,
            'x mu2': -1.0, # Interacción paramétrica
            'x^2': 1.0,
            'x^3': -1.0,
            'x y': -1.0,
            'x^2 y': -1.0
        }
        
        return [eq_x, eq_y]