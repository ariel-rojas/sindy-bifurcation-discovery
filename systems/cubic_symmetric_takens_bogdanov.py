import numpy as np
from numba import jit, prange
from .base import BaseSystem

# =============================================================================
# FUNCIONES JIT NIVEL MÓDULO (GLOBALES)
# =============================================================================

@jit(nopython=True, cache=True)
def _cubic_bt_ode_func(t, state_arr, param_arr):
    """
    Función ODE para el Despliegue de Simetría Cúbica.
    Ecuaciones:
    x' = y
    y' = mu1*x + mu2*y - x^3 - x^2*y
    """
    x = state_arr[0]
    y = state_arr[1]
    
    mu1 = param_arr[0]
    mu2 = param_arr[1]
    
    x_dot = y
    y_dot = mu1 * x + mu2 * y - (x * x * x) - (x * x * y)
    
    return np.array([x_dot, y_dot], dtype=np.float32)

@jit(nopython=True, parallel=True, cache=True)
def _cubic_bt_vf_func(x_vals, y_vals, param_arr):
    """Función de Campo Vectorial paralela para Simetría Cúbica."""
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
            
            U[i, j] = y
            V[i, j] = mu1 * x + mu2 * y - (x * x * x) - (x * x * y)
            
    return U, V

@jit(nopython=True, cache=True)
def _cubic_bt_build_features(x, y, param_arr):
    """
    Constructor de features polinómicas (Grado 3).
    Se mantiene idéntico, ya que el grado 3 es exactamente lo que 
    necesitamos para la simetría cúbica.
    """
    x = np.float32(x)
    y = np.float32(y)
    mu1 = np.float32(param_arr[0])
    mu2 = np.float32(param_arr[1])
    
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    m1_2 = mu1 * mu1
    m1_3 = m1_2 * mu1
    m2_2 = mu2 * mu2
    m2_3 = m2_2 * mu2
    
    return np.array([
        1.0,                                                # 1
        x, y, mu1, mu2,                                     # Grado 1
        x2, x*y, x*mu1, x*mu2, y2, y*mu1, y*mu2, m1_2, mu1*mu2, m2_2, # Grado 2
        x3, x2*y, x2*mu1, x2*mu2, x*y2, x*y*mu1, x*y*mu2, x*m1_2, x*mu1*mu2, x*m2_2,
        y3, y2*mu1, y2*mu2, y*m1_2, y*mu1*mu2, y*m2_2,
        m1_3, m1_2*mu2, mu1*m2_2, m2_3
    ], dtype=np.float32)


# =============================================================================
# CLASE DEL SISTEMA
# =============================================================================

class CubicSymmetricTakensBogdanov(BaseSystem):
    """
    Implementación del despliegue de codimensión 2 con simetría cúbica
    (rotacional pi) de la bifurcación de Takens-Bogdanov.
    (Caso a3 = -1, b3 = -1 del libro de Guckenheimer & Holmes)
    """
    
    # --- 1. Metadatos ---
    name = "Cubic Symmetric Takens-Bogdanov"
    state_names = ["x", "y"]
    param_names = ["mu1", "mu2"]
    
    # Rangos enfocados en el cuadrante positivo donde ocurre la magia global
    param_ranges = [(-0.2, 1.0), (-0.2, 1.0)]
    state_limits = [(-2.0, 2.0), (-1.5, 1.5)]
    
    # --- Metadatos de Zonificación (Topología) ---
    zone_names = {
        1: "1 Punto Fijo (mu1 <= 0)",
        2: "3 Puntos Fijos + 1 Ciclo Gigante (Sobre Hopf)",
        3: "3 Puntos Fijos + 3 Ciclos (2 Pequeños, 1 Gigante)",
        4: "3 Puntos Fijos + 2 Ciclos Gigantes (Ocho Roto)",
        5: "3 Puntos Fijos + Cero Oscilaciones (Debajo de SNPO)"
    }

    # --- 2. Métodos que devuelven las funciones globales ---

    @staticmethod
    def get_ode_jit():
        return _cubic_bt_ode_func

    @staticmethod
    def get_vector_field_jit():
        return _cubic_bt_vf_func

    @staticmethod
    def get_numba_features_func(degree=3):
        if degree != 3:
             raise ValueError("Por ahora solo está implementado grado 3 manual.")
        return _cubic_bt_build_features

    # --- 3. Utilidades Específicas ---

    @staticmethod
    def classify_point(param_arr):
        """
        Clasifica la topología basada en las fronteras analíticas deducidas:
        Hopf (mu2 = mu1), Homoclínica (mu2 = 0.8*mu1), SNPO (mu2 = 0.752*mu1).
        """
        mu1, mu2 = param_arr[0], param_arr[1]
        
        if mu1 <= 0:
            return 1 # Origen es el único punto fijo
            
        # Constantes analíticas
        homoclinic_ratio = 4.0 / 5.0  # 0.8
        snpo_ratio = 0.752            # Mínimo de R(alpha)
        
        if mu2 >= mu1:
            return 2 # Arriba de Hopf: 1 Ciclo Gigante Atractor
        elif mu2 >= homoclinic_ratio * mu1:
            return 3 # Entre Hopf y Homoclínica: Nacen ciclos pequeños
        elif mu2 >= snpo_ratio * mu1:
            return 4 # Entre Homoclínica y Silla-Nodo: 2 Ciclos Gigantes (Atractor y Repulsor)
        else:
            return 5 # Por debajo de todo: Cero ciclos límite, espirales a los focos

    @staticmethod
    def calculate_fixed_points(param_arr):
        """
        Calcula los puntos fijos analíticos y su matriz Jacobiana.
        """
        mu1 = param_arr[0]
        mu2 = param_arr[1]
        
        if mu1 <= 0:
            fixed_x = np.array([0.0], dtype=np.float32)
        else:
            fixed_x = np.array([0.0, np.sqrt(mu1), -np.sqrt(mu1)], dtype=np.float32)
            
        results = np.empty((fixed_x.shape[0], 3), dtype=np.float32)
        
        for i, x0 in enumerate(fixed_x):
            # Matriz Jacobiana: J = [[0, 1], [mu1 - 3*x^2, mu2 - x^2]]
            traza = mu2 - (x0 * x0)
            det = (3.0 * x0 * x0) - mu1
            
            results[i, 0] = x0
            results[i, 1] = traza
            results[i, 2] = det
            
        return results

    def get_bifurcation_curves(self):
        """Curvas analíticas rectas que nacen del origen para mu1 > 0."""
        mu1_vals = np.linspace(0, self.param_ranges[0][1], 200)
        
        mu2_hopf = mu1_vals
        mu2_homoclinic = (4.0 / 5.0) * mu1_vals
        mu2_snpo = 0.752 * mu1_vals
        
        return {
            "Hopf (Bh)": (mu1_vals, mu2_hopf, "blue", "--"),
            "Homoclínica Ocho (Bsc)": (mu1_vals, mu2_homoclinic, "red", "-."),
            "Silla-Nodo Periódica (Bpo)": (mu1_vals, mu2_snpo, "green", ":")
        }

    @staticmethod
    def get_true_coefficients():
        """
        Coeficientes teóricos para el algoritmo PySINDy.
        x' = y
        y' = mu1*x + mu2*y - x^3 - x^2*y
        """
        eq_x = {'y': 1.0}
        
        # PySINDy ordenará los términos interactivos alfabéticamente
        eq_y = {
            'x mu1': 1.0,
            'y mu2': 1.0,
            'x^3': -1.0,
            'x^2 y': -1.0
        }
        
        return [eq_x, eq_y]