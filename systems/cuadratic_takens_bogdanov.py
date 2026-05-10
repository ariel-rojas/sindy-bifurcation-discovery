import numpy as np
from numba import jit, prange
from .base import BaseSystem # Asumo que tienes esta clase base definida en tu proyecto

# =============================================================================
# FUNCIONES JIT NIVEL MÓDULO (GLOBALES)
# =============================================================================

@jit(nopython=True, cache=True)
def _bt_2jet_ode_func(t, state_arr, param_arr):
    """
    Función ODE pura para la Bifurcación de Bogdanov-Takens.
    Implementa el despliegue universal exacto del 2-jet:
    x' = y
    y' = mu1 + mu2*y + x^2 + x*y
    """
    x = state_arr[0]
    y = state_arr[1]
    
    mu1 = param_arr[0]
    mu2 = param_arr[1]
    
    x_dot = y
    y_dot = mu1 + mu2 * y + x * x + x * y
    
    return np.array([x_dot, y_dot], dtype=np.float32)

@jit(nopython=True, parallel=True, cache=True)
def _bt_2jet_vf_func(x_vals, y_vals, param_arr):
    """Función de Campo Vectorial paralela para el 2-jet de Bogdanov-Takens."""
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
            V[i, j] = mu1 + mu2 * y + x * x + x * y
            
    return U, V

@jit(nopython=True, cache=True)
def _bt_build_features(x, y, param_arr):
    """
    Constructor de features polinómicas (Grado 3).
    Mantenemos el grado 3 para evaluar la capacidad de la regresión dispersa
    de descartar correctamente los términos de orden superior.
    """
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

class CuadraticTakensBogdanov(BaseSystem):
    """
    Implementación del sistema Bogdanov-Takens usando el despliegue universal 
    finitamente determinado por su 2-jet.
    """
    
    # --- 1. Metadatos ---
    name = "Cuadratic Takens-Bogdanov"
    state_names = ["x", "y"]
    param_names = ["mu1", "mu2"]
    
    # Rangos optimizados para capturar la ventana estrecha de la homoclínica
    # mu1 negativo captura los puntos fijos, mu2 positivo captura la Hopf
    param_ranges = [(-0.5, 0.2), (-0.1, 0.6)] 
    state_limits = [(-1.5, 1.5), (-1.0, 1.0)]
    
    # --- Metadatos de Zonificación (Topología Global) ---
    zone_names = {
        1: "Región I: Sin equilibrios (mu1 > 0)",
        2: "Región II: Silla y Fuente (mu1 <= 0, fuera de Hopf)",
        3: "Región IIIb: Biestabilidad con Ciclo Límite (Hopf superada)",
        4: "Región IIIa: Silla y Sumidero (Catástrofe Homoclínica cruzada)"
    }

    # --- 2. Métodos que devuelven las funciones globales ---

    @staticmethod
    def get_ode_jit():
        return _bt_2jet_ode_func

    @staticmethod
    def get_vector_field_jit():
        return _bt_2jet_vf_func

    @staticmethod
    def get_numba_features_func(degree=3):
        if degree != 3:
             raise ValueError("Por ahora solo está implementado grado 3 manual.")
        return _bt_build_features

    # --- 3. Utilidades Específicas ---

    @staticmethod
    def classify_point(param_arr):
        """
        Clasifica rigurosamente el espacio de parámetros analizando 
        la estabilidad lineal y las perturbaciones globales de Melnikov.
        """
        mu1, mu2 = param_arr[0], param_arr[1]
        
        # Región I: Silla-Nodo no ha ocurrido (no hay raíces reales)
        if mu1 > 0:
            return 1 
            
        # Fronteras analíticas calculadas en el despliegue
        hopf_threshold = np.sqrt(-mu1)
        homoclinic_threshold = np.sqrt(-mu1 * 25.0 / 49.0)
        
        # Nos enfocamos en la mitad superior del plano para las bifurcaciones principales
        if mu2 > hopf_threshold:
            return 2 # Región II: El nodo es repulsivo, no hay ciclo límite.
            
        elif mu2 > homoclinic_threshold:
            return 3 # Región IIIb: El nodo es atractivo, el ciclo límite sobrevive.
            
        else:
            return 4 # Región IIIa: El ciclo límite chocó con la silla y se destruyó.

    @staticmethod
    def calculate_fixed_points(param_arr):
        """
        Calcula los puntos fijos analíticos y sus propiedades de linealización.
        x^2 + mu1 = 0 => x = +/- sqrt(-mu1)
        """
        mu1 = param_arr[0]
        mu2 = param_arr[1]
        
        if mu1 > 0:
            return np.empty((0, 3), dtype=np.float32)
            
        x_plus = np.sqrt(-mu1)
        x_minus = -np.sqrt(-mu1)
        
        fixed_x = np.array([x_plus, x_minus], dtype=np.float32)
        results = np.empty((2, 3), dtype=np.float32)
        
        for i, x0 in enumerate(fixed_x):
            # Matriz Jacobiana en y=0: J = [[0, 1], [2x, mu2 + x]]
            traza = mu2 + x0
            det = -2.0 * x0
            
            results[i, 0] = x0
            results[i, 1] = traza
            results[i, 2] = det
            
        return results

    def get_bifurcation_curves(self):
        """
        Devuelve las tres fronteras analíticas exactas para graficar 
        el mapa de codimensión 2.
        """
        # Curva Silla-Nodo: mu1 = 0
        mu2_sn = np.linspace(*self.param_ranges[1], 100)
        mu1_sn = np.zeros_like(mu2_sn)
        
        # Rango para las parábolas (mu2 > 0)
        mu2_parabolas = np.linspace(0, self.param_ranges[1][1], 200)
        
        # Curva de Hopf: mu1 = -mu2^2
        mu1_h = - (mu2_parabolas ** 2)
        
        # Curva Homoclínica (Aproximación de Melnikov): mu1 = -(49/25)*mu2^2
        mu1_hc = - (49.0 / 25.0) * (mu2_parabolas ** 2)
        
        return {
            "Saddle-Node": (mu1_sn, mu2_sn, "black", "-"),
            "Hopf": (mu1_h, mu2_parabolas, "blue", "--"),
            "Homoclinic": (mu1_hc, mu2_parabolas, "red", "-.")
        }

    @staticmethod
    def get_true_coefficients():
        """
        Coeficientes teóricos exactos para el algoritmo de identificación.
        Ecuaciones reales:
        x' = y
        y' = mu1 + mu2*y + x^2 + x*y
        """
        # Ecuación 0: x'
        eq_x = {'y': 1.0}
        
        # Ecuación 1: y'
        eq_y = {
            'mu1': 1.0,
            'y mu2': 1.0, # PySINDy ordena alfabéticamente o por orden de variables
            'x^2': 1.0,
            'x y': 1.0
        }
        
        return [eq_x, eq_y]