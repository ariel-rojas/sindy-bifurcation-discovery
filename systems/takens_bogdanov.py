import numpy as np
from numba import jit, prange
from scipy.optimize import brentq
from .base import BaseSystem


@jit(nopython=True, cache=True)
def _tb_ode_func(t, state_arr, param_arr):
    """
    Núcleo de las ecuaciones diferenciales ordinarias (ODE).
    
    Ecuaciones:
        x' = y
        y' = -mu1 - mu2*x + x^2 - x^3 - (x^2 + x)*y
    """
    x, y = state_arr[0], state_arr[1]
    mu1, mu2 = param_arr[0], param_arr[1]
    
    x_dot = y
    y_dot = -mu1 - mu2 * x + x**2 - x**3 - (x**2 + x) * y
    
    return np.array([x_dot, y_dot], dtype=np.float32)

@jit(nopython=True, parallel=True, cache=True)
def _tb_vf_func(x_vals, y_vals, param_arr):
    """
    Calcula el campo vectorial de forma paralela sobre una malla.
    """
    nx, ny = len(x_vals), len(y_vals)
    U = np.zeros((ny, nx), dtype=np.float32)
    V = np.zeros((ny, nx), dtype=np.float32)
    
    mu1, mu2 = param_arr[0], param_arr[1]
    
    for i in prange(ny):
        for j in range(nx):
            x, y = x_vals[j], y_vals[i]
            U[i, j] = y
            V[i, j] = -mu1 - mu2 * x + x**2 - x**3 - (x**2 + x) * y
            
    return U, V

@jit(nopython=True, cache=True)
def _tb_build_features(x, y, param_arr):
    """
    Constructor manual de librerías polinómicas de grado 3.
    Incluye interacciones entre variables de estado y parámetros.
    """
    x, y = np.float32(x), np.float32(y)
    mu1, mu2 = np.float32(param_arr[0]), np.float32(param_arr[1])
    
    # Pre-cálculos de potencias
    x2, x3 = x**2, x**3
    y2, y3 = y**2, y**3
    m1_2, m1_3 = mu1**2, mu1**3
    m2_2, m2_3 = mu2**2, mu2**3
    
    return np.array([
        1.0,                                            # Bias
        x, y, mu1, mu2,                                 # Grado 1
        x2, x*y, x*mu1, x*mu2, y2, y*mu1, y*mu2,        # Grado 2 (Mezclas)
        m1_2, mu1*mu2, m2_2,                            # Grado 2 (Params)
        x3, x2*y, x2*mu1, x2*mu2, x*y2, x*y*mu1,        # Grado 3
        x*y*mu2, x*m1_2, x*mu1*mu2, x*m2_2,
        y3, y2*mu1, y2*mu2, y*m1_2, y*mu1*mu2, y*m2_2,
        m1_3, m1_2*mu2, mu1*m2_2, m2_3
    ], dtype=np.float32)


@jit(nopython=True, cache=True)
def _tb_homoclinic_residual_kernel(mu1, mu2, x_a, x_saddle, dt, t_max):
    """
    Kernel Numba para el residuo de homoclina usando shooting desde el punto medio.
    """
    # Condición inicial: Punto medio entre el foco y la silla.
    # Garantiza estar en el régimen rápido, lejos de la ralentización de los puntos fijos.
    x = (x_a + x_saddle) * 3 / 4
    y = 0.0

    n_steps = int(t_max / dt)
    max_x = x

    for _ in range(n_steps):
        # --- Paso RK4 ---
        k1x = y
        k1y = -mu1 - mu2*x + x*x - x*x*x - (x*x + x)*y

        xa = x + 0.5 * dt * k1x
        ya = y + 0.5 * dt * k1y
        k2x = ya
        k2y = -mu1 - mu2*xa + xa*xa - xa*xa*xa - (xa*xa + xa)*ya

        xb = x + 0.5 * dt * k2x
        yb = y + 0.5 * dt * k2y
        k3x = yb
        k3y = -mu1 - mu2*xb + xb*xb - xb*xb*xb - (xb*xb + xb)*yb

        xc = x + dt * k3x
        yc = y + dt * k3y
        k4x = yc
        k4y = -mu1 - mu2*xc + xc*xc - xc*xc*xc - (xc*xc + xc)*yc

        x = x + (dt / 6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x)
        y = y + (dt / 6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y)

        if x > max_x:
            max_x = x

        # --- Terminación temprana ---
        if max_x > x_saddle:
            return max_x - x_saddle

    return max_x - x_saddle


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class TakensBogdanov(BaseSystem):
    """
    Implementación del sistema dinámico Takens-Bogdanov en su forma normal.
    Este sistema es fundamental para estudiar bifurcaciones de codimensión 2.
    """
    
    # --- Metadatos del Sistema ---
    name = "Takens-Bogdanov"
    state_names = ["x", "y"]
    param_names = ["mu1", "mu2"]
    
    # --- Novedad: Regiones de Exploración (Phase 1) ---
    regions = {
        "base": {
            "param_ranges": [(-0.1, 0.25), (-0.1, 0.4)],
            "state_limits": [(-1.0, 1.0), (-0.5, 0.5)]
        },
        "far_z1": {
            "param_ranges": [(-5, -4), (4, 5)],
            "state_limits": [(-3.0, 3.0), (-3.0, 3.0)]
        },
        "far_z2": {
            "param_ranges": [(4, 5), (4, 5)],
            "state_limits": [(-3.0, 3.0), (-3.0, 3.0)]
        },
        "far_z5": {
            "param_ranges": [(0.05, 0.15), (-5, -4)],
            "state_limits": [(-3.0, 3.0), (-3.0, 3.0)]
        }
    }
    
    # Configuración por defecto (Asegura retrocompatibilidad total)
    active_region = "base"
    param_ranges = regions["base"]["param_ranges"]
    state_limits = regions["base"]["state_limits"]
    
    # Diccionario de zonas de comportamiento dinámico
    zone_names = {
        1: "1 Punto Fijo (mu1 < 0)",
        2: "1 Punto Fijo (mu1 >= 0)",
        3: "3 Puntos Fijos (mu1 < 0, mu2 > 0)",
        4: "Zona de Ciclo Límite (Interna a Homoclina)", # Nueva Zona 4
        5: "Zona de Escape (Post-Homoclina)"             # Nueva Zona 5
    }
    
    _homoclinic_cache = None


    @classmethod
    def set_region(cls, region_name):
        """
        Cambia la región activa de exploración, actualizando los límites
        de parámetros y estados correspondientes.
        """
        if region_name not in cls.regions:
            raise ValueError(f"La región '{region_name}' no existe. Opciones disponibles: {list(cls.regions.keys())}")
        
        cls.active_region = region_name
        cls.param_ranges = cls.regions[region_name]["param_ranges"]
        cls.state_limits = cls.regions[region_name]["state_limits"]
        
    # --- Implementación de Métodos Abstractos ---
    @staticmethod
    def get_ode_jit():
        """Retorna la función x' = f(t, x, p) compilada."""
        return _tb_ode_func

    @staticmethod
    def get_vector_field_jit():
        """Retorna la función de campo vectorial optimizada para mallas."""
        return _tb_vf_func

    @staticmethod
    def get_numba_features_func(degree=3):
        """Retorna el generador de términos para regresión simbólica."""
        if degree != 3:
            raise ValueError("Takens-Bogdanov requiere grado 3 para su forma normal.")
        return _tb_build_features

    # --- Análisis de Estabilidad y Clasificación ---

    @classmethod
    def classify_point(cls, param_arr, manual_data=None):
        """
        Clasifica usando fronteras analíticas y, si está disponible, 
        la información de la curva homoclina manual.
        """
        mu1, mu2 = param_arr
        coeffs = [-1.0, 1.0, -mu2, -mu1]
        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)].real
        num_roots = len(real_roots)

        if num_roots == 1:
            return 1 if mu1 < 0 else 2

        if num_roots == 3:
            if mu1 < 0 and mu2 >= 0:
                return 3
            
            # Estamos en la cuña (Candidata a Z4 o Z5)
            # Si tenemos datos manuales o curva detectada, refinamos:
            if manual_data:
                # Lógica para decidir entre 4 y 5 basada en la curva homoclina
                # Por defecto, antes de la detección, devolvemos 4 (o una zona temporal)
                return cls._refine_with_homoclinic(param_arr, manual_data)
            
            return 4 # Zona base antes de subdividir
        return 0
    

    @staticmethod
    def calculate_fixed_points(param_arr):
        """
        Calcula la ubicación de los puntos fijos y sus propiedades de estabilidad.
        
        Returns:
            np.array: Matriz de [x, traza_jacobiano, det_jacobiano].
        """
        mu1, mu2 = param_arr
        coeffs = np.array([-1.0, 1.0, -float(mu2), -float(mu1)])
        roots = np.roots(coeffs)
        
        fixed_x = roots[np.isreal(roots)].real.astype(np.float32)
        if fixed_x.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        results = np.empty((len(fixed_x), 3), dtype=np.float32)
        for i, x0 in enumerate(fixed_x):
            # Jacobiano: [[0, 1], [ -mu2 + 2x - 3x^2 - (2x+1)y, -(x^2 + x) ]]
            # Evaluado en y=0:
            traza = -(x0**2 + x0)
            det = mu2 - 2*x0 + 3*x0**2
            
            results[i, 0] = x0
            results[i, 1] = traza
            results[i, 2] = det
            
        return results

    def get_bifurcation_curves(self, detected_homoclinic=None):
        """
        Retorna curvas analíticas. Si se le pasa la curva detectada, la incluye.
        """
        t = np.linspace(-1.5, 1.5, 400)
        mu1_sn = 2 * t**3 - t**2
        mu2_sn = -3 * t**2 + 2 * t

        mu2_h = np.linspace(0, 0.4, 100)
        mu1_h = np.zeros_like(mu2_h)

        curves = {
            "Saddle-Node": (mu1_sn, mu2_sn, "red", "-"),
            "Hopf": (mu1_h, mu2_h, "cyan", "--"),
        }

        if detected_homoclinic is not None:
            mu1_hc, mu2_hc = detected_homoclinic
            curves["Homoclinic"] = (mu1_hc, mu2_hc, "magenta", "-.")
            
        return curves
    @staticmethod
    def get_true_coefficients():
        """
        Retorna los coeficientes teóricos exactos del sistema para validación (SINDy).
        """
        return [
            {'y': 1.0}, # x'
            {           # y'
                'mu1': -1.0, 
                'x mu2': -1.0, 
                'x^2': 1.0, 
                'x^3': -1.0, 
                'x y': -1.0, 
                'x^2 y': -1.0
            }
        ]