import numpy as np
from numba import jit, prange
from .base import BaseSystem

# ============================================================================
# Funciones JIT del Campo Vectorial (Dimensional)
# ============================================================================

@jit(nopython=True, cache=True)
def _syrinx_physical_ode(t, state_arr, param_arr):
    """
    Ecuación diferencial dimensional para el modelo de la siringe.
    param_arr orden: [m, g1, g2, k1, k2, c, f0, psub, alab, a01, da, tau]
    """
    x, y = state_arr[0], state_arr[1]
    m, g1, g2, k1, k2, c, f0, psub, alab, a01, da, tau = param_arr

    # Apertura instantánea (Denominador)
    denom = a01 + x + tau * y
    # Seguridad numérica para evitar división por cero en la singularidad
    if abs(denom) < 1e-9: 
        denom = 1e-9 if denom >= 0 else -1e-9

    x_dot = y
    y_dot = (1.0 / m) * (
        -(k1 + k2 * x**2) * x 
        -(g1 + g2 * y**2) * y 
        - c * (x**2) * y 
        + f0 
        + alab * psub * ((da + 2.0 * tau * y) / denom)
    )
    return np.array([x_dot, y_dot], dtype=np.float32)

@jit(nopython=True, parallel=True, cache=True)
def _syrinx_physical_vector_field(x_vals, y_vals, param_arr):
    """Calcula la grilla del campo vectorial para visualización."""
    nx, ny = len(x_vals), len(y_vals)
    U, V = np.zeros((ny, nx), dtype=np.float32), np.zeros((ny, nx), dtype=np.float32)
    for i in prange(ny):
        for j in range(nx):
            # t=0.0 ya que el sistema es autónomo
            res = _syrinx_physical_ode(0.0, np.array([x_vals[j], y_vals[i]], dtype=np.float32), param_arr)
            U[i, j], V[i, j] = res[0], res[1]
    return U, V

# ============================================================================
# Clase Principal del Modelo
# ============================================================================

class SyrinxModel(BaseSystem):
    name = "Syrinx Model (Dimensional)"
    state_names = ["x", "y"]
    param_names = ["kappa1", "psub"]
    full_param_names = [
        "m", "gamma1", "gamma2", "kappa1", "kappa2", 
        "c", "f0", "psub", "alab", "a01", "delta_a", "tau"
    ]

    # Valores físicos por defecto (CGS/SI consistentes)
    default_params = {
        "m": 0.00394, "gamma1": 0.0298, "gamma2": 1.0, 
        "kappa1": 0.35, "kappa2": 234.0, "c": 0.552, 
        "f0": 0.0704, "psub": 15000.0, "alab": 0.00068, 
        "a01": 0.163, "delta_a": -0.0011, "tau": 0.00025
    }

    @staticmethod
    def get_ode_jit(): return _syrinx_physical_ode

    @staticmethod
    def get_vector_field_jit(): return _syrinx_physical_vector_field

    @classmethod
    def build_full_param_arr(cls, **kwargs):
        """Mezcla parámetros de control (kappa1, psub) con los estructurales."""
        arr = np.zeros(len(cls.full_param_names), dtype=np.float64)
        for i, name in enumerate(cls.full_param_names):
            # Prioridad: 1. Argumentos de la función, 2. Valores por defecto
            arr[i] = kwargs.get(name, cls.default_params[name])
        return arr

    # ------------------------------------------------------------------
    # ANÁLISIS DE ESTABILIDAD Y LINEALIZACIÓN
    # ------------------------------------------------------------------
    @classmethod
    def linearization_at(cls, x, p):
        """Calcula traza y determinante de la Jacobiana en el punto fijo (x, 0)."""
        m, g1, g2, k1, k2, c, f0, psub, alab, a01, da, tau = p
        
        # J21 = d(y_dot)/dx
        j21 = (1.0/m) * (-(k1 + 3*k2 * x**2) - (alab * psub * da) / (a01 + x)**2)
        # J22 = d(y_dot)/dy
        j22 = (1.0/m) * (-g1 - c * x**2 + (alab * psub * tau * (2*a01 + 2*x - da)) / (a01 + x)**2)
        
        # En el sistema [0, 1; J21, J22]: Tr = J22, Det = -J21
        return float(j22), float(-j21)

    @classmethod
    def calculate_fixed_points(cls, param_arr):
        """Encuentra raíces del polinomio de equilibrio y evalúa estabilidad."""
        m, g1, g2, k1, k2, c, f0, psub, alab, a01, da, tau = param_arr
        A = alab * psub * da
        
        # Coeficientes del polinomio derivado: 
        # -k2*x^4 - (k2*a01)*x^3 - k1*x^2 + (f0 - k1*a01)*x + (f0*a01 + A) = 0
        coeffs = [-k2, -k2*a01, -k1, f0 - k1*a01, f0*a01 + A]
        roots = np.roots(coeffs)
        
        results = []
        for r in roots:
            if abs(r.imag) < 1e-10:
                x_eq = r.real
                tr, det = cls.linearization_at(x_eq, param_arr)
                results.append([x_eq, 0.0, tr, det])
        return np.array(results)

    # ------------------------------------------------------------------
    # SOLVERS INVERSOS (Curvas de Bifurcación)
    # ------------------------------------------------------------------
    @classmethod
    def saddle_node_params(cls, x, p_struct):
        """Resuelve psub y k1 para det(J)=0 dado un x de equilibrio."""
        m, g1, g2, _, k2, c, f0, _, alab, a01, da, tau = p_struct
        
        # psub obtenido de igualar det(J)=0 con la ec. de equilibrio
        num_p = -(2*k2*x**3 + f0) * (a01 + x)**2
        den_p = alab * da * (a01 + 2*x)
        psub_sn = num_p / den_p
        
        # k1 obtenido de det(J)=0
        k1_sn = (-4*k2*x**3 - 3*k2*a01*x**2 + f0) / (a01 + 2*x)
        return psub_sn, k1_sn

    @classmethod
    def hopf_params(cls, x, p_struct):
        """Resuelve psub y k1 para tr(J)=0 dado un x de equilibrio."""
        m, g1, g2, _, k2, c, f0, _, alab, a01, da, tau = p_struct
        
        num_p = (g1 + c*x**2) * (a01 + x)**2
        den_p = alab * tau * (2*a01 + 2*x - da)
        psub_h = num_p / den_p
        
        k1_h = -k2*x**2 + f0/x + (psub_h * alab * da) / (x * (a01 + x))
        return psub_h, k1_h

    # ------------------------------------------------------------------
    # GENERACIÓN DE DATOS PARA EL VISOR
    # ------------------------------------------------------------------
    @classmethod
    def get_physical_bifurcation_curves(cls, physical_params, x_min=-0.01, x_max=0.01, n_points=5000):
        """Genera trayectorias (psub, kappa1) para SN y Hopf evitando duplicación de argumentos."""
        x_vals = np.linspace(x_min, x_max, n_points)
        
        # 1. Creamos una copia base de los parámetros estructurales (sin los de control)
        # para evitar el error de "multiple values for keyword argument"
        base_params = {k: v for k, v in physical_params.items() if k not in ['kappa1', 'psub']}
        p_struct = cls.build_full_param_arr(**physical_params)
        
        sn_p, sn_k, h_p, h_k = [], [], [], []
        
        for x in x_vals:
            # Saddle-Node
            ps, k1 = cls.saddle_node_params(x, p_struct)
            if ps > 0 and k1 > 0:
                sn_p.append(ps); sn_k.append(k1)
            
            # Hopf
            ps_h, k1_h = cls.hopf_params(x, p_struct)
            if ps_h > 0 and k1_h > 0:
                # 2. Aquí combinamos manualmente para evitar el error de duplicados
                merged_h = base_params.copy()
                merged_h.update({'kappa1': k1_h, 'psub': ps_h})
                p_eval_h = cls.build_full_param_arr(**merged_h)
                
                _, det = cls.linearization_at(x, p_eval_h)
                if det > 0:
                    h_p.append(ps_h); h_k.append(k1_h)
                    
        return {
            "Saddle-Node": {"psub": np.array(sn_p), "kappa1": np.array(sn_k), "color": "green", "linestyle": "--"},
            "Hopf": {"psub": np.array(h_p), "kappa1": np.array(h_k), "color": "red", "linestyle": "-"}
        }

    @classmethod
    def find_takens_bogdanov_points(cls, physical_params, x_range=(0.001, 0.5)):
        """Encuentra intersecciones donde SN y Hopf coinciden (TB)."""
        from scipy.optimize import brentq
        p_struct = cls.build_full_param_arr(**physical_params)

        def objective(x):
            ps_sn, _ = cls.saddle_node_params(x, p_struct)
            ps_h, _ = cls.hopf_params(x, p_struct)
            return ps_sn - ps_h

        try:
            x_tb = brentq(objective, x_range[0], x_range[1])
            ps_tb, k1_tb = cls.saddle_node_params(x_tb, p_struct)
            return [{"x_star": x_tb, "k1": k1_tb, "P": ps_tb}]
        except:
            return []