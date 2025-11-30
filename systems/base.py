# systems/base.py
import numpy as np

class BaseSystem:
    """
    Clase base abstracta para definir sistemas dinámicos de N-dimensiones
    con M-parámetros.
    """
    # --- Metadatos (Sobrescribir en la clase hija) ---
    name = "Base System"
    
    # Nombres y dimensiones para validación y ploteo
    state_names = ["x", "y"]     # La longitud define state_dim (N)
    param_names = ["mu1", "mu2"] # La longitud define param_dim (M)
    
    # Configuración por defecto (Rangos para precompute)
    # Formato: [(min, max), (min, max), ...]
    # Orden: Corresponde a param_names
    param_ranges = [(-0.1, 0.1), (-0.1, 0.1)]
    
    # Límites del espacio de fase para simulación/visualización
    # Orden: Corresponde a state_names
    state_limits = [(-1.0, 1.0), (-1.0, 1.0)]

    @property
    def state_dim(self):
        return len(self.state_names)

    @property
    def param_dim(self):
        return len(self.param_names)

    @staticmethod
    def get_ode_jit():
        """
        Debe devolver una función compilada con @jit(nopython=True) con la firma:
        
        def ode_func(t, state_arr, param_arr):
            # state_arr: array float32 de shape (N,)
            # param_arr: array float32 de shape (M,)
            ...
            return d_state_dt (array float32 de shape (N,))
            
        Raises:
            NotImplementedError: Si la subclase no lo implementa.
        """
        raise NotImplementedError("La subclase debe implementar get_ode_jit")

    @staticmethod
    def get_vector_field_jit():
        """
        Debe devolver una función compilada con @jit(nopython=True, parallel=True) 
        para calcular el campo vectorial en una grilla 2D (usada en visualización).
        
        Firma esperada:
        def vf_func(x_vals, y_vals, param_arr):
            ...
            return U, V
        """
        raise NotImplementedError("La subclase debe implementar get_vector_field_jit")
    
    @staticmethod
    def get_true_coefficients():
        """
        Devuelve los valores reales de los coeficientes para calcular el error.
        
        Debe devolver una LISTA de DICCIONARIOS.
        - La lista tiene longitud = state_dim (una entrada por ecuación: x', y', ...).
        - Cada diccionario mapea: "nombre_termino_sindy" -> valor_teorico.
        
        Ejemplo para x' = y:
            [{'y': 1.0}, ...]
        """
        raise NotImplementedError("La subclase debe implementar get_true_coefficients")