# core/integrators.py
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def rk4_general(ode_func, y0, t_start, t_end, dt, params):
    """
    Integrador Runge-Kutta 4 de paso fijo (Genérico para N-Dimensiones).
    
    Args:
        ode_func : Función JITeada con firma f(t, y, params) -> dy/dt.
        y0       : Array inicial (float32) de shape (state_dim,).
        t_start  : Tiempo inicial (float32).
        t_end    : Tiempo final (float32).
        dt       : Paso de tiempo (float32).
        params   : Array de parámetros (float32) de shape (param_dim,).
        
    Returns:
        sol      : Array (state_dim, n_steps) con la trayectoria.
    """
    # Calcular número de pasos
    n_steps = int(np.round((t_end - t_start) / dt)) + 1
    
    # Obtener dimensión del estado desde y0
    dim = y0.shape[0]
    
    # Inicializar array de solución (float32 para eficiencia en HDF5)
    sol = np.zeros((dim, n_steps), dtype=np.float32)
    sol[:, 0] = y0.astype(np.float32)

    y_curr = sol[:, 0].copy()
    t_curr = t_start

    # Constantes RK4 pre-casteadas
    f_half = np.float32(0.5)
    f_two = np.float32(2.0)
    f_six = np.float32(6.0)

    for i in range(n_steps - 1):
        # k1, k2, k3, k4 serán arrays de shape (dim,)
        k1 = ode_func(t_curr, y_curr, params)
        
        k2 = ode_func(t_curr + f_half * dt, 
                      y_curr + f_half * dt * k1, 
                      params)
        
        k3 = ode_func(t_curr + f_half * dt, 
                      y_curr + f_half * dt * k2, 
                      params)
        
        k4 = ode_func(t_curr + dt, 
                      y_curr + dt * k3, 
                      params)
        
        # Actualización vectorial
        y_next = y_curr + (k1 + f_two * k2 + f_two * k3 + k4) * (dt / f_six)
        
        sol[:, i + 1] = y_next
        y_curr = y_next
        t_curr = t_curr + dt

    return sol