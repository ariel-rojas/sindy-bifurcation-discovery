# core/metrics.py
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def calculate_velocity_norm(trajectory, dt):
    """Calcula ||v|| = sqrt(x'^2 + y'^2)."""
    dx = trajectory[0, 1:] - trajectory[0, :-1]
    dy = trajectory[1, 1:] - trajectory[1, :-1]
    return np.sqrt(dx*dx + dy*dy) / dt

@jit(nopython=True, cache=True)
def find_convergence_time(trajectory, t_eval, tol=1e-3, window=50):
    """
    T_eff para PUNTOS FIJOS.
    Mira hacia atrás: ¿Desde cuándo la trayectoria no se mueve de la posición final?
    """
    n = trajectory.shape[1]
    if n < window: return t_eval[-1]

    # Posición promedio final
    x_end = np.mean(trajectory[0, -window:])
    y_end = np.mean(trajectory[1, -window:])
    
    # Recorrer hacia atrás
    for i in range(n - window, 0, -1):
        dist = np.sqrt((trajectory[0, i] - x_end)**2 + (trajectory[1, i] - y_end)**2)
        if dist > tol:
            return t_eval[i+1] # Aquí entró a la zona
            
    return t_eval[0]

def calculate_variance_stability_time(trajectory, t_eval, window_size=1000):
    """
    T_eff para CICLOS LÍMITE.
    Calcula cuándo la varianza (amplitud de oscilación) se vuelve constante.
    (No usa Numba porque pandas.rolling es muy eficiente, o numpy striding).
    """
    x = trajectory[0, :]
    
    # Varianza en ventana móvil
    # Usamos una implementación simple con numpy para velocidad
    # (Var = E[x^2] - E[x]^2)
    
    if len(x) < window_size: return t_eval[-1]
    
    # Truco rápido para rolling variance con numpy convolution
    kernel = np.ones(window_size) / window_size
    mean = np.convolve(x, kernel, mode='valid')
    mean_sq = np.convolve(x**2, kernel, mode='valid')
    rolling_var = mean_sq - mean**2
    
    # Si la varianza cambia menos de un 5% respecto al final, consideramos estable
    final_var = np.mean(rolling_var[-100:]) # Promedio del final
    
    if final_var < 1e-6: 
        return 0.0 # Es un punto fijo (varianza 0), este criterio no aplica primariamente
        
    # Buscar hacia atrás cuándo la varianza difiere del final
    threshold = 0.05 * final_var # 5% tolerancia
    
    # Ajuste de índices por la convolución
    offset = window_size - 1
    
    for i in range(len(rolling_var) - 1, 0, -1):
        if abs(rolling_var[i] - final_var) > threshold:
            idx = i + offset
            return t_eval[idx] if idx < len(t_eval) else t_eval[-1]
            
    return t_eval[0]