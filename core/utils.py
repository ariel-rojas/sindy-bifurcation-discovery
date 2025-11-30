# core/utils.py
import numpy as np
import itertools

def generate_param_grid(param_ranges, density):
    """
    Genera una lista plana de todas las combinaciones de parámetros.
    
    Args:
        param_ranges: Lista de tuplas [(min, max), (min, max), ...]
        density: Entero (puntos por dimensión) o lista de enteros.
        
    Returns:
        job_list: Lista de arrays de parámetros (cada uno shape (M,))
    """
    dim = len(param_ranges)
    
    # Si density es un solo número, replicarlo
    if isinstance(density, int):
        densities = [density] * dim
    else:
        densities = density
        
    # Generar linspace para cada dimensión
    linspaces = []
    for i, (p_min, p_max) in enumerate(param_ranges):
        vals = np.linspace(p_min, p_max, densities[i], dtype=np.float32)
        linspaces.append(vals)
        
    # Producto cartesiano (todas las combinaciones)
    # itertools.product devuelve tuplas, las convertimos a arrays
    cartesian_product = itertools.product(*linspaces)
    
    job_list = [np.array(p, dtype=np.float32) for p in cartesian_product]
    
    return job_list