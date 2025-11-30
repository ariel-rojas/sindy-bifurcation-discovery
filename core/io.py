# core/io.py
import numpy as np
import h5py
import os

def make_param_key(param_arr):
    """
    Genera una clave única string a partir de un array de parámetros.
    Ej: [0.1, 0.2] -> "0.1000_0.2000"
    """
    # Usamos 4 decimales por consistencia, unidos por guion bajo
    return "_".join([f"{p:.4f}" for p in param_arr])

def parse_param_key(key):
    """
    Inverso de make_param_key.
    Ej: "0.1000_0.2000" -> np.array([0.1, 0.2], dtype=float32)
    """
    return np.array([float(x) for x in key.split("_")], dtype=np.float32)

def save_results_to_hdf5(hf_handle, key, data_dict):
    """
    Escribe un diccionario de resultados en un grupo HDF5.
    
    Estructura esperada de data_dict:
    {
        "fixed_points": np.array(...),
        "vector_field": {
            "x_vals": ..., "y_vals": ..., "U": ..., "V": ...
        },
        "trajectories": np.array(...) # (N_traj, dim, steps)
    }
    """
    # Crear grupo principal para esta configuración de parámetros
    if key in hf_handle:
        del hf_handle[key] # Sobrescribir si existe (o manejar error antes)
    grp = hf_handle.create_group(key)
    
    # 1. Puntos Fijos
    if "fixed_points" in data_dict and data_dict["fixed_points"] is not None:
        fp = data_dict["fixed_points"]
        if fp.size > 0:
            grp.create_dataset("fixed_points", data=fp, dtype="f4")

    # 2. Campo Vectorial
    if "vector_field" in data_dict:
        vf_data = data_dict["vector_field"]
        vf_grp = grp.create_group("vector_field")
        # Guardamos todo lo que venga en el diccionario del VF
        for k, v in vf_data.items():
            vf_grp.create_dataset(k, data=v, dtype="f4")

    # 3. Trayectorias (con compresión)
    if "trajectories" in data_dict:
        traj_data = data_dict["trajectories"]
        traj_grp = grp.create_group("trajectories")
        
        # Chunking: (1 trayectoria, todas las dims, todos los pasos)
        # Esto permite leer una trayectoria individual eficientemente
        chunks = (1, traj_data.shape[1], traj_data.shape[2])
        
        traj_grp.create_dataset(
            "all_trajectories",
            data=traj_data,
            dtype="f4",
            compression="gzip",
            chunks=chunks
        )

def load_npz_to_dict(npz_path):
    """
    Carga un .npz temporal y lo estructura en un diccionario limpio
    listo para save_results_to_hdf5.
    """
    with np.load(npz_path) as data:
        # Reconstruir estructura
        out = {}
        
        if "fixed_points" in data:
            out["fixed_points"] = data["fixed_points"]
            
        if "trajectories" in data:
            out["trajectories"] = data["trajectories"]
            
        # Detectar claves de vector field (U, V, x_vals, etc)
        vf_keys = ["U", "V", "x_vals", "y_vals"]
        if any(k in data for k in vf_keys):
            out["vector_field"] = {k: data[k] for k in vf_keys if k in data}
            
        return out