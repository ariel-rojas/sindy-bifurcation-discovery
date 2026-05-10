from dataclasses import dataclass, field
from typing import Tuple, List

# =============================================================================
# 1. PARÁMETROS FÍSICOS (DIMENSIONALES)
# =============================================================================
@dataclass
class PhysicalParams:
    """
    Parámetros biomecánicos base del sistema siringe/pliegues vocales.
    Todas las magnitudes deben expresarse en un sistema coherente (ej. CGS: g, cm, s).
    """
    # Mecánicos
    m: float =0.000126868606427288    # Masa [g]
    gamma1: float = 0.0017067506405885084   # Disipación lineal [g/s]
    gamma2: float = 1e-3         # Disipación no lineal [g/(cm^2 s)]
    kappa2: float = 920.7382759234667    # Rigidez no lineal [g/(cm^2 s^2)]
    c: float = 22.234185873414148   # Disipación cruzada (coupling) [g/(cm^2 s)]
    f0: float = 0.09117712122079111      # Fuerza constante / Tensión base [dyn]

    # Geométricos y Aerodinámicos
    a01: float = 0.11522728975738765    # Apertura glótica en reposo [cm]
    alab: float = 0.0009848213811445162    # Área efectiva de los labios de la siringe [cm^2]
    delta_a: float = -0.001235132210176794 # Asimetría en el canal [cm]
    tau: float = 1.2432588841737166e-05     # Tiempo de tránsito / Retardo de fase [s]

    # Parámetros de control (Valores por defecto si no hay barrido)
    kappa1: float = 20.497113867081037        # Rigidez lineal [g/s^2]
    psub: float = 7936.599159471031       # Presión subglótica [dyn/cm^2]

# =============================================================================
# 2. CONFIGURACIÓN DEL ESPACIO DE FASE
# =============================================================================
@dataclass
class PhaseSpaceConfig:
    """Define los límites y la densidad de las condiciones iniciales (ICs)."""
    # Límites para visualización y generación de grilla global
    x_lim_physical: Tuple[float, float] = (-0.1, 0.02) # Posición [cm]
    y_lim_physical: Tuple[float, float] = (-0.05, 0.05)   # Velocidad [cm/s]
    
    # Grilla global de ICs
    n_traj_per_axis: int = 0
    
    # Perturbación local para ICs cerca de puntos fijos
    local_ic_delta_x: float = 0.002  # [cm]
    local_ic_delta_y: float = 0.002    # [cm/s]
    local_ic_points_per_axis: int = 10
    line_ic_points: int = 0

# =============================================================================
# 3. CONFIGURACIÓN DEL BARRIDO (SWEEP)
# =============================================================================
@dataclass
class SweepConfig:
    """Configuración del barrido de parámetros en el espacio de control."""
    grid_density: int = 10  # Resolución de la grilla de parámetros (NxN)
    
    # Rangos para los parámetros de control [kappa1, psub]
    physical_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.750, 0.815),    # Rango para kappa1 (Rigidez)
        (8630, 8642),  # Rango para psub (Presión)
    ])

# =============================================================================
# 4. CONFIGURACIÓN NUMÉRICA (INTEGRADOR)
# =============================================================================
@dataclass
class NumericConfig:
    """Parámetros del resolvedor ODE y filtros de seguridad."""
    # Tiempo físico real
    t_span_physical: Tuple[float, float] = (0, 5) # [s]
    n_steps: int = 50000
    
    # Resolución del campo vectorial para guardado
    vf_resolution: int = 120

    # Filtros de divergencia (Bail-outs físicos)
    max_radius_physical: float = 15.0   # Radio máximo de escape [cm]
    max_velocity_physical: float = 30.0 # Velocidad máxima permitida [cm/s]

# =============================================================================
# CONFIGURACIÓN MAESTRA (EXPERIMENT)
# =============================================================================
@dataclass
class ExperimentConfig:
    """Clase maestra que consolida toda la configuración del proyecto."""
    system_name: str = "Syrinx Dimensional Model"
    batch_size: int = 40  # Cantidad de simulaciones por lote (paralelismo)
    
    physical_params: PhysicalParams = field(default_factory=PhysicalParams)
    phase_space: PhaseSpaceConfig = field(default_factory=PhaseSpaceConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    numeric: NumericConfig = field(default_factory=NumericConfig)

# Instancia global para importación directa
EXPERIMENT = ExperimentConfig()