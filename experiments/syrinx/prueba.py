import numpy as np
import matplotlib.pyplot as plt

def analyze_hopf_region():
    # 1. Parámetros físicos base (extraídos de tu SyrinxModel)
    m: float = 0.001814507993934839     # Masa [g]
    g1: float = 0.007465120548965725   # Disipación lineal [g/s]
    g2: float = 1.0          # Disipación no lineal [g/(cm^2 s)]
    k2: float = 965.249201360614    # Rigidez no lineal [g/(cm^2 s^2)]
    c: float = 0.06092228732718396      # Disipación cruzada (coupling) [g/(cm^2 s)]
    f0: float = 0.06551237644460184       # Fuerza constante / Tensión base [dyn]

    # Geométricos y Aerodinámicos
    a01: float = 0.13484086482347213     # Apertura glótica en reposo [cm]
    alab: float = 0.0002910510783799829    # Área efectiva de los labios de la siringe [cm^2]
    da: float = -0.002275695567017906 # Asimetría en el canal [cm]
    tau: float = 3.642893008281856e-06     # Tiempo de tránsito / Retardo de fase [s]

    
    # 2. Rango de parametrización x (coordenada de equilibrio)
    # Evaluamos para x > 0.001. No evaluamos en 0 exacto por la singularidad (f0/x).
    x_vals = np.linspace(-100, 100, 2000)
    
    h_p = []
    h_k = []
    
    for x in x_vals:
        # Psub para la curva de Hopf
        num_p = (g1 + c*x**2) * (a01 + x)**2
        den_p = alab * tau * (2*a01 + 2*x - da)
        psub_h = num_p / den_p
        
        # Kappa1 para la curva de Hopf
        k1_h = -k2*x**2 + f0/x + (psub_h * alab * da) / (x * (a01 + x))
        
        # Determinante de la Jacobiana (debe ser > 0 para que sea Hopf real y no ensilladura)
        j21 = (1.0/m) * (-(k1_h + 3*k2 * x**2) - (alab * psub_h * da) / (a01 + x)**2)
        det = -j21
        
        # Filtro de validez del modelo físico
        if k1_h > 0 and psub_h > 0 and det > 0:
            h_p.append(psub_h)
            h_k.append(k1_h)

    # 3. Renderizado del análisis
    plt.figure(figsize=(10, 6))
    
    # Dibujar la curva de Hopf real
    plt.plot(h_p, h_k, 'r-', lw=2.5, label='Curva de Hopf parametrizada')
    
    # Dibujar la "ventana" de la imagen que compartiste para entender la escala
    plt.axvspan(13200, 13450, ymin=0, ymax=1, color='yellow', alpha=0.2, 
                label='Región visible en tu Visor actual')
    
    # Límites de kappa1 de tu visor
    plt.axhline(0.1, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(0.55, color='gray', linestyle='--', alpha=0.5)

    plt.xlabel(r'Presión Subglótica $p_{sub}$ (dyn/cm$^2$)')
    plt.ylabel(r'Rigidez $\kappa_1$ (dyn/cm)')
    plt.title('Ubicación real de la Curva de Hopf en el espacio de parámetros')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    plt.show()

if __name__ == "__main__":
    analyze_hopf_region()