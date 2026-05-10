# Esquema de tesis — SINDy y descubrimiento de Formas Normales en bifurcaciones de codimensión 2

**Autor:** Ari Rojas
**Fecha:** 2026-05-06 (versión de trabajo)
**Estado:** Esquema consolidado, previo al drafteo de capítulos.

---

## Premisa central

La pregunta de fondo de esta tesis es si SINDy (*Sparse Identification of Nonlinear Dynamics*) es —como suele leerse en la práctica— una "máquina de hacer chorizos": un algoritmo de extracción ciega de ecuaciones a partir de datos, sin compromiso con la estructura matemática del sistema subyacente.

La hipótesis que se va a poner a prueba es la opuesta. SINDy no es una caja negra agnóstica: su sesgo inductivo es la **rareza** (sparsity) del modelo recuperado. La regresión lineal dispersa, núcleo del algoritmo, penaliza la cantidad de términos activos hasta dejar la representación más parsimoniosa compatible con los datos. Ese sesgo coincide exactamente con la estructura de las **Formas Normales** (FN) de un sistema dinámico cerca de una bifurcación: cerca de una degeneración, la dinámica esencial se reduce a un puñado de términos resonantes y todo lo demás puede absorberse mediante cambios de coordenadas. En la literatura emergente sobre data-driven dynamics esto aparece, casi como axioma, en frases como "*SINDy is readily extended to encompass parameterized systems, allowing for the discovery of normal forms associated with a bifurcation parameter μ*" (Brunton, Proctor & Kutz, 2016) y se vuelve explícito en trabajos posteriores que diseñan arquitecturas para forzar la representación latente a una FN canónica (Kalia, Brunton & Kutz, 2021).

La tesis acepta esa correspondencia teórica como punto de partida, pero la convierte en una pregunta empírica: **¿cuándo este *matching* estructural se traduce en una recuperación efectiva de la Forma Normal a partir de datos?** El trabajo ataca la pregunta en dos niveles de dificultad creciente:

1. **Nivel canónico** — datos generados desde la FN exacta (Takens-Bogdanov extendida): ¿recupera SINDy lo que sabe que está ahí? ¿Bajo qué condiciones de muestreo?
2. **Nivel físico** — datos generados desde un sistema biomecánico no polinómico (modelo de masa 1 modificado para la siringe aviar) que admite una bifurcación TB en su espacio de parámetros: ¿recupera SINDy una FN, o algo transformable a ella, sin que la biblioteca de candidatos contenga el término no polinómico (denominador) del modelo original?

El segundo experimento es la apuesta fuerte de la tesis: si funciona, valida la idea de que SINDy puede operar como un destilador de FN incluso cuando el modelo "verdadero" no vive en el espacio de la biblioteca. Si falla, el resultado igual es informativo y delimita con precisión el alcance del método.

---

## Hilo conductor entre secciones

```
[Sección 0]                  [Sección 1]                    [Sección 2]                  [Sección 3]
Marco teórico        →   Validación canónica     →   Experimento decisivo    →   Diagnóstico
SINDy + FN              TB analítica                      Siringe aviar               Cierre
```

- **De 0 a 1:** la Introducción establece *por qué* esperaríamos que SINDy recuperase FN. La Sección 1 lo verifica en el laboratorio más limpio posible (datos sintéticos generados desde una FN conocida).
- **De 1 a 2:** una vez calibrado lo que SINDy sabe hacer en un caso favorable, se sube la apuesta a un sistema físico cuyo *ground truth* no es polinómico. La Sección 2 es donde la premisa central se juega.
- **De 2 a 3:** el cierre vuelve sobre la pregunta inicial y la responde con evidencia, marcando los límites del método.

Cada sección tiene como salida un *resultado falsable* explícito —no se trata de un barrido exploratorio—.

---

## Sección 0 — Introducción

**Objetivo:** establecer el aparato conceptual y matemático mínimo para que la pregunta de investigación tenga sentido.

### 0.1 ¿Qué es SINDy?

Presentación del marco data-driven de descubrimiento de ecuaciones (Brunton, Proctor & Kutz, 2016). SINDy parte de una serie temporal $\{x(t_k)\}$ y de su derivada estimada, propone una biblioteca de funciones candidatas $\Theta(x)$ —típicamente polinomios de bajo orden— y resuelve

$$
\dot{X} = \Theta(X)\,\Xi
$$

por regresión rala, buscando un $\Xi$ con la mayor cantidad posible de coeficientes nulos. Ubicar SINDy dentro del panorama de equation discovery (Symbolic Regression, EDM, autoencoders + SINDy) y explicitar lo que lo distingue: la salida es interpretable, lineal en los parámetros, y no requiere arquitectura neuronal.

### 0.2 Regresión lineal dispersa: el núcleo matemático

Desarrollo de la maquinaria de mínimos cuadrados con penalización (LASSO, STLSQ — *Sequential Thresholded Least-Squares*). Énfasis en por qué la dispersión no es un capricho regularizador, sino una **hipótesis sobre el mundo**: si el sistema verdadero es ralo, entonces el estimador disperso es el correcto en el sentido de Occam. Esta subsección es donde se siembra la conexión con FN: las Formas Normales retienen únicamente los términos no lineales resonantes, todo lo demás se absorbe en cambios de coordenadas; por construcción, son objetos ralos. Esto convierte la búsqueda de FN desde datos en un problema *naturalmente* alineado con el sesgo inductivo de SINDy.

### 0.3 Espacio de hiperparámetros y selección de modelo

Discusión de los tres ejes principales: el umbral de dispersión $\lambda$ (corta o no un coeficiente), el coeficiente Ridge $\alpha$ (regulariza la magnitud), y el orden de la biblioteca polinómica. Cómo cada uno afecta la curva de Pareto entre error de ajuste y cantidad de términos activos. Estrategias de selección: criterios tipo BIC, *cross-validation* sobre trayectorias, y validación física por simulación con el modelo recuperado (este último es el criterio que se usa en las secciones 1 y 2).

### 0.4 Pregunta de investigación

Planteo formal de la hipótesis de la tesis:

> *Dadas trayectorias de un sistema dinámico que admite una bifurcación de codimensión 2, ¿bajo qué condiciones de muestreo y de biblioteca SINDy recupera —exacta o módulo cambio de coordenadas— la Forma Normal de la bifurcación?*

Subpreguntas derivadas que estructuran las secciones siguientes:

- (Q1) ¿Es necesario observar todas las zonas topológicas del diagrama de bifurcaciones para identificar la FN, o alcanza con una sola zona si se permite variar los parámetros de control?
- (Q2) ¿Qué pasa si los parámetros de control están **fijos** durante el experimento?
- (Q3) ¿Qué hace SINDy si los datos no son cercanos a la degeneración?
- (Q4) ¿La conclusión sobrevive al pasar de un sistema generado desde su FN a un sistema físico cuya dinámica está fuera del espacio de la biblioteca?

---

## Sección 1 — SINDy aplicado a Formas Normales (Codimensión 2)

**Objetivo:** validar (o falsar) la hipótesis sobre un sistema controlado donde la respuesta correcta es conocida analíticamente. Es el experimento de calibración del método.

### 1.1 Formas Normales y reducción a la variedad central

Recapitulación breve de la teoría de FN y del Center Manifold Theorem: cómo, cerca de una bifurcación, la dinámica relevante vive en un subespacio de baja dimensión y admite una representación canónica universal módulo cambios de coordenadas. Es el núcleo teórico que justifica por qué *tiene sentido* preguntarle a SINDy por una FN: el objeto buscado existe y es único hasta equivalencia.

### 1.2 Codimensión 2 y la bifurcación de Takens-Bogdanov

Definición formal: la bifurcación TB ocurre en sistemas con un punto fijo cuya linealización tiene un autovalor doble en cero con bloque de Jordan no trivial. Es de **codimensión 2** porque hace falta variar dos parámetros independientes para encontrarla genéricamente, y de su entorno emanan tres curvas de bifurcaciones de codimensión 1 (saddle-node, Andronov-Hopf, homoclínica) que dividen el plano de parámetros en cinco zonas topológicamente distintas. El despliegue universal en su versión extendida (con términos de orden superior que se incluyen para tener un modelo simulable globalmente bien comportado) es

$$
\begin{aligned}
\dot{x} &= y \\
\dot{y} &= -\mu_1 - \mu_2 x + x^2 - x^3 - xy - x^2 y
\end{aligned}
$$

Justificación de por qué TB es el caso de estudio ideal: combina dos subgrupos de bifurcación distintos (estática y de Hopf) en un solo punto, su FN es polinómica y compatible con la biblioteca natural de SINDy, y el diagrama tiene riqueza topológica suficiente para diseñar varios protocolos de muestreo no triviales.

### 1.3 Diseño experimental: cuatro estrategias de muestreo

Esta es la subsección crítica. Cada estrategia responde una pregunta operativa sobre lo que SINDy puede o no puede hacer en condiciones progresivamente más realistas.

**Caso A — Datos de las cinco zonas.** Trayectorias muestreadas en todo el diagrama de bifurcaciones, cubriendo las cinco regiones topológicas. Es el escenario más favorable posible: SINDy ve el sistema "completo" en términos cualitativos. Pregunta: ¿es capaz el algoritmo de recuperar exactamente los siete términos del despliegue universal, con sus coeficientes? Funciona como **cota superior** del desempeño del método y como sanidad de la implementación.

**Caso B — Datos de una sola zona, variando los parámetros de control.** Trayectorias confinadas a una única región topológica (se repite el experimento para cada una de las cinco), pero con $(\mu_1,\mu_2)$ variando dentro de esa zona. Modela el escenario experimental típico: el experimentador puede perturbar las variables de control —presión, voltaje, temperatura, etc.— pero no puede arrastrar al sistema a través de las curvas de bifurcación porque no las conoce a priori o porque sería invasivo. Pregunta: ¿cuánta información topológica necesita SINDy para reconstruir la FN, o alcanza con la dependencia paramétrica local?

**Caso C — Datos de una sola zona, parámetros fijos.** Trayectorias con $(\mu_1, \mu_2)$ congelados en un valor de cada zona. Modela el escenario más restrictivo y, en muchos campos, el más realista: una serie temporal larga con condiciones experimentales fijas. Pregunta: ¿puede SINDy recuperar la FN sin información sobre la dependencia paramétrica? La hipótesis es que pierde los términos en $\mu$ pero conserva la estructura no lineal, y se discute qué se gana y qué se pierde respecto al Caso B.

**Caso D — Datos lejos de la zona de bifurcación TB.** Trayectorias muestreadas en una región del espacio de parámetros donde el sistema **no** está cerca de la degeneración. Es el control negativo del trabajo: SINDy nunca está restringido a operar cerca de bifurcaciones, y un experimentador puede no tener idea de si su sistema está cerca de una. Pregunta: ¿qué entrega SINDy en ese caso? La predicción es que entrega un modelo válido localmente pero estructuralmente distinto de la FN de TB —un ajuste fenomenológico— y eso permite, por contraste, caracterizar lo que es específicamente "ver la FN" frente a "ajustar la dinámica local".

El resultado conjunto de los cuatro casos es un mapa de operación: en qué condiciones SINDy efectivamente recupera la FN, y en cuáles entrega un modelo ralo pero no canónico.

### 1.4 Aprovechamiento de la herramienta y *domain knowledge*

Bajo el lema *"Si lo sé usar, y cuando sale, lo sé aprovechar"*: discusión del rol del conocimiento experto en el ajuste de hiperparámetros y en la interpretación. El umbral de dispersión, el orden de la biblioteca y la elección de la grilla de muestreo no son neutrales; un usuario informado encuentra la FN, uno no informado encuentra un ajuste polinómico cualquiera. Esta subsección hace explícito el costo de operación del método y desbarata la lectura "automatizada" de SINDy.

### 1.5 Resultados esperados de la sección

Salida concreta: un *mapa de recuperación* que indica, para cada combinación (estrategia de muestreo) × (hiperparámetros), cuán cerca queda el modelo recuperado de la FN exacta, medido tanto por error en coeficientes como por equivalencia topológica del retrato de fases reconstruido. Este mapa es el insumo de calibración para la Sección 2.

---

## Sección 2 — SINDy aplicado a un sistema físico real: la siringe aviar

**Objetivo:** llevar la metodología validada en la Sección 1 a un sistema cuya dinámica **no** vive en el espacio de la biblioteca polinómica de SINDy, y evaluar si aún así el algoritmo recupera la Forma Normal de Takens-Bogdanov del sistema en torno a una degeneración conocida.

### 2.1 Contexto biológico y mecánico

Breve presentación del aparato fonador aviar: la siringe como oscilador no lineal forzado por flujo aéreo, y el rol de los músculos sirngo-laríngeos en el control de la vocalización. Justificación de por qué este sistema —además de ser interesante en sí mismo— es un buen banco de prueba: tiene baja dimensión efectiva, exhibe bifurcaciones de codimensión 2 documentadas (en particular TB) en su espacio de parámetros, y existe un modelo biomecánico de referencia con el que comparar.

### 2.2 El modelo de masa 1 modificado

Presentación del modelo biomecánico que oficia de *ground truth* en este capítulo. La estructura genérica es

$$
\ddot{x} + \frac{f(x,\dot{x};\theta)}{g(x;\theta)} = 0
$$

donde el término en denominador es lo que vuelve interesante el experimento desde el punto de vista metodológico: **el modelo verdadero no es polinómico**, mientras que la biblioteca de SINDy en este capítulo se restringe a polinomios hasta orden 3. Aclaración clave: no es objetivo recuperar la ecuación física tal cual; el objetivo es recuperar una **forma reducida**, idealmente la FN de TB en la vecindad del punto crítico, o un objeto polinómico transformable a ella mediante un cambio de coordenadas.

Parámetros de control: presión en los sacos aéreos y tensión muscular. Locación del punto de codimensión 2 en este espacio (a referenciar desde el diagrama de bifurcaciones del modelo).

### 2.3 Flujo de trabajo

**Generación y preprocesamiento de datos.** Simulaciones del modelo de masa 1 modificado integradas con el integrador RK4 propio del repositorio, en torno al punto TB. Estimación de derivadas, tratamiento de transitorios, criterios de descarte por estiffness o no-convergencia (heredados de las métricas ya implementadas en `core/metrics`).

**Grillado del espacio de fases y de parámetros.** Discretización razonada de $(x, \dot{x})$ y $(\text{presión}, \text{tensión})$ informada por el diagrama de bifurcaciones del sistema. Estrategia de muestreo: prevalece el análogo del Caso B de la Sección 1 (zona única en torno a TB, parámetros variando) por ser el más realista para un experimento de medición sobre una preparación biológica o un modelo computacional con costo no trivial.

**Parametrización del modelo SINDy.** Ajuste fino de hiperparámetros sobre los datos de la siringe, anclado en el rango calibrado en la Sección 1. Discusión explícita de las diferencias respecto a la calibración canónica: el sistema verdadero está fuera de la biblioteca, por lo que se espera mayor sensibilidad al umbral $\lambda$ y al coeficiente Ridge.

**Reconstrucción.** Integración del modelo recuperado por SINDy y comparación de su retrato de fases con el del modelo biomecánico. La pregunta operativa: ¿el modelo recuperado tiene el mismo *unfolding* topológico que TB?

**Comparación crítica.** Contraste cuantitativo (error de coeficientes contra una FN de referencia obtenida analíticamente sobre el modelo de masa 1) y cualitativo (equivalencia topológica del diagrama de bifurcaciones reconstruido versus el de la siringe).

### 2.4 Resultado decisivo de la sección

La Sección 2 produce uno de tres veredictos posibles, todos científicamente útiles:

- **Recuperación directa:** SINDy entrega una FN de TB con coeficientes consistentes con los del modelo biomecánico desarrollado a mano. La premisa central queda confirmada en su versión más fuerte.
- **Recuperación módulo transformación:** SINDy entrega un modelo polinómico ralo que, mediante un cambio de coordenadas explícito, es equivalente a la FN de TB. Confirmación parcial: el método ve la estructura, no las coordenadas.
- **No recuperación:** SINDy entrega un ajuste fenomenológico local sin estructura canónica reconocible. Falsación de la hipótesis fuerte y delimitación clara del alcance del método.

---

## Sección 3 — Conclusiones

**Objetivo:** integrar los resultados de las secciones 1 y 2 y responder la pregunta de investigación de la Sección 0.

### 3.1 Eficacia del método

Síntesis de la evidencia. Si la cadena 1 → 2 cierra (recuperación canónica + recuperación física, total o módulo cambio de coordenadas), se argumenta que la arquitectura matemática de SINDy es excepcionalmente adecuada para descubrir FN cerca de degeneraciones, **precisamente** porque ambos objetos —el modelo recuperado y la FN buscada— son ralos por construcción. Si el cierre es parcial, se acota con precisión qué función puede asumir SINDy en un *workflow* de descubrimiento de bifurcaciones (por ejemplo: detector de estructura local, no estimador exacto de coeficientes).

### 3.2 Limitaciones identificadas

Mapeo honesto de los regímenes en los que el método falla o degrada: ruido, no-polinomialidad fuerte del *ground truth*, observación parcial del estado, costo del *domain knowledge* en la elección de hiperparámetros. Esta subsección es la respuesta directa a la lectura ingenua de SINDy como "máquina de hacer chorizos".

### 3.3 Implicancias y trabajo futuro

Reflexión sobre las puertas que abre la metodología: derivación sistemática de modelos reducidos a partir de mediciones en sistemas biológicos complejos, diagnóstico de proximidad a bifurcaciones desde datos experimentales, posibilidad de automatizar el paso analítico de Center Manifold Reduction. Conexión con las extensiones modernas del marco SINDy (autoencoders + FN, SINDy-PI para sistemas implícitos, SINDyCP para parámetros de control) como caminos naturales si el alcance del método estándar resulta insuficiente.

---

## Bibliografía de referencia (núcleo)

**Marco original y ampliaciones de SINDy:**

- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*. **PNAS**, 113(15), 3932–3937. — Paper fundacional. Recupera explícitamente la FN de Hopf con el parámetro de bifurcación tratado como variable.
- Kaiser, E., Kutz, J. N., & Brunton, S. L. (2018). *Sparse identification of nonlinear dynamics for model predictive control in the low-data limit*. **Proc. R. Soc. A**, 474(2219). — Régimen de pocos datos, relevante para Sección 2.
- Champion, K., Lusch, B., Kutz, J. N., & Brunton, S. L. (2019). *Data-driven discovery of coordinates and governing equations*. **PNAS**, 116(45), 22445–22451. — El argumento explícito de que descubrir ecuaciones requiere primero descubrir coordenadas; cita central para la discusión del Caso "recuperación módulo transformación" en Sección 2.
- Kalia, M., Brunton, S. L., Kutz, J. N. *et al.* (2021). *Learning normal form autoencoders for data-driven discovery of universal, parameter-dependent governing equations*. arXiv:2106.05102. — La conexión más directa de SINDy con la teoría de FN; cita obligada para sostener la premisa central.
- Kaheman, K., Kutz, J. N., & Brunton, S. L. (2020). *SINDy-PI: a robust algorithm for parallel implicit sparse identification of nonlinear dynamics*. **Proc. R. Soc. A**, 476(2242). — Variante implícita; discutible para la Sección 2 dado el término en denominador del modelo de la siringe.

**Aplicaciones recientes a sistemas con bifurcaciones:**

- Nicolaou, Z. G., *et al.* (2023). *Data-driven discovery and extrapolation of parameterized pattern-forming dynamics*. **Phys. Rev. Research**, 5, L042017. — SINDyCP, parámetros de control externos.
- Conti, P., *et al.* (2025). *Sparse Identification for bifurcating phenomena in Computational Fluid Dynamics*. arXiv:2502.11194. — Aplicación directa a fenómenos bifurcantes.
- *Online learning in bifurcating dynamic systems via SINDy and Kalman filtering* (2025), **Nonlinear Dynamics**. — Estimación online en sistemas bifurcantes.

**Teoría de bifurcaciones (referencias clásicas):**

- Kuznetsov, Y. A. (2004). *Elements of Applied Bifurcation Theory* (3rd ed.). Springer. — Texto canónico para Takens-Bogdanov, FN y despliegues universales.
- Guckenheimer, J., & Holmes, P. (1983). *Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields*. Springer. — Tratamiento clásico de variedades centrales y FN.
- Wiggins, S. (2003). *Introduction to Applied Nonlinear Dynamical Systems and Chaos* (2nd ed.). Springer. — Complemento moderno.

**Sistema físico (siringe / vocalización aviar):**

- *Ubicar aquí las referencias específicas del modelo de masa 1 modificado y de los trabajos previos del grupo o de la literatura sobre dinámica de la siringe (por ejemplo, trabajos de Mindlin, Laje, Amador y colaboradores, según corresponda al modelo concreto que se utiliza).*

---

## Decisiones pendientes / a discutir con el director

1. **Formato final del documento:** ¿LaTeX desde el arranque o se redacta primero en Markdown y se convierte? (Esto afecta cómo se organizan figuras y bibliografía desde el principio.)
2. **Alcance de la Sección 2:** ¿se incluye el caso lejos de TB también para la siringe, o se mantiene como control negativo solo en la Sección 1?
3. **Bibliografía de la siringe:** completar la lista con las referencias del modelo concreto utilizado (presumiblemente trabajos del grupo de Mindlin/Amador, pero requiere confirmación).
4. **Métricas de "equivalencia topológica":** ¿usar el conjunto de bifurcaciones detectadas como criterio principal, o números de Morse/índices de equilibrios?
