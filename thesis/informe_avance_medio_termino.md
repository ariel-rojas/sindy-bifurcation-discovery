# Informe de avance de medio término — Tesis de Licenciatura

**Estudiante:** Ari Rojas
**Correo electrónico:** arirojas50@gmail.com
**Director/a:** *[completar]*
**Co-director/a:** *[completar si corresponde]*
**Fecha del informe:** 7 de mayo de 2026
**Título tentativo:** *Identificación de Formas Normales en bifurcaciones de codimensión 2 mediante SINDy: del caso canónico de Takens-Bogdanov a la siringe aviar.*

---

## 1. Resumen y contexto

### 1.1 El método y su premisa

**SINDy** (*Sparse Identification of Nonlinear Dynamics*), introducido por Brunton, Proctor y Kutz en 2016, es un método data-driven para descubrir las ecuaciones diferenciales que gobiernan un sistema dinámico a partir de mediciones de su evolución temporal. El procedimiento es directo: se propone una biblioteca extensa de funciones candidatas —típicamente polinomios de bajo orden— y se resuelve un problema de regresión lineal para encontrar la combinación de funciones que mejor reproduce la derivada observada. La hipótesis clave del método es la **parsimonia**: en una base apropiada, la dinámica de la mayoría de los sistemas físicos puede describirse mediante muy pocos términos. Esta hipótesis se materializa imponiendo que la solución del problema de regresión sea **rala**, esto es, que la mayoría de los coeficientes recuperados sean exactamente cero.

La premisa de raredad no es arbitraria. Cerca de una bifurcación, la teoría clásica de **Formas Normales** garantiza que la dinámica esencial del sistema queda descripta, módulo cambio de coordenadas, por un polinomio de pocos términos —los términos resonantes—; el resto del campo vectorial se absorbe en la transformación. Las Formas Normales son, por construcción, ralas. La tesis investiga si esta coincidencia estructural entre el sesgo inductivo del algoritmo (rareza) y el objeto matemático buscado (Forma Normal rala) es suficiente para que SINDy funcione efectivamente como un método de descubrimiento de Formas Normales a partir de datos, en vez de un mero ajuste fenomenológico.

### 1.2 Estructura del trabajo

La pregunta empírica se aborda en dos experimentos de dificultad creciente:

**Experimento 1 — Bifurcación canónica de Takens-Bogdanov.** Takens-Bogdanov (TB) es la bifurcación de codimensión 2 que ocurre cuando el Jacobiano de un punto fijo tiene un autovalor doble en cero con bloque de Jordan no trivial. De su despliegue universal emergen tres curvas de bifurcación de codimensión 1 —saddle-node, Hopf y homoclínica— que se encuentran en el punto TB y dividen el plano de los dos parámetros de control en **cinco zonas dinámicas distintas**: una zona con un único punto fijo (silla, $\mu_1<0$); una zona con un único punto fijo de estabilidad opuesta ($\mu_1\geq 0$); una zona donde coexisten tres puntos fijos sin ciclo límite; una zona donde a esos tres puntos fijos se le suma un ciclo límite estable, interno a la curva homoclínica; y una zona post-homoclínica de escape, sin ciclo. El sistema utilizado como *ground truth* en este experimento es el despliegue extendido $\dot{x}=y$, $\dot{y} = -\mu_1 - \mu_2 x + x^2 - x^3 - xy - x^2 y$, que tiene siete términos no nulos sobre una biblioteca polinómica de 35 candidatos: el problema es estructuralmente ralo. El experimento se diseña sobre **cuatro estrategias de muestreo**, pensadas para aproximar progresivamente las restricciones de un experimento real: (A) trayectorias en las cinco zonas a la vez —cota superior favorable, todo el diagrama observado—; (B) trayectorias confinadas a una sola zona pero variando los parámetros de control dentro de ella —el caso de un experimentador que puede perturbar parámetros pero no puede arrastrar al sistema a través de una bifurcación—; (C) trayectorias con parámetros fijos en un único punto —experimento sin control paramétrico, una serie temporal larga con condiciones congeladas—; y (D) trayectorias muestreadas lejos del punto TB —control negativo, para diagnosticar qué hace SINDy cuando los datos no son cercanos a una degeneración—.

**Experimento 2 — Siringe aviar.** El segundo experimento lleva el método a un sistema físico real. La siringe aviar puede modelarse como un oscilador no lineal de un grado de libertad cuya dinámica responde a un modelo de masa 1 modificado:

$$
\dot{x} = y, \qquad \dot{y} = \frac{1}{m}\!\left[-(\kappa_1 + \kappa_2 x^2)x - (\gamma_1 + \gamma_2 y^2)y - c\, x^2 y + f_0 + \alpha_{\text{lab}} P_{\text{sub}} \,\frac{\delta_a + 2\tau y}{a_{01} + x + \tau y}\right].
$$

Los dos **parámetros de control** son la tensión muscular $\kappa_1$ y la presión subglótica $P_{\text{sub}}$; el resto de las constantes son estructurales. En el plano $(P_{\text{sub}}, \kappa_1)$ se localiza numéricamente un **punto de Takens-Bogdanov** —intersección de las curvas saddle-node y Hopf detectadas analíticamente para esta familia—, y se generan trayectorias de entrenamiento en una ventana paramétrica alrededor de él. La pregunta operativa del experimento es la siguiente: dado que el sistema verdadero contiene un término no polinómico (el cociente con denominador $a_{01} + x + \tau y$) y por lo tanto **no vive en el espacio de la biblioteca polinómica de SINDy**, ¿es capaz el método de recuperar la Forma Normal de TB —exacta, o módulo un cambio de coordenadas reconstruible— a partir de las trayectorias observadas? Tres veredictos son posibles y todos son científicamente útiles: recuperación directa, recuperación módulo transformación, o no recuperación con caracterización del régimen de falla.

### 1.3 Avance global

A medio término, el primer experimento está sustancialmente avanzado: el pipeline completo de generación de datos, entrenamiento, evaluación y comparación está implementado, con resultados parciales para las cinco zonas bajo distintas estrategias de muestreo. El segundo experimento se encuentra en fase inicial: el sistema está implementado en el código pero aún no se han producido resultados sistemáticos. La escritura de la tesis arrancó recientemente, con el esquema general consolidado y dos capítulos teóricos en primera redacción. **El cronograma para finalizar dentro del plazo de un año es factible siempre que se resuelvan, en el orden indicado, las dificultades enumeradas en la sección 3**.

---

## 2. Estado del trabajo

**Marco teórico y bibliografía núcleo (completo).** El relevamiento sobre SINDy (trabajo original de Brunton, Proctor & Kutz 2016 y derivados) y sobre teoría de bifurcaciones (Kuznetsov 2004, Guckenheimer & Holmes 1983) está cerrado, con la conexión explícita SINDy ↔ Formas Normales documentada en la literatura, especialmente en el trabajo sobre *normal form autoencoders* de Kalia, Brunton & Kutz (2021), que sostiene la premisa central de la tesis.

**Experimento 1 — implementación completa, resultados parciales.** El repositorio `sindy-bifurcation-discovery` contiene el sistema TB extendido implementado con aceleración Numba JIT, generación masiva de trayectorias en grilla con almacenamiento HDF5, detección numérica automática de la curva homoclínica, entrenamiento SINDy con biblioteca polinómica de orden 3 sobre el estado aumentado $(x, y, \mu_1, \mu_2)$, búsqueda de hiperparámetros por grid search y refinamiento por hill climbing, evaluación cuantitativa con tres métricas complementarias (error relativo por término, identificación de términos espurios, error agregado), y un visualizador interactivo para diagnosticar resultados. Los experimentos se ejecutaron sobre los cuatro casos de muestreo previstos; los resultados preliminares confirman la hipótesis de partida en los casos A y B, y permiten distinguir empíricamente entre regímenes de recuperación estable, recuperación parcial y no recuperación. Resta consolidar el análisis sistemático del Caso D y presentar los resultados de los Casos B y C en una tabla comparativa final por zona.

**Experimento 2 — implementación inicial, pendiente de resultados.** El sistema biomecánico está implementado en `systems/syrinx.py` con la ecuación dimensional completa, junto con solvers analíticos para las curvas de saddle-node y Hopf y un detector numérico del punto TB en el plano $(P_{\text{sub}}, \kappa_1)$. Existen scripts iniciales de entrenamiento y simulación (`syrinx/sindy_training.py`, `simulate_sindy.py`, `mean_sindy.py`), pero no se han ejecutado aún los barridos sistemáticos comparables a los del primer experimento.

**Escritura.** El esquema general de la tesis está consolidado, con la pregunta de investigación, las hipótesis falsables por capítulo y el hilo conductor explícito entre secciones. Las primeras redacciones del capítulo introductorio (SINDy) y del capítulo de Takens-Bogdanov (~25 páginas LaTeX en total) están en estado de borrador iterable. Los capítulos de resultados experimentales y de aplicación a la siringe están sin redactar.

---

## 3. Dificultades encontradas

**Organización del código y cuaderno desactualizado.** El repositorio acumuló durante el desarrollo varios scripts que quedaron parcial o totalmente obsoletos, y el cuaderno de bitácora se actualizó de manera irregular. Esto dificulta la reconstrucción de qué experimentos están vigentes. *Acción correctiva:* dedicar una semana al curado del repositorio y a poner el cuaderno al día antes de avanzar con el Experimento 2.

**Riesgo del Experimento 2.** La pregunta del segundo experimento admite tres veredictos posibles, dos de los cuales requieren análisis adicional (en particular el caso "recuperación módulo cambio de coordenadas" exige construir explícitamente la transformación entre las coordenadas recuperadas y las canónicas de TB). *Acción preventiva:* explorar tempranamente, en cuanto haya primeros resultados de la siringe, qué veredicto es el efectivo, para ajustar el alcance del análisis posterior antes de avanzar con la redacción.

**Bibliografía específica del modelo de la siringe.** Las referencias del modelo biomecánico están parcialmente identificadas pero no consolidadas. *Acción correctiva:* confirmar con la dirección la lista exacta de citas antes del cierre del Capítulo 2.

**Costo computacional.** El barrido completo (zona × caso de muestreo × hiperparámetros × ventana temporal × bootstrap) representa varios miles de entrenamientos individuales. La infraestructura local funciona pero los tiempos son significativos. *Acción correctiva:* evaluar migrar parte del barrido a cómputo en la nube; la integración con Google Drive ya está parcialmente implementada en `statistical_sweep.py`.

**Decisiones metodológicas pendientes con la dirección.** Quedan abiertas: (i) la métrica precisa de "equivalencia topológica" para comparar el retrato de fases recuperado con el verdadero; (ii) si el caso "lejos de TB" se incluye también en el experimento de la siringe; (iii) el formato final de presentación (LaTeX desde el arranque versus Markdown con conversión posterior).

---

## 4. Cronograma para los próximos seis meses y productos al medio término

| Mes | Actividad |
|---|---|
| 1 | Curado del repositorio y del cuaderno; consolidación de resultados del Experimento 1; redacción del capítulo de resultados de TB. |
| 2–3 | Ejecución sistemática del Experimento 2 (siringe) sobre las estrategias de muestreo equivalentes. |
| 4 | Análisis de resultados de la siringe; construcción —si corresponde— de la transformación entre coordenadas para el caso "recuperación módulo cambio de coordenadas". |
| 5 | Redacción del capítulo de la siringe y del capítulo de conclusiones; revisión iterativa con la dirección. |
| 6 | Revisión final, formato, defensa preparatoria. |

**Productos disponibles a la fecha del informe.**

- Repositorio funcional con pipeline completo para el Experimento 1.
- Esquema de tesis consolidado con pregunta de investigación, hipótesis por capítulo y bibliografía núcleo.
- Dos capítulos en primera redacción (introducción a SINDy, ~10 páginas; Takens-Bogdanov, ~15 páginas).
- Resultados parciales del Experimento 1 sobre las cinco zonas y las cuatro estrategias de muestreo.
- Implementación inicial del Experimento 2 con el sistema biomecánico codificado y scripts de entrenamiento operativos.

*El cronograma propuesto contempla la finalización dentro del plazo de un año previsto. Las dificultades identificadas son tratables y no comprometen la viabilidad del trabajo.*
