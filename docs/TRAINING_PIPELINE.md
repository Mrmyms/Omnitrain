# Pipeline de Entrenamiento Industrial: Conectoma v2.1 (Industrial)

Este documento define la metodología oficial recomendada para entrenar sistemas robóticos autónomos impulsados por **Liquid Neural Networks (LNN)** y arquitecturas **Closed-form Continuous-time (CfC)**. La metodología v2.1 introduce estabilidad de grado industrial para despliegues críticos.

---

## Metodología de Entrenamiento (Industrial Curriculum)

Entrenar una LNN requiere un enfoque radicalmente distinto al de un Transformer estático, ya que las derivadas fluyen a través del tiempo continuo ($\Delta t$). Este pipeline integra estabilidad formal y paridad de datos.

### Fase 1: Pre-entrenamiento Sensorial y Paridad de Datos
Antes de procesar dinámicas temporales, el sistema debe aprender a normalizar la realidad.
*   **Captura de Estadísticas (v2.1):** Durante la carga del dataset (`OmniLogDataset`), el sistema captura automáticamente la media y desviación estándar de cada sensor.
*   **Importancia:** Estas estadísticas se guardan en el bundle `.omni`. Sin ellas, el robot sufriría de "degradación de datos" al recibir valores crudos en tiempo real que no coinciden con la distribución de entrenamiento.

### Fase 2: Clonación de Comportamiento (Stateful BPTT)
Enseñanza de los reflejos motores base mediante demostraciones humanas.
*   **Mecánica:** El entrenamiento es **Stateful**. El estado latente del cerebro se propaga entre secuencias contiguas de una trayectoria, permitiendo que el robot aprenda dependencias temporales de largo alcance (ej. recordar que pasó por una puerta hace 10 segundos).

### Fase 3: Inyección de Caos (Domain Randomization)
La ventaja principal de las LNN es su resiliencia natural a condiciones fuera de distribución (OOD).
*   **Mecánica:** Se inyecta ruido gaussiano y fallos de sensores (dropout). 
*   **Nota Técnica:** El ruido se aplica **después** de la normalización Z-Score pero **antes** del clamping de activación, permitiendo que la red aprenda a ignorar señales ruidosas sin saturarse.

### Fase 4: Estabilidad Lagrangiana (Formal Safety)
Pulido final del modelo con garantías de seguridad matemática.
*   **Lagrangian Dual Update (v2.1):** Se utiliza un optimizador primal-dual para ajustar el peso de la seguridad. Las actualizaciones del multiplicador de Lagrange ($\lambda$) se realizan **por secuencia**, eliminando las oscilaciones violentas de versiones anteriores y logrando una política de seguridad mucho más estable.
*   **OmniShield:** Si la red propone una acción peligrosa, el escudo (ICNN) proyecta la acción a la zona segura mediante una optimización de proyección cuadrática (QP).

### Fase 5: Consolidación Sináptica (Structured Pruning)
Optimización post-entrenamiento para hardware de bajo consumo (Edge Computing).
*   **Mecánica:** Se eliminan las neuronas y conexiones más débiles de forma estructural. Esto reduce el tamaño del modelo hasta en un 60% y disminuye la latencia en dispositivos NVIDIA Jetson o Qualcomm.

---

## Referencias y Fuentes Oficiales (MIT)

1.  **Arquitectura CfC:** Nature Machine Intelligence (2022). Ramin Hasani, et al.
2.  **Robustez OOD:** Science Robotics (2023). Makram Chahine, Ramin Hasani, et al.
3.  **Seguridad Formal:** ICNN-based Control Barrier Functions (2021).

---
*OmniTrain Project Documentation - 2026 (v2.1 Industrial)*
