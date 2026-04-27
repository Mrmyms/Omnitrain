# Pipeline de Entrenamiento Industrial: Redes Neuronales Líquidas (CfC) 🎓🤖

Este documento define la metodología oficial recomendada para entrenar sistemas robóticos autónomos impulsados por **Liquid Neural Networks (LNN)** y arquitecturas **Closed-form Continuous-time (CfC)**. La metodología está diseñada para pasar de la simulación al mundo físico (Sim-to-Real) con garantías de seguridad matemática.

---

## Metodología de 4 Fases (Curriculum Pipeline)

Entrenar una LNN requiere un enfoque radicalmente distinto al de un Transformer estático, ya que las derivadas fluyen a través del tiempo continuo ($\Delta t$). Este pipeline sigue los estándares establecidos por MIT CSAIL.

### Fase 1: Pre-entrenamiento Sensorial (Representation Learning)
Antes de procesar dinámicas temporales, el sistema debe aprender a comprimir la realidad espacial.
*   **Mecánica:** Se congelan las células líquidas (`CfCCell`). Se entrenan únicamente los proyectores (`CNNProjector` y auto-modality heads) utilizando técnicas de auto-codificación (Autoencoders) o aprendizaje contrastivo.
*   **Objetivo:** Reducir la dimensionalidad de una imagen de cámara (ej. 1080p) o un barrido Lidar a un vector de latentes compacto sin perder información de profundidad.
*   **Fundamento:** Las ecuaciones diferenciales operan mejor sobre espacios de estado compactos y densos, no sobre píxeles en crudo.

### Fase 2: Clonación en Bucle Abierto (Behavioral Cloning con BPTT)
Enseñanza de los reflejos motores base mediante demostraciones humanas.
*   **Mecánica:** Se graban trayectorias expertas (humanos operando el robot en Isaac Sim o hardware físico). El modelo se entrena usando **Backpropagation Through Time (BPTT)** sobre ventanas de secuencia de tiempo. 
*   **Requisito Técnico:** Es **crítico** inyectar el tensor de $\Delta t$ exacto en cada paso de inferencia, ya que las celdas líquidas usan el tiempo como variable de compuerta.
*   **Objetivo:** El robot aprende a predecir la acción humana ideal dados los sensores.

### Fase 3: Inyección de Caos (Curriculum & Domain Randomization)
La ventaja principal de las LNN es su resiliencia natural a condiciones fuera de distribución (Out-of-Distribution - OOD). Esta fase fuerza el desarrollo de esa resiliencia.
*   **Mecánica:** Se somete al modelo a un "Plan de Estudios" progresivo dentro del simulador:
    *   *Nivel 1:* Entorno ideal.
    *   *Nivel 2 (Sensor Dropout):* Se apaga aleatoriamente el 10%-30% de los rayos Lidar.
    *   *Nivel 3 (Física):* Se aleatoriza la fricción del suelo, la masa del robot y la latencia del bus de control.
*   **Objetivo:** Las neuronas líquidas aprenden a no depender de un sensor ruidoso, equilibrando internamente sus constantes de tiempo para mantener la inercia del movimiento.

### Fase 4: Calibración de Seguridad y RL (Closed-Loop + CBF)
Pulido final del modelo interactuando en tiempo real con el simulador y el escudo de seguridad.
*   **Mecánica:** El robot se despliega en Isaac Sim usando Aprendizaje por Refuerzo (RL) como *Soft Actor-Critic*. Si la red neuronal propone una acción peligrosa, el **OmniShield** (Control Barrier Function) proyecta la acción a la zona segura.
*   **Entrenamiento:** Se penaliza a la red (`Barrier Loss`) cuando obliga al escudo a intervenir. 
*   **Objetivo:** Un robot que es agresivamente eficiente pero se auto-corrige fracciones de segundo antes de acercarse al límite físico de colisión.

---

## 📚 Referencias y Fuentes Oficiales (MIT)

El pipeline detallado anteriormente se basa en metodologías comprobadas y publicadas por el grupo de Robótica e Inteligencia Artificial del MIT:

1.  **Arquitectura CfC y Velocidad de Inferencia:**
    *   *Paper:* "Closed-form continuous-time neural networks"
    *   *Publicación:* Nature Machine Intelligence (2022).
    *   *Autores:* Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, et al.
    *   *Relevancia:* Establece la base matemática para computar derivadas líquidas sin los lentos *ODE Solvers*, habilitando el entrenamiento BPTT rápido para la Fase 2.

2.  **Robustez OOD y Navegación de Drones (Sim-to-Real):**
    *   *Paper:* "Robust flight navigation out of distribution with liquid neural networks"
    *   *Publicación:* Science Robotics (2023).
    *   *Autores:* Makram Chahine, Ramin Hasani, et al.
    *   *Relevancia:* Demuestra la efectividad de la Fase 3 (Domain Randomization), logrando que drones entrenados en ambientes limpios naveguen exitosamente en bosques densos sin re-entrenamiento.

3.  **Concepto Original (LTC):**
    *   *Paper:* "Liquid Time-Constant Networks"
    *   *Publicación:* AAAI (2021).
    *   *Relevancia:* Define la ecuación fundamental donde las constantes de tiempo y los retardos se adaptan a la entrada sensorial.

---
*OmniTrain Project Documentation - 2026*
