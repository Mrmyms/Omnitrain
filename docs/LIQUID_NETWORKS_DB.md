# Database: Liquid Neural Networks (LNN) 🧠💧

Este documento centraliza la investigación, especificaciones técnicas y fuentes oficiales de las Redes Neuronales Líquidas (LNN) y sus evoluciones (CfC), desarrolladas por el **MIT CSAIL**.

## 1. Definición y Fundamento Matemático
Las LNN son una clase de Redes Neuronales Recurrentes (RNN) de **tiempo continuo**. Se definen mediante ecuaciones diferenciales ordinarias (ODE) donde la derivada del estado oculto $h(t)$ depende no solo de la entrada $x(t)$ sino de una constante de tiempo dinámica:

$$\frac{dh(t)}{dt} = -[w_{sys} + w_{in} \cdot x(t)] \odot h(t) + w_{in} \cdot x(t)$$

Esto permite que la red tenga una "plasticidad" inherente, ajustando su velocidad de respuesta según la urgencia y variabilidad de la señal de entrada.

## 2. Hitos y Papers Oficiales

| Año | Paper / Hito | Publicación | Enlace/Referencia |
| :--- | :--- | :--- | :--- |
| **2021** | *Liquid Time-Constant Networks* | AAAI | [arXiv:2006.04439](https://arxiv.org/abs/2006.04439) |
| **2022** | *Closed-form Continuous-time (CfC)* | Nature | [Nature Machine Intelligence](https://www.nature.com/articles/s42256-022-00556-7) |
| **2023** | *Robust Flight Navigation (OOD)* | Science | [Science Robotics](https://www.science.org/doi/10.1126/scirobotics.adc9672) |
| **2024** | *Liquid Foundation Models (LFMs)* | Liquid AI | [Liquid.ai](https://www.liquid.ai/) |

## 3. Benchmarks de Rendimiento

### A. Eficiencia de Parámetros (Parsimonia)
En pruebas de conducción autónoma (lane-keeping):
*   **CNN/ResNet Tradicional:** Requieren >100,000 parámetros.
*   **Liquid (LTC):** Logró la misma precisión con solo **19 neuronas** y menos de **1,000 parámetros**.

### B. Robustez "Out-of-Distribution" (OOD)
En el paper de *Science Robotics (2023)*, drones equipados con LNN fueron entrenados en entornos simples y luego desplegados en:
*   Bosques densos.
*   Cambios de iluminación extrema.
*   Presencia de ruido visual masivo.
**Resultado:** Las LNN superaron a los Transformers y LSTMs en un **40% en tasa de éxito** de navegación en entornos desconocidos.

### C. Velocidad de Inferencia (CfC)
La arquitectura CfC elimina la necesidad de usar solvers de ODE (como Runge-Kutta), permitiendo:
*   Velocidades de inferencia **10x a 100x más rápidas** que las RNN continuas originales.
*   Consumo de memoria constante independientemente de la longitud de la secuencia (ideal para robots con poca RAM).

## 4. Comparativa Técnica Profunda

| Métrica | Transformers | LSTM / GRU | Liquid (CfC) |
| :--- | :--- | :--- | :--- |
| **Memoria** | Escala con el contexto ($N^2$) | Constante | **Constante (Ultra-baja)** |
| **Tiempo** | Discreto | Discreto | **Continuo** |
| **Adaptabilidad** | Baja (Requiere Fine-tuning) | Media | **Muy Alta (Inherente)** |
| **Hardware** | Requiere GPU/TPU | CPU / Mobile | **Microcontroladores / Edge** |
| **Interpretabilidad** | Casi nula (Caja Negra) | Baja | **Alta (Mecánica de Sistemas)** |

## 5. Arquitectura del "Cerebro Líquido"

1.  **Encoder:** Proyecta entradas sensoriales a un espacio de estado.
2.  **Liquid Core (LTC/CfC):** El motor dinámico que evoluciona el estado interno usando el "tiempo" como variable fundamental.
3.  **Gate Mechanism:** Filtra qué información sensorial es relevante para alterar la dinámica actual.

## 6. Recursos y Código
*   **Repositorio Oficial:** [MIT-LCP/LTC](https://github.com/raminhasani/liquid_time_constant_networks)
*   **Librería CfC:** [github.com/raminhasani/cfc](https://github.com/raminhasani/cfc)
*   **Liquid AI:** Empresa spinoff del MIT que está industrializando estos modelos para despliegue masivo.

---
*Este documento es parte de la base de conocimiento de OmniTrain para la transición hacia arquitecturas de tiempo continuo.*
