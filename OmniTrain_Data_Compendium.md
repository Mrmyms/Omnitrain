# OmniTrain — Compendio Completo de Datos e Investigación

> **Proyecto:** OmniTrain v2.1.0  
> **Autor:** Manuel Yobani Martinez Sanchez (Independent Researcher)  
> **Fecha de compilación:** 16 de Julio de 2026  
> **Repositorio:** [github.com/mrmyms/Omnitrain](https://github.com/mrmyms/Omnitrain)

---

## I. Arquitectura del Proyecto OmniTrain

OmniTrain es un framework end-to-end que permite entrenar redes neuronales de tiempo continuo (CfC / LTC / NCP) en PyTorch, exportarlas a un formato binario ultra-eficiente (`.omnibit`), y ejecutarlas en microcontroladores (ESP32-S3) sin dependencias externas ni asignación dinámica de memoria.

### Estructura de Directorios

```
Omnitrain/
├── src/
│   ├── omnitrain/                 # Librería principal de Python (pip install)
│   │   ├── __init__.py            # API pública v2.1.0
│   │   ├── fusion_core.py         # LiquidFusionCore - Motor de entrenamiento central
│   │   ├── sparse_cfc.py          # SparseCfC - Red dispersa con máscara de adyacencia
│   │   ├── trainer.py             # LiquidTrainer - Bucle de entrenamiento
│   │   ├── esp32_exporter.py      # Exportador a .omnibit (Dense + CSR Sparse)
│   │   ├── jetson_exporter.py     # Exportador a Jetson Nano (TensorRT)
│   │   ├── sdk.py                 # ProjectManager, LiquidTrainer, EdgeDeployer, AgentRunner
│   │   ├── omni_shield.py         # Sistema de seguridad para agentes autónomos
│   │   ├── omni_stream.py         # Streaming de datos en tiempo real
│   │   ├── token_bus.py           # Bus de comunicación inter-componentes
│   │   ├── quantize_omni.py       # Utilidades de cuantización INT8
│   │   ├── dataset.py             # Cargador de datasets para entrenamiento
│   │   ├── curriculum.py          # Entrenamiento por currículum
│   │   ├── async_fusion.py        # Fusión asíncrona de sensores
│   │   ├── diagnostics_and_monitoring.py
│   │   ├── environment_registry.py
│   │   ├── heads.py               # Cabezas de salida (steering, velocity, etc.)
│   │   ├── serial_logger.py       # Logger serial para ESP32
│   │   ├── launcher.py            # Lanzador de entrenamiento
│   │   ├── plugins.py             # Sistema de plugins
│   │   ├── snpe_runner.py         # Runner para Qualcomm SNPE
│   │   └── tensorrt_runner.py     # Runner para NVIDIA TensorRT
│   │
│   ├── cpp_engine/                # Motores de inferencia nativos en C++
│   │   ├── core/
│   │   │   ├── src/
│   │   │   │   ├── OmniEngine.cpp      # Motor CfC denso (Arch 0-3)
│   │   │   │   ├── OmniEngineLSTM.cpp  # Motor LSTM bare-metal
│   │   │   │   ├── OmniEngineGRU.cpp   # Motor GRU bare-metal
│   │   │   │   └── OmniEngineNCP.cpp   # Motor SparseCfC/NCP (Arch 4, CSR)
│   │   │   └── include/
│   │   ├── desktop/               # Wrapper para pruebas en PC
│   │   ├── esp32/                 # Implementación específica ESP-IDF
│   │   └── hal/                   # Hardware Abstraction Layer
│   │
│   └── omni_bus_core.cpp          # Bus de comunicación nativo (C++ -> Python binding)
│
├── hil_test/                      # Hardware-in-the-Loop para ESP32-S3
│   ├── include/
│   │   └── esp_omni_engine.hpp    # Motor optimizado Xtensa LX7 (IRAM_ATTR, CSR, ALIGN16)
│   ├── src/
│   │   ├── esp_omni_engine.cpp    # Implementación del motor con #pragma GCC unroll
│   │   ├── main.cpp               # Punto de entrada del firmware
│   │   └── OmniShield.cpp         # Capa de seguridad en hardware
│   ├── platformio.ini             # Configuración PlatformIO (ESP32-S3)
│   └── pc_hil_server.py           # Servidor HIL en Python (simula sensores desde PC)
│
├── paper_experiments/             # Scripts de todos los experimentos del paper
│   ├── train_f110_rl.py           # Entrenamiento RL en F1TENTH (Evolución)
│   ├── train_f110_rl_qat.py       # Entrenamiento QAT (Cuantización Evolutiva INT8)
│   ├── train_f110_ncp.py          # Entrenamiento supervisado del NCP ganador
│   ├── architecture_search_ncp.py # Búsqueda de Arquitectura (192 configuraciones)
│   ├── topology_search_ncp.py     # Búsqueda de Topología 3D
│   ├── train_and_compare.py       # Comparación CfC vs LSTM vs GRU (CartPole)
│   ├── simulate_f1tenth.py        # Simulación visual del agente F1TENTH
│   ├── evaluate_f1tenth.py        # Evaluación formal del agente
│   ├── train_final_showdown.py    # Comparación final multi-arquitectura
│   ├── test_int8_rl.py            # Test de cuantización INT8 en RL
│   └── data/                      # Datasets y resultados CSV
│
├── OmniTrain_Paper/               # Paper largo (IEEE Transactions, ~8 páginas)
│   └── main.tex
├── paper_draft.tex                # Paper corto (IEEE, ~4 páginas)
├── config.yaml                    # Configuración general del proyecto
└── README.md                      # Documentación pública
```

### API Pública de Python

```python
import omnitrain  # v2.1.0

# Clases principales expuestas:
# - omnitrain.ProjectManager       → Gestión de proyectos de entrenamiento
# - omnitrain.LiquidTrainer        → Bucle de entrenamiento PyTorch
# - omnitrain.EdgeDeployer         → Despliegue a hardware edge
# - omnitrain.AgentRunner          → Ejecución de agentes entrenados
# - omnitrain.LiquidFusionCore     → Motor central de fusión de sensores
# - omnitrain.ESP32Exporter        → Exportación a .omnibit
# - omnitrain.JetsonExporter       → Exportación a TensorRT
# - omnitrain.ESP32SerialLogger    → Logger serial para depuración
# - omnitrain.EnvironmentRegistry  → Registro de ambientes de simulación
# - omnitrain.SparseCfC            → Clase PyTorch de la red CfC dispersa
```

---

## II. Datos del Paper Original (CartPole / Péndulo Invertido)

### Configuración Experimental
| Parámetro | Valor |
|---|---|
| **Tarea** | Processor-in-the-Loop (PiL) Inverted Pendulum (CartPole) |
| **Hardware** | ESP32-S3 (Dual-Core Xtensa LX7 @ 240 MHz, 512 KB SRAM) |
| **Flash** | SPI Flash @ 80 MHz (Quad I/O, QIO) |
| **Software** | ESP-IDF 5.1.2, Arduino ESP32 Core 2.0.14, PlatformIO |
| **Compilador** | `xtensa-esp32s3-elf-g++` @ `-O2` (sin LTO) |
| **Framework de entrenamiento** | PyTorch 2.1.2 (CUDA 12.1) en NVIDIA RTX 5070 |
| **Secuencias** | 5,000 pasos @ 50 Hz (100 s de tiempo simulado) |
| **Seeds estadísticos** | 30 semillas independientes |
| **Pérdida de paquetes simulada** | 0%, 20%, 60% (Zero-Order Hold) |
| **Método estadístico** | Welch t-test (two-tailed), Holm-Bonferroni, Cohen's d |

### Modelos Comparados
| Arquitectura | Parámetros | Hidden Dim |
|---|---|---|
| LSTM | 1,425 | 16 |
| GRU | 1,073 | 16 |
| CfC (Full, 3-branch) | 2,273 | 16 |

### Hiperparámetros (Idénticos para todos)
| Parámetro | Valor |
|---|---|
| Optimizador | AdamW (lr=1e-3, weight decay=1e-4) |
| Pérdida | Huber Loss (δ=1.0) |
| Batch Size | 1 |
| Épocas | 50 (sin early stopping) |

### Resultados: Resiliencia Temporal (MSE de Fuerza de Control, Newtons)
| Arquitectura | 0% Loss | 20% Loss | 60% Loss |
|---|---|---|---|
| LSTM | 0.00005 ± 0.00003 | — | 0.00007 |
| GRU | — | — | 0.00007 |
| **CfC (Ours)** | **0.00002 ± 0.00002** | — | **0.00003** |

### Resultados: Time-to-Failure (Pasos, 500 = éxito completo)
| Arquitectura | 0% Loss | 20% Loss | 60% Loss |
|---|---|---|---|
| LSTM | — | — | 50.0 ± 22.6 |
| GRU | — | — | 108.8 ± 136.3 |
| **CfC (Ours)** | — | — | **257.3 ± 202.7** |

### Significancia Estadística (CfC vs Baselines, TTF)
| Comparación | Loss % | p_corr | Cohen's d |
|---|---|---|---|
| LSTM vs CfC | 0% | 0.001 | −1.03 |
| LSTM vs CfC | 20% | <0.001 | −1.16 |
| LSTM vs CfC | 60% | <0.001 | −1.41 |
| GRU vs CfC | 0% | 0.001 | −0.99 |
| GRU vs CfC | 20% | 0.001 | −1.02 |
| GRU vs CfC | 60% | 0.002 | −0.85 |

> Nota: Cohen's d negativo indica que CfC supera al baseline.

### Métricas de Hardware (ESP32-S3)
| Arquitectura | SRAM | Flash | Latencia | Poder |
|---|---|---|---|---|
| LSTM (TFLite Micro) | 42.5 KB | 64.2 KB | 8.12 ms | 105 mW |
| GRU (TFLite Micro) | 38.2 KB | 58.1 KB | 7.45 ms | 98 mW |
| LSTM-XIP (bare-metal) | 1.5 KB | 17.0 KB | 6.15 ms | 65 mW |
| GRU-XIP (bare-metal) | 1.4 KB | 16.0 KB | 5.80 ms | 62 mW |
| **CfC-XIP (Ours)** | **1.2 KB** | **16.5 KB** | **3.43 ms** | **52 mW** |

### Paridad Numérica FP32
- Diferencia máxima absoluta entre PyTorch y ESP32: **|δ|_max < 10⁻⁴**
- Secuencia evaluada: 1,000 pasos

### HIL (Hardware-in-the-Loop) Proof of Concept
- **Sensor:** MPU6050 (I2C, 6-axis acelerómetro/giroscopio)
- **Actuador:** Motor DC con driver PWM
- **MSE Angular:** 0.012 rad²
- **Nota:** Single-run (n=1), no es una caracterización estadística

---

## III. Datos del F1TENTH (Carreras Autónomas)

### Configuración de Entrenamiento
| Parámetro | Valor |
|---|---|
| **Tarea** | F1TENTH Autonomous Racing (Simulador Gym) |
| **Sensores** | LiDAR 1D, 25 rayos (downsampled de 1080) |
| **Salidas** | 2 (Ángulo de dirección, Velocidad) |
| **Método de Entrenamiento** | Distributed Evolution Strategy (ES) |
| **Generaciones** | ~1,000 |
| **Agentes paralelos** | Múltiples por generación |
| **GPU** | NVIDIA RTX 5070 |

### Arquitectura SparseCfC
| Parámetro | Valor |
|---|---|
| **Neuronas totales (H)** | 100 |
| **Dimensión de entrada (I)** | 25 (LiDAR rays) |
| **Dimensión de salida (O)** | 2 (steering, velocity) |
| **Sparsity (dispersión)** | 75% (solo 25% de sinapsis están activas) |
| **Tipo de máscara** | Adjacency matrix binaria (NCP connectome) |
| **Parámetros totales** | ~4,000 |

### Resultados de Entrenamiento F1TENTH
| Modelo | Precisión | Método | Top Fitness |
|---|---|---|---|
| MLP (Standard) | FP32 | Proximal Policy (PPO) | 14,200 |
| Dense CfC | FP32 | Evolution (ES) | 22,500 |
| Dense CfC | INT8 (PTQ) | Evolution + PTQ | **4,100** ⚠️ |
| **SparseCfC (Ours)** | **INT8 (QAT)** | **QAT Evolution** | **30,259** 🏆 |

> El modelo con PTQ colapsó por degradación de la time-gate σ(−f·Δt).  
> El modelo con QAT absorbió estructuralmente el ruido de cuantización y dominó.

### Hito Histórico
- **Distancia autónoma alcanzada:** >318 metros
- **"Wall of Death" (punto donde redes Dense FP32 fallaban):** 318 m
- **El SparseCfC INT8 QAT lo superó limpiamente.**

### Hardware Deployment (ESP32-S3)
| Métrica | TFLite Micro (Dense) | OmniEngine (SparseCfC) |
|---|---|---|
| Flash (Weights) | 48 KB | **14.2 KB** |
| SRAM Allocation | 120 KB (Arena) | **1.5 KB** (Buffers) |
| Inference Latency | 8.4 ms | **1.2 ms** |

---

## IV. Búsqueda de Arquitectura NCP (89-107 Modelos)

### Top 5: Menor Error Absoluto (MSE)
| Sensoriales | Proceso | Header | Densidad | Neuronas | Sinapsis | MSE |
|---|---|---|---|---|---|---|
| 20 | 100 | 50 | 10% | 170 | 1,983 | **0.0438** |
| 10 | 30 | 50 | 50% | 90 | 2,748 | **0.0449** |
| 10 | 10 | 50 | 50% | 70 | 1,728 | **0.0459** |
| 20 | 10 | 20 | 50% | 50 | 667 | **0.0473** |
| 20 | 30 | 20 | 25% | 70 | 709 | **0.0474** |

### Top 3: Pareto-Óptimos (Eficiencia Biológica)
| Sensoriales | Proceso | Header | Densidad | Neuronas | Sinapsis | MSE |
|---|---|---|---|---|---|---|
| **20** | **10** | **10** | **25%** | **40** | **270** | **0.0478** |
| 20 | 10 | 20 | 50% | 50 | 667 | 0.0473 |
| 10 | 30 | 50 | 10% | 90 | 530 | 0.1182 |

> **Ganador Pareto (20-10-10):** 40 neuronas, 270 sinapsis. Casi idéntico al conectoma de *C. elegans* (302 neuronas).

### Topologías Ganadoras (Entrenamiento Completo)
1. **Modelo Minimalista (Linear 20-10-10):** 40 neuronas, 25% density, MSE 0.0478
2. **Modelo Volumétrico (3D Array Cube 5×5×4):** 100 neuronas, 42% menos sinapsis, MSE 0.038

---

## V. Hallazgos Científicos Clave

### V.1 Efecto Information Bottleneck (Autoencoder)
- Poner N_sensory = N_inputs (25=25) **degrada** el MSE un 39.5%
- Reducir a N_sensory = 20 fuerza representaciones latentes comprimidas
- **Datos:** (20-10-10, 25%) = MSE 0.0478 vs (25-15-15, 25%) = MSE 0.0667

### V.2 Rendimientos Decrecientes en Escalamiento
| Transición | Neuronas añadidas | ΔMSE | Factor de Mejora (MSE/neurona) |
|---|---|---|---|
| 40 → 100 | +60 | 0.0075 | **1.25 × 10⁻⁴** |
| 100 → 200 | +100 | 0.0014 | **0.14 × 10⁻⁴** (10x peor) |

> **Conclusión:** El "punto dulce" para MCUs es 50-100 neuronas. Más allá, los retornos colapsan.

### V.3 Umbral Mínimo de Dispersión
- Densidad < 10%: gradientes mueren consistentemente (MSE plateau ~0.467)
- **Densidad óptima: 25%** (75% de sinapsis eliminadas, máximo rendimiento)

### V.4 La Regla de Oro de la Cuantización de ODEs
> **"La cuantización Post-Training (PTQ) destruye las compuertas temporales de las redes CfC. El término σ(−f(x,I)·Δt) es hipersensible: al comprimir los pesos de f(·) en 256 bins discretos, el producto se redondea a cero o satura, colapsando t_gate a un valor constante de 0.5 y aniquilando la memoria de tiempo continuo de la red."**

- **Dense CfC FP32 → INT8 PTQ:** 22,500 puntos → 4,100 puntos (colapso del 82%)
- **SparseCfC INT8 QAT:** 30,259 puntos (superó al FP32 por 34%)

---

## VI. Formato Binario `.omnibit`

### Estructura General
| Offset | Tamaño | Contenido |
|---|---|---|
| 0 | 5 bytes | Magic: `OMNI` + version byte |
| 5 | 1 byte | **Arch Flag:** 0=CfC, 1=GRU, 2=LSTM, **4=SparseCfC (CSR)** |
| 6-7 | 2 bytes | Padding |
| 8 | 24 bytes | Dimensiones: d_in, d_model, d_out, backbone_units, N_weights, N_tensors |
| 32 | N_t × 4 | Table of Contents (offsets de cada tensor) |
| 32 + N_t×4 | N_w × 4 | Blob de pesos contiguos (FP32, little-endian, 4-byte aligned) |

### Arch Flag 4 (SparseCfC) — Arrays CSR
| Tensor | Descripción |
|---|---|
| `bb_val` | Valores no-cero de la matriz backbone (float) |
| `bb_col` | Índices de columna (uint32) |
| `bb_row` | Punteros de fila (uint32) |
| `bb_b` | Bias del backbone (float) |
| `f_w`, `f_b` | Pesos/bias de la cabeza f (time-gate) |
| `g_w`, `g_b` | Pesos/bias de la cabeza g (candidate state) |
| `h_w`, `h_b` | Pesos/bias de la cabeza h (historical state) |
| `fc_w`, `fc_b` | Pesos/bias de la capa de salida lineal |

### Exportación Automática
El exportador (`esp32_exporter.py`) genera automáticamente:
1. Archivo `.omnibit` (binario para Flash)
2. Archivo `.h` (C-Header con array embebido para `#include` directo)

---

## VII. Motor C++ ESP32-S3 (Xtensa LX7)

### Optimizaciones Implementadas
| Optimización | Descripción |
|---|---|
| `IRAM_ATTR` | Funciones críticas cargadas en Instruction RAM (evita Flash cache misses) |
| `ALIGN16` | Buffers alineados a 128 bits para coincidir con el bus de datos del ESP32-S3 |
| `__restrict` | Punteros sin aliasing para habilitar vectorización del compilador |
| `#pragma GCC optimize ("O3, unroll-loops")` | Máxima optimización global |
| `#pragma GCC unroll 4` | Desenrollado de bucles internos |
| **CSR MatMul** | O(N_nonzero) en lugar de O(R×C) — salta 75% de operaciones |
| **Zero-Copy (mmap)** | Pesos leídos directo desde Flash DROM, 0 bytes de SRAM para pesos |

### Funciones del Motor
| Función | Atributo | Complejidad |
|---|---|---|
| `matmul()` | `IRAM_ATTR` | O(R × C) — Dense |
| `matmul_csr()` | `IRAM_ATTR` | O(N_nonzero) — Sparse |
| `Step()` | `IRAM_ATTR` | Punto de entrada de inferencia |
| `apply_sparse_cfc()` | `IRAM_ATTR` | Loop completo del ODE |
| `Load()` | — | Parser del .omnibit (Dense + CSR) |

### Buffers Estáticos (SRAM Total)
| Buffer | Tamaño | Propósito |
|---|---|---|
| `state_buffer_[256]` | 1 KB | Estado oculto h(t) |
| `latents_[256]` | 1 KB | Proyección de entrada |
| `b_state_[256]` | 1 KB | Backbone state |
| `x_in_[512]` | 2 KB | Concatenación [input; h(t)] |
| **Total worst-case** | **~5 KB** | (con buffers temporales) |

---

## VIII. Ecuaciones Matemáticas Fundamentales

### LTC ODE (Hasani 2021)
$$\frac{dx(t)}{dt} = -\left[\frac{1}{\tau} + f(x, I; \theta_f)\right]x(t) + A \cdot f(x, I; \theta_f)$$

### CfC Closed-Form Approximation (Hasani 2022)
$$x(t+\Delta t) = \sigma(-f \cdot \Delta t) \odot \tanh(g(x, I)) + [1 - \sigma(-f \cdot \Delta t)] \odot \tanh(h(x, I))$$

### SparseCfC Backbone (Ours)
$$\mathbf{b}(t) = \tanh\big( (\mathbf{W}_{bb} \odot \mathbf{M}) [\mathbf{x}(t) ; \mathbf{h}(t)] + \beta_{bb} \big)$$

### QAT Mutation (Ours)
$$\theta_{mutated} = \text{Clamp}\Big( \lfloor \frac{\theta + \sigma \mathcal{N}(0, I)}{\Delta_{q}} \rceil \times \Delta_{q}, -127, 127 \Big)$$

---

## IX. Referencias Bibliográficas

1. **Hasani et al. (2022)** — "Closed-form continuous-time neural networks," *Nature Machine Intelligence*, vol. 4, no. 11, pp. 992–1003.
2. **Hasani et al. (2021)** — "Liquid time-constant networks," *AAAI*, vol. 35, no. 9, pp. 7657–7666.
3. **Lechner et al. (2020)** — "Neural circuit policies enabling auditable autonomy," *Nature Machine Intelligence*, vol. 2, pp. 642–652.
4. **O'Kelly et al. (2020)** — "F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning," *CoRL*.
5. **David et al. (2021)** — "TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems," *MLSys*.
6. **Hubara et al. (2017)** — "Quantized neural networks: Training neural networks with low precision weights and activations," *JMLR*, vol. 18.
7. **Chen et al. (2018)** — "Neural ordinary differential equations," *NeurIPS*, pp. 6571–6583.
8. **Hochreiter & Schmidhuber (1997)** — "Long short-term memory," *Neural Computation*, vol. 9, no. 8.
9. **Cho et al. (2014)** — "Learning phrase representations using RNN encoder-decoder for statistical machine translation," *EMNLP*.
10. **Cohen (1988)** — *Statistical Power Analysis for the Behavioral Sciences*, L. Erlbaum Associates.

---

## X. Estado Actual y Próximos Pasos

### Completado ✅
- [x] Framework OmniTrain v2.1.0 (Python + C++)
- [x] Arquitectura SparseCfC con máscara NCP
- [x] Exportador `.omnibit` con CSR + auto-generación de C-Header
- [x] Motor C++ optimizado Xtensa LX7 (IRAM_ATTR, CSR, ALIGN16)
- [x] Búsqueda de Arquitectura (89-107 modelos evaluados)
- [x] Entrenamiento QAT Evolutivo (Top Fitness: 30,259)
- [x] Paper corto (4 páginas IEEE) — `paper_draft.tex`
- [x] Paper largo (8 páginas IEEE Transactions) — `OmniTrain_Paper/main.tex`

### Pendiente 🔲
- [ ] Validación Hardware-in-the-Loop (HIL) con ESP32-S3 físico (llega mañana)
- [ ] Medición real de latencia de inferencia con osciloscopio
- [ ] Medición real de consumo energético con INA219
- [ ] Compilar papers en Overleaf y generar PDFs finales
- [ ] Pruebas de soak test (estabilidad por horas)
