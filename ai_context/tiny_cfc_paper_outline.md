# Efficient Closed-Form Continuous-Time Neural Networks on Commodity Microcontrollers

**Target Venues:** ACM Transactions on Embedded Computing Systems (TECS) / ICML Workshop on Hardware-Aware Efficient Training (HAET) / TinyML Research Symposium.

---

## Abstract
Deploying sequence-modeling neural networks on deeply embedded hardware (TinyML) remains challenging due to strict memory constraints and the irregular sampling rates (jitter) inherent to physical sensors. While Recurrent Neural Networks (RNNs) like LSTMs are standard, they assume discrete, synchronous time steps and rely on heavy deployment frameworks like TensorFlow Lite for Microcontrollers (TFLite Micro), which incur significant SRAM overhead. In this paper, we demonstrate the native deployment of **Closed-form Continuous-time (CfC)** neural networks on commodity microcontrollers (ESP32). By utilizing a custom, static-memory Zero-Copy binary payload (`.omnibit`), we bypass traditional framework overhead. We provide a rigorous evaluation against TFLite Micro LSTMs, demonstrating superior resilience to sensor jitter, exact mathematical parity with PyTorch ($MSE < 10^{-6}$), lower inference latency, and a drastically reduced SRAM footprint. All code, datasets, and compilation tools are open-sourced for reproducibility.

---

## 1. Introduction
*   **The TinyML Challenge:** The explosion of IoT and edge robotics demands intelligence on sub-$5 microcontrollers (e.g., 512KB RAM). 
*   **The Problem with Discrete Time:** Real-world sensors suffer from packet loss, varying clock frequencies, and hardware interrupts. Traditional LSTMs fail mathematically under irregular time intervals unless computationally expensive interpolation is used.
*   **The Memory Bottleneck:** Standard frameworks (TFLite Micro) use dynamic arena allocation and schema-heavy formats (FlatBuffers), eating into the precious SRAM needed for the actual latent state.
*   **Our Contribution:** We bring the state-of-the-art CfC architecture to bare-metal microcontrollers using a custom Zero-Copy engine. We prove that CfCs are not only more robust to physical sensor realities but can be executed more efficiently than traditional LSTMs.

## 2. Background and Related Work
*   **Continuous-Time Neural Networks:** Discuss the foundational work of Ramin Hasani (MIT) on Liquid Time-Constant (LTC) and CfC networks. Note that while their parameter efficiency is known, their bare-metal hardware implementation characteristics remain under-explored.
*   **Embedded ML Frameworks:** Compare TFLite Micro, microTVM, and ST X-CUBE-AI. Explain why their generalized graph-execution engines introduce overhead for highly recurrent ODE-based models.

## 3. Methodology: Bare-Metal CfC Execution
This section details the engineering behind the open-source exporter and the C++ execution engine.

### 3.1 The CfC Mathematical Kernel
*   Formulate the core CfC update rule: 
    $$x(t) = I(x, t) \odot (1 - f(x, t)) + h(x, t) \odot f(x, t)$$
*   Explain how the time delta ($\Delta t$) is fed directly into the kernel, allowing native handling of irregular sampling.

### 3.2 The Zero-Copy Payload Architecture (`.omnibit`)
*   Describe the memory layout: A 24-byte header (Magic, Version, Architecture), followed by a deterministic Table of Contents (TOC), and a contiguous floating-point blob.
*   **Static Memory Mapping:** Explain how the ESP32 maps the Flash memory directly to the CPU cache (via `mmap` or ESP-IDF `spi_flash_mmap`), allowing the C++ engine to multiply weights *without* copying them into SRAM.

## 4. Experimental Setup
*   **Hardware:** ESP32-S3 (Xtensa Dual-Core 32-bit LX7, 512KB SRAM, no PSRAM used to ensure strict constraints).
*   **Baselines:** A standard LSTM model trained to parity and deployed via TensorFlow Lite for Microcontrollers (TFLite Micro).
*   **Task/Dataset:** A synthetic predictive maintenance or robotic kinematic task where time-series data is artificially corrupted with $0\%$, $20\%$, and $50\%$ jitter/packet-loss to test temporal resilience.

## 5. Results and Benchmarks
*(This section relies on clear, reproducible metrics)*

### 5.1 Temporal Resilience (Accuracy vs. Jitter)
*   **Metric:** Mean Absolute Error (MAE) on the test set.
*   **Analysis:** Show a line graph where the LSTM's accuracy crashes as packet loss increases, while the CfC maintains a stable error rate by intrinsically adjusting its ODE step via $\Delta t$.

### 5.2 Execution Latency
*   **Metric:** Microseconds ($\mu s$) per inference step.
*   **Analysis:** Compare the overhead of TFLite's `Invoke()` function (which traverses a node graph) versus the static, unrolled loops of the `OmniEngine` C++ CfC implementation.

### 5.3 Memory Footprint (Flash & SRAM)
*   **Metric:** Kilobytes (KB) utilized.
*   **Analysis:** 
    *   *Flash:* Compare the size of `.tflite` (FlatBuffers) vs `.omnibit`.
    *   *SRAM:* Compare TFLite's Tensor Arena allocation vs OmniEngine's purely static allocation of only the hidden state $h_t$ (Zero-Copy advantage).

### 5.4 Mathematical Parity
*   **Metric:** Mean Squared Error (MSE) between Python PyTorch (FP32) and ESP32 C++ (FP32).
*   **Analysis:** Prove that embedded optimization did not corrupt the mathematics ($MSE = 0.00e+00$), ensuring "what you train is exactly what you deploy."

## 6. Conclusion
*   **Summary:** We demonstrated that Continuous-Time Neural Networks are not just a theoretical novelty, but a highly practical solution for severely constrained embedded systems facing real-world sensory imperfections.
*   **Impact:** By removing the overhead of generalized ML frameworks and leveraging Zero-Copy memory semantics, researchers can deploy state-of-the-art Liquid architectures on commodity hardware.

## 7. Open Source Statement
*   "To foster further research in continuous-time edge AI, the complete PyTorch training pipeline, the ESP32 C++ inference engine, and the benchmark suite are provided as an open-source repository at [GitHub Link]."
