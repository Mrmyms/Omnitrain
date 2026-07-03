# Paper Structure Guide: Efficient CfC on Microcontrollers

Based on an extensive review of recent literature spanning Liquid Neural Networks, TinyML memory constraints, and embedded Zero-Copy execution, here is the definitive, section-by-section guide for your paper. 

**Target Audience:** TinyML Research Symposium / ACM TECS.
**Core Narrative:** *LSTMs are memory-heavy and struggle with physical sensor jitter. CfCs solve the jitter, but existing frameworks are too heavy for MCUs. Our Zero-Copy `.omnibit` architecture solves both.*

---

## 1. Abstract
*   **Goal (150-250 words):** Hook the reader immediately.
*   **What to write:**
    *   State that deploying sequence models on microcontrollers (MCUs) is blocked by severe SRAM limits (typically <512KB).
    *   Mention that physical sensors have irregular sampling rates (jitter), which breaks discrete-time RNNs/LSTMs.
    *   Introduce your solution: A native, bare-metal deployment of Closed-form Continuous-time (CfC) networks.
    *   Highlight the **Zero-Copy** memory mapping (`.omnibit`) that bypasses framework overhead (like TFLite Micro).
    *   End with the killer metrics: "Exact mathematical parity (MSE < 10^-6), X% less SRAM usage than TFLite, and robust performance under 60% packet loss."

## 2. Introduction
*   **Goal:** Establish the conflict (Memory vs. Time) and state your contributions.
*   **Paragraph 1 (The TinyML Boom):** Discuss the rise of edge robotics and IoT. Cite the need to process time-series data locally due to latency and privacy.
*   **Paragraph 2 (The Discrete-Time & Memory Problem):** Explain how LSTMs process data sequentially. Cite how maintaining hidden states and multiple gates exhausts SRAM. Explain the *Jitter* problem (sensors failing or delaying).
*   **Paragraph 3 (The CfC Promise):** Introduce Hasani et al. (2022) and their CfC solution, which is highly parameter-efficient and continuous. However, note a gap in the literature: *No one has optimized CfC for bare-metal execution without Python/heavy frameworks.*
*   **Paragraph 4 (Contributions):** Bullet points:
    1. A custom C++ ODE solver for CfCs on Cortex-M/Xtensa.
    2. A Zero-Copy binary format mapping Flash directly to CPU cache.
    3. Extensive benchmarking against TFLite Micro LSTMs.

## 3. Related Work
*   **Goal:** Prove you know the State-of-the-Art (SotA).
*   **Section 3.1 (Continuous-Time Networks):** Cite *Hasani et al. (2020)* on Liquid Time-Constant networks, and *Hasani et al. (2022)* on Closed-form Continuous-time networks. Mention that current implementations rely on high-level APIs (PyTorch).
*   **Section 3.2 (TinyML & Memory Constraints):** Cite literature on MCU memory bottlenecks. Discuss standard workarounds like INT8 Quantization and Pruning. Contrast your method by showing you solve the SRAM issue *architecturally* via Zero-Copy, not just by shrinking weights.
*   **Section 3.3 (Zero-Copy Inference):** Cite recent advancements (like PyTorch's *ExecuTorch* `TensorPtr`) that aim to eliminate CPU data copying. Explain how your `.omnibit` format achieves this on a much smaller scale (ESP32).

## 4. Methodology: The OmniEngine Architecture
*   **Goal:** Show the math and the hardware engineering.
*   **Section 4.1 (Mathematical Reformulation):** 
    *   Show the Hasani 2022 equation: $h_{t+1} = (1 - t_{int}) \cdot (g \cdot \tilde{h} + (1 - g) \cdot h_t) + t_{int} \cdot h_t$
    *   Explain how $\Delta t$ is fed into the system, intrinsically solving the sensor jitter problem.
*   **Section 4.2 (Zero-Copy Memory Mapping):**
    *   Include a diagram (can be text/ASCII) of the `.omnibit` format (Header -> TOC -> Float Blob).
    *   Explain how the ESP32 uses SPI Flash Memory Mapping (`mmap`) to read weights directly from the ROM, meaning the SRAM is *only* used for the hidden state ($h_t$), effectively dropping RAM usage to almost zero.

## 5. Experimental Setup
*   **Goal:** Prove the tests are fair and reproducible.
*   **Hardware:** Detail the ESP32-S3 (Dual-Core LX7, 512KB SRAM, no PSRAM).
*   **Baselines:** Describe the LSTM trained in TensorFlow and exported to TFLite Micro.
*   **Dataset:** Explain the synthetic dataset. Describe exactly how you injected 0%, 20%, and 60% packet loss to simulate real-world sensor failure.

## 6. Results and Evaluation
*   **Goal:** Present the metrics where OmniTrain crushes the baseline.
*   **Section 6.1 (Temporal Resilience):** Show a line chart comparing the Accuracy/MAE of LSTM vs CfC as packet loss increases. (CfC should stay flat, LSTM should crash).
*   **Section 6.2 (Memory Footprint):** Bar chart comparing SRAM usage. TFLite Micro (Tensor Arena allocation) vs OmniEngine (Static single-buffer allocation).
*   **Section 6.3 (Inference Latency):** Microseconds per step. OmniEngine C++ loop vs TFLite `Invoke()`.
*   **Section 6.4 (Silicon Parity):** Emphasize that the MCU C++ output matched the PyTorch FP32 output perfectly ($MSE = 0.00$), proving the custom engine doesn't sacrifice precision.

## 7. Conclusion & Future Work
*   **Goal:** Wrap up and look forward.
*   Summarize that CfCs are the superior choice for real-world TinyML due to their continuous nature, and that the Zero-Copy architecture makes them feasible on 5-dollar hardware.
*   **Future Work:** Mention adding INT8 quantization to the `.omnibit` format to halve the Flash footprint.

## 8. Open Source Statement
*   "All PyTorch exporter scripts, C++ engine code, and benchmark datasets are available at [GitHub Repo] to foster reproducibility in embedded continuous-time research."
