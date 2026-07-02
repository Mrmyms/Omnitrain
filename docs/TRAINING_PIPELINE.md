# Training Pipeline: Conectoma v2.1

This document defines the official recommended methodology for training autonomous robotic systems powered by **Liquid Neural Networks (LNN)** and **Closed-form Continuous-time (CfC)** architectures. The v2.1 methodology introduces enterprise-grade stability for critical deployments.

---

## Training Methodology (Curriculum)

Training an LNN requires a radically different approach than a static Transformer, as derivatives flow across continuous time ($\Delta t$). This pipeline integrates formal stability and data parity.

### Phase 1: Sensory Pre-training and Data Parity
Before processing temporal dynamics, the system must learn to normalize reality.
*   **Statistics Capture (v2.1):** During dataset loading (`OmniLogDataset`), the system automatically captures the mean and standard deviation of each sensor.
*   **Importance:** These statistics are saved in the `.omni` bundle. Without them, the robot would suffer from "data degradation" upon receiving raw values in real-time that do not match the training distribution.

### Phase 2: Behavior Cloning (Stateful BPTT)
Teaching base motor reflexes through human demonstrations.
*   **Mechanics:** Training is **Stateful**. The latent state of the brain is propagated between contiguous sequences of a trajectory, allowing the robot to learn long-range temporal dependencies (e.g., remembering it passed through a door 10 seconds ago).

### Phase 3: Chaos Injection (Domain Randomization)
The main advantage of LNNs is their natural resilience to Out-Of-Distribution (OOD) conditions.
*   **Mechanics:** Gaussian noise and sensor failures (dropout) are injected. 
*   **Technical Note:** Noise is applied **after** Z-Score normalization but **before** activation clamping, allowing the network to learn to ignore noisy signals without saturating.

### Phase 4: Lagrangian Stability (Formal Safety)
Final polishing of the model with mathematical safety guarantees.
*   **Lagrangian Dual Update (v2.1):** A primal-dual optimizer is used to adjust the weight of safety. Updates to the Lagrange multiplier ($\lambda$) are performed **per sequence**, eliminating the violent oscillations of previous versions and achieving a much more stable safety policy.
*   **OmniShield (RK4 Consistency):** Guarantee of mathematical parity between training and inference by using Runge-Kutta 4 integration at all levels of the shield.

### Phase 5: Synaptic Consolidation (Structured Pruning)
Post-training optimization for low-power hardware (Edge Computing).
*   **Mechanics:** The weakest neurons and connections are structurally removed. This reduces the model size by up to 60% and decreases latency on NVIDIA Jetson or Qualcomm devices.

---

## References and Official Sources (MIT)

1.  **CfC Architecture:** Nature Machine Intelligence (2022). Ramin Hasani, et al.
2.  **OOD Robustness:** Science Robotics (2023). Makram Chahine, Ramin Hasani, et al.
3.  **Formal Safety:** ICNN-based Control Barrier Functions (2021).

---
*OmniTrain Project Documentation - 2026 (v2.1)*
