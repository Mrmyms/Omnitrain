# OmniTrain: Capabilities & Features Specification
**Official Capabilities Paper** | **Version 1.0.0**

---

## Executive Summary
OmniTrain is a high-performance framework for industrial robotics, designed to unify **Continuous-time Intelligence** with **Formal Safety Verification**. This document provides an exhaustive list of the system's capabilities across its core modules: AI, Safety, Transport, and Deployment.

---

## 1. Core Artificial Intelligence (The BioLiquid Brain)
OmniTrain moves beyond discrete-step neural networks to continuous-time reasoning and dynamic physical adaptation.

- **BioLiquid Neural Networks (LTC + CfC):** 
    - Real-time adaptation of time-constants based on sensory input.
    - Official MIT Liquid Time-constant (LTC) constraints (Softplus mapping) to guarantee mathematical stability.
    - Sub-millisecond latency via Closed-form Continuous-time (CfC) equations.
- **Continual Learning (Hebbian Plasticity):**
    - Post-deployment adaptation without backpropagation.
    - Dynamic synapse rewiring (Oja's Rule) for "Sim-to-Real" wear-and-tear compensation.
- **Multi-Brain Hub (Modular NCPs):**
    - Interconnected Neural Circuit Policies (e.g., Perception, Pilot, Safety modules).
    - Global Latent Bus for organ-to-organ communication and explainable reasoning.
- **Continuous Temporal Encoding (CTE):**
    *   **[Verified Active]** High-dimensional sinusoidal mapping of absolute arrival timestamps.
    *   Support for asynchronous sensor fusion (Lidar at 20Hz, Camera at 60Hz, IMU at 1000Hz).

---

## 2. Safety & Verification (OmniShield)
Safety is not a heuristic in OmniTrain; it is a mathematical guarantee.

- **OmniShield v2 Barrier Layers:**
    - Input Convex Neural Networks (ICNN) for learning safe operating envelopes.
    - Deterministic projection of unsafe control signals into safe sets ($\mathcal{C}$).
- **Formal Verification:**
    - Automated safety audits for trained models.
    - Interval-based constraint checking for sensor ranges.
- **Tiered Emergency System:**
    - **Tier 1:** Neural-safe control.
    - **Tier 2:** Formal CBF (Control Barrier Function) projection.
    - **Tier 3:** Hard-coded emergency fallback for system failure.

---

## 3. Infrastructure & Transport (TokenBus)
Zero-latency communication for real-world robotics.

- **TokenBus C++ Engine:**
    - POSIX Shared Memory transport for sub-millisecond IPC.
    - Lock-free circular buffers for zero-copy data transfer.
- **Universal Modality Plugins:**
    - Native support for Lidar, Vision, Scalar, and Vector data.
- **Real-time Monitoring:**
    - Live pulse-based monitoring of all sensor streams via CLI.

---

## 4. Robotics Ecosystem Integration (ROS 2 & Sim)
OmniTrain is designed to live within the modern robotics stack.

- **Native ROS 2 Humble/Iron Support:**
    - **Ingest Bridge:** Automated subscription to ROS 2 topics with zero-copy injection into the C++ bus.
    - **Output Bridge:** High-frequency (100Hz+) publication of predictions back to the ROS ecosystem.
    - **Thread Isolation:** Deterministic separation of ROS communication and AI inference.
- **NVIDIA Isaac Sim / Omniverse:**
    - Direct bridge for synthetic data generation and digital twin training.
    - Support for Isaac Gym reinforcement learning environments.

---

## 5. Development Workflow & CLI
A developer-first experience modeled after modern technical interfaces.

- **Interactive Console:**
    - Slash-command based REPL with autocompletion.
    - Rounded, industrial-grade dashboards using the `rich` library.
- **Curriculum-Based Training:**
    - 3-phase automated training: **Imitation** → **Safety** → **Chaos**.
    - Integrated data recording and management.
- **Project Scaffolding:**
    - `/init` command to generate ready-to-use robotics project archetypes.
    - Interactive YAML configuration editor.

---

## 6. Deployment & Industrial Integration
Seamless transition from training to the factory floor.

- **Hardware Acceleration Cascade:**
    - Automated provider selection: **NVIDIA DLA** → **TensorRT** → **CUDA** → **CPU**.
- **Edge-Ready Payload:**
    - Export to `.omni` bundles or optimized **ONNX** graphs.
    - Integrated quantization (INT8/FP16) for edge devices.
- **Diagnostic Suite:**
    - Automated "Health Checks" and industrialization tests.
    - Model inspection tools for architecture verification.

---

**© 2026 OmniTrain Research Group**
*"Fuse Everything. Trust Nothing. Verify Formally."*
