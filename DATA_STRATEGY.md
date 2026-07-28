# OmniTrain: Data Strategy for Rewritten Paper
## IEEE Embedded Systems Letters (Mejorado)

---

## 1. DATASETS REQUERIDOS

### 1.1 PRIMARY: F1TENTH Autonomous Racing
**Propósito:** Validación HIL en tiempo real

#### Formato: F1TENTH Telemetry Bundle (.tar.gz)
```
f1tenth_telemetry/
├── README.md (metadata, sampling rate, hardware version)
├── lidar_sequences/
│   ├── closed_loop_001.bin (25 rays × float32, 20 Hz, 5000 steps)
│   ├── closed_loop_002.bin
│   ├── ... (min 5 complete laps)
│   └── manifest.json (timestamps, velocities, collisions)
├── ground_truth/
│   ├── steering_commands.npy (5000×1, commanded steering angle)
│   ├── velocity_commands.npy (5000×1, commanded velocity)
│   └── collision_timestamps.npy (indices where collision occurred)
├── imu_data/
│   └── accelerometer_gyro.bin (9-DOF, int16, synchronized with LiDAR)
└── metadata.json
   {
     "hardware": "ESP32-S3 @ 240 MHz",
     "circuit_length_m": 28.0,
     "sampling_frequency_hz": 20,
     "total_sequences": 5,
     "total_steps": 25000,
     "lidar_ray_count": 25,
     "range_min_m": 0.15,
     "range_max_m": 5.0
   }
```

**Por qué es crítico:**
- Real hardware timing variability
- Validación de jitter resilience (Theorem 1)
- Direct comparison vs LSTM/GRU crashes

---

### 1.2 SECONDARY: Synthetic Training Dataset (Isaac Lab)
**Propósito:** Generación escalable de trajectories sintéticas

#### Formato: Isaac Gym XIP Training Corpus
```
isaac_training_corpus/
├── simulation_config.yaml (physics params, friction, mass)
├── trajectories_train/
│   ├── 00000.h5 (HDF5 - efficient para ML)
│   │   ├── /lidar (shape: [10000, 25], dtype: float32)
│   │   ├── /steering (shape: [10000, 1], dtype: float32)
│   │   ├── /velocity (shape: [10000, 1], dtype: float32)
│   │   ├── /rewards (shape: [10000], dtype: float32)
│   │   └── /metadata (JSON string con seed, physics variation)
│   ├── 00001.h5
│   └── ... (10,000+ trajectories)
├── trajectories_val/
│   └── ... (2,000 trajectories, disjoint seeds)
└── statistics.json
   {
     "train_samples": 10000000,
     "val_samples": 2000000,
     "lidar_mean": [2.34, 2.41, ...],  // por ray
     "lidar_std": [0.67, 0.71, ...],
     "steering_range": [-1.0, 1.0],
     "velocity_range": [0.0, 2.5],
     "collision_rate_raw": 0.032
   }
```

**Datos sintéticos con variación física:**
- 50 diferentes configuraciones de fricción
- 10 variaciones de masa del robot
- 20 degradaciones de sensor (ruido LiDAR Gaussiano σ = 0.05-0.2)

---

### 1.3 TERTIARY: Quantization Analysis Dataset
**Propósito:** Demostrar Theorem 1 (PTQ Collapse)

#### Formato: INT8 Grid Analysis Traces (.csv + .npz)

```
ptq_analysis/
├── weight_distributions.npz
│   ├── time_gate_activations (shape: [4000, 1000], float32)
│   │   // 4000 neurons × 1000 timesteps from trained FP32 model
│   ├── sensory_projections (shape: [25, 1000], float32)
│   ├── inter_neuron_weights (shape: [667, 1000], float32)
│   └── command_layer (shape: [2, 1000], float32)
│
├── quantization_artifacts.csv
│   // Columns: layer_idx, weight_name, fp32_min, fp32_max, 
│   //          int8_scale, dead_zone_threshold, 
│   //          % weights in dead zone, rank
│
├── gate_dynamics_comparison.h5
│   ├── /fp32_gate (shape: [1000], dtype: float32)
│   ├── /int8_gate_std_ptq (shape: [1000], dtype: float32)
│   ├── /int8_gate_qat (shape: [1000], dtype: float32)
│   └── /timestamps (shape: [1000], dtype: float32)
│
└── failure_modes.json
   {
     "ptq_collapse_step": 47,
     "gate_variance_fp32": 0.0187,
     "gate_variance_int8_ptq": 1.2e-6,
     "gate_variance_int8_qat": 0.0156,
     "theoretical_dead_zone_width": 0.016,
     "empirical_dead_zone_hits": 0.823
   }
```

---

### 1.4 QUATERNARY: Sensor Jitter & Packet Loss Synthetic
**Propósito:** Validar robustez temporal (Table IV)

#### Formato: Sensor Degradation Scenarios (.pkl)

```
jitter_scenarios/
├── baseline_perfect.pkl
│   └── 30 random seeds × 5000 steps each
│
├── packet_loss_20pct.pkl
│   └── Zero-Order-Hold interpolation (synthetic dropout)
│   └── 30 random seeds × 5000 steps each
│
├── packet_loss_60pct.pkl
│   └── 30 random seeds × 5000 steps each
│
├── temporal_jitter_uniform.pkl
│   └── Δt ∈ [0.04, 0.06] (±20% variation around 20 Hz)
│
└── scenario_metadata.json
   {
     "baseline": {
       "packet_loss_rate": 0.0,
       "temporal_jitter_std_ms": 0.0,
       "total_episodes": 30,
       "mean_steps_to_failure": null
     },
     "packet_loss_20pct": {
       "packet_loss_rate": 0.2,
       "zoh_strategy": "extrapolate_last_valid",
       "total_episodes": 30,
       "mean_steps_to_failure_lstm": 500,
       "mean_steps_to_failure_gru": 500,
       "mean_steps_to_failure_cfc": 500
     },
     "packet_loss_60pct": {
       "packet_loss_rate": 0.6,
       "total_episodes": 30,
       "mean_steps_to_failure_lstm": "50.0 ± 22.6",
       "mean_steps_to_failure_gru": "108.8 ± 136.3",
       "mean_steps_to_failure_cfc": "257.3 ± 202.7"
     }
   }
```

---

### 1.5 QUINARY: Architecture Ablation Study
**Propósito:** Justificar 20-10-10 NCP connectivity

#### Formato: Multi-Model Performance Matrix (.json + .xlsx)

```
ablation_study/
├── results_matrix.json
│   {
│     "experiments": [
│       {
│         "arch_name": "Dense-CfC",
│         "parameters": 4000,
│         "sparsity": 0.0,
│         "precision": "FP32",
│         "fitness_mean": 22500,
│         "fitness_std": 1200,
│         "sram_kb": 16.5,
│         "flash_kb": 64.2,
│         "latency_ms": 8.12,
│         "hardware_platform": "ESP32-S3"
│       },
│       {
│         "arch_name": "NCP 20-10-20",
│         "parameters": 667,
│         "sparsity": 0.50,
│         "precision": "INT8",
│         "fitness_mean": 3176,
│         "fitness_std": 450,
│         "sram_kb": 6.1,
│         "flash_kb": 16.5,
│         "latency_ms": 6.15
│       },
│       {
│         "arch_name": "NCP 20-10-10",
│         "parameters": 270,
│         "sparsity": 0.75,
│         "precision": "INT8",
│         "fitness_mean": 21100,
│         "fitness_std": 890,
│         "sram_kb": 1.5,
│         "flash_kb": 13.8,
│         "latency_ms": 1.22
│       }
│     ]
│   }
│
├── sensory_compression_analysis.h5
│   ├── /dense_mapping (25→25)
│   │   ├── mse: 0.0
│   │   ├── latency_ms: 0.34
│   ├── /compressed_mapping (25→20)
│   │   ├── mse: 0.0095  // 39.5% reduction
│   │   ├── latency_ms: 0.28
│   └── /reconstruction_error_per_ray (shape: [25])
│
└── pareto_frontier.csv
   // columns: parameters, sparsity, fp32_fitness, int8_fitness, 
   //          sram_kb, latency_ms, pareto_dominated
```

---

### 1.6 SENARY: Numerical Validation Corpus
**Propósito:** Verificar paridad PyTorch ↔ Embedded C (1.22 ms latency proof)

#### Formato: Deterministic Rollout Traces (.npz)

```
numerical_validation/
├── pytorch_reference/
│   ├── weights_fp32.npz (original trained model)
│   ├── state_trajectories_1000steps.h5
│   │   ├── /hidden_state (shape: [1000, 15], dtype: float32)
│   │   ├── /output (shape: [1000, 2], dtype: float32)
│   │   └── /input_lidar (shape: [1000, 25], dtype: float32)
│   └── state_trajectories_quantized_int8.h5
│       └── (same structure, INT8 casted to FP32 in ALU)
│
├── esp32_firmware/
│   ├── hardware_trace_1000steps.csv
│   │   // Columns: step_idx, hidden_state_0, hidden_state_1, ...,
│   │   //          output_0, output_1, timestamp_us
│   └── firmware_version.txt
│
└── validation_report.json
   {
     "max_absolute_deviation_hidden_state": 1.2e-4,
     "max_absolute_deviation_output": 8.7e-5,
     "pytorch_latency_ms": 1.19,
     "esp32_latency_ms": 1.22,
     "latency_overhead_percent": 2.5,
     "bit_exact_match": false,
     "numerical_stability_pass": true,
     "validation_seeds": 10
   }
```

---

## 2. DATA COLLECTION PROTOCOL

### 2.1 F1TENTH Hardware Collection
```bash
# Pseudocode de protocolo de recolección
1. Calibrate LiDAR (25-ray downsampling, 20 Hz clock sync)
2. Record 5 complete lap sequences (no crashes)
   - Pre-generate velocity profile
   - Validate IMU sync within ±2 ms
   - Save raw .bin files
3. Inject policy (LSTM/GRU/CfC) in closed-loop
4. Monitor collisions (ground truth via replay)
5. Verify timestamp monotonicity + no frame drops
6. Compress to .tar.gz with manifest
```

### 2.2 Isaac Lab Corpus Generation
```python
# Pseudo-code para 10,000+ trajectories
for seed in range(10000):
    env = IsaacGymEnv(
        friction_mu=sample_gaussian(0.7, 0.15),
        mass_robot=sample_gaussian(3.5, 0.4),
        lidar_noise_std=sample_uniform(0.05, 0.2)
    )
    trajectory = env.rollout(
        policy=behavioral_cloning_oracle,
        max_steps=1000,
        seed=seed
    )
    trajectory.to_hdf5(f"trajectories_train/{seed:06d}.h5")
    
    # Log statistics per ray, per timestep
    accumulate_statistics(trajectory)

# Finalize normalization constants
compute_and_save_statistics()
```

---

## 3. DATA AVAILABILITY & REPRODUCIBILITY

### 3.1 GitHub Artifact Structure
```
https://github.com/mrmyms/OmniTrain
└── data/
    ├── README.md (dataset usage guide)
    ├── download_all.sh (DVC or Zenodo links)
    ├── f1tenth_telemetry.tar.gz (≈500 MB)
    ├── isaac_training_corpus.tar.gz (≈2.1 GB, optional)
    ├── ptq_analysis.tar.gz (≈50 MB)
    ├── jitter_scenarios.tar.gz (≈100 MB)
    └── ablation_study.json (≈2 MB)
```

### 3.2 Zenodo/OSF Registration
- **DOI assigned** para citación permanente
- Versioning por release del paper
- Checksum validation (SHA256) para integridad

---

## 4. DATA FORMAT RATIONALE

| Dataset | Format | Why? |
|---------|--------|------|
| F1TENTH Telemetry | `.bin` + `.json` | Raw hardware traces; compact; reproducible timestamps |
| Training Corpus | `HDF5` | Efficient I/O para ML; hierarchical; built-in compression |
| Quantization Analysis | `.npz` + `.csv` | NumPy interop; human-readable comparison metrics |
| Jitter Scenarios | `.pkl` | Pickle preserves Python objects; includes randomness seeds |
| Ablation Results | `.json` | Human-readable; direct inclusion en paper como tablas |
| Numerical Validation | `.npz` + `.csv` | Direct PyTorch ↔ ESP32 bit-level comparison |

---

## 5. MINIMUM VIABLE DATASET (MVP)

Si el tiempo/storage es limitado, estos son los **imprescindibles**:

1. **F1TENTH Telemetry** (3 laps completas, ≈250 MB)
2. **PTQ Analysis Dataset** (demostración de Theorem 1, ≈50 MB)
3. **Ablation Study Results** (JSON, ≈2 MB)
4. **Jitter Packet Loss Scenarios** (síntesis, ≈50 MB)

**Total MVP: ≈350 MB** → Publicable en Zenodo con acceso abierto

---

## 6. PROPOSED UPDATES TO PAPER SECTIONS

### 6.1 Nueva sección: "IV.A Data Availability"
```latex
\subsubsection{Data Availability}
All datasets, code, and trained models are publicly available
at https://doi.org/10.5281/zenodo.XXXXXXX to support 
reproducibility. The F1TENTH telemetry corpus contains 5 
complete lap sequences (25K timesteps, 20 Hz); the quantization 
analysis traces document the PTQ collapse phenomenon across 
4,000 neurons; the jitter scenarios validate temporal resilience 
under 0\%, 20\%, and 60\% packet loss. Training was conducted 
using NVIDIA Isaac Lab v1.4 with 10,000+ synthetic trajectories 
generated across 50 friction and 10 mass variations.
```

### 6.2 Enhanced Table III (hardware performance con data pedigree)
```
\begin{table}[!h]
\caption{F1TENTH Hardware-in-the-Loop Performance (ESP32-S3).
Telemetry from \textit{n}=5 complete lap sequences per architecture.
Ground truth collisions verified post-hoc.}
...
```

---

## 7. QUALITY METRICS FOR DATASETS

- ✅ **Completeness:** 100% timestep coverage, no missing values
- ✅ **Synchronization:** LiDAR ↔ IMU ↔ motor commands ≤ 2ms skew
- ✅ **Reproducibility:** All seeds logged; simulation configs versioned
- ✅ **Validation:** Checksums SHA256 para cada archivo
- ✅ **Documentation:** Metadata JSON con unidades, rangos, hardware version

---

## 8. STORAGE COST ESTIMATE

| Dataset | Size | Justification |
|---------|------|---------------|
| F1TENTH HIL | 250 MB | 5 laps × 25 rays × float32 × 5000 steps × overhead |
| Isaac Training | 2.1 GB | 10K trajectories × 1000 steps × compression ratio 0.85 |
| PTQ Analysis | 50 MB | Weight distributions + gate dynamics traces |
| Jitter Scenarios | 100 MB | 30 seeds × 5000 steps × 6 degradation levels |
| Ablation Study | 2 MB | JSON + CSV results |
| Validation Corpus | 75 MB | PyTorch + ESP32 traces, 1000 steps × multiple seeds |
| **TOTAL** | **≈2.5 GB** | **Comprimido: ≈700 MB** |

---

## 9. NEXT STEPS (ACCIÓN)

1. **Recolectar F1TENTH**: 5 laps nuevas con timestamp logging exacto
2. **Exportar pesos quantizados**: Guardar INT8 activations de la 4K dense model
3. **Reproducir jitter scenarios**: Resynthesizar con random seeds documentados
4. **Crear Zenodo release**: DOI permanente previo al paper submission
5. **Validar bit-level paridad**: Ejecutar 1000-step rollouts ESP32 vs PyTorch
6. **Documentar metadata.json exhaustivo**: Unidades, calibración, hardware versions

---
