# OmniTrain Paper Experiments

This directory contains the exact code necessary to reproduce the data, training results, benchmarks, and visualizations presented in the paper *"Efficient Closed-Form Continuous-Time Neural Networks on Commodity Microcontrollers"*.

The experiment demonstrates a **Processor-in-the-Loop (PiL)** simulation of an Inverted Pendulum (CartPole), as well as physical **Hardware-in-the-Loop (HIL)** capabilities. It proves that continuous-time neural networks (CfC) can maintain predictive stability over lossy communication channels (temporal jitter) and unmodeled dynamic friction, whereas traditional discrete-time RNNs (LSTM, GRU) experience catastrophic failure.

## 🚀 One-Click Reproducibility (Recommended)

To guarantee "500% Reproducibility", we provide a unified shell script that will initialize a virtual environment, install dependencies, simulate the physics datasets, train all neural network architectures, run the ablation studies, and generate all final vector graphics and plots used in the paper.

```bash
chmod +x run_all.sh
./run_all.sh
```
*(All generated assets will be stored automatically in the `data/` and `../ai_context/` directories).*

---

## 🔬 Manual Step-by-Step Execution

If you prefer to audit or run the experiments individually, follow these steps:

### 1. Environment Setup

Install the necessary dependencies in a clean virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*(Note: The main `omnitrain` package must be present in the parent directory so `BioLiquidCell` can be imported).*

### 2. Generating the Dataset

Simulate the physics of the Inverted Pendulum to generate the baseline dataset. This script automatically injects artificial packet loss ($20\%$ and $60\%$).

```bash
python simulate_pendulum.py --samples 5000
```

### 3. Training the Models (Temporal Jitter Test)

Instantiate the LSTM, GRU, and CfC architectures. The script trains all models under ideal conditions ($0\%$ packet loss) and then evaluates their Mean Time-To-Failure (TTF) under the $20\%$ and $60\%$ packet loss regimes.

```bash
python train_and_compare.py
```

### 4. LSTM Data Augmentation (Ablation Study)

Run the ablation study demonstrating that even when the discrete LSTM is trained using Data Augmentation (explicitly exposed to 20% ZOH packet loss), it fails to generalize to the physical continuous-time dynamics.

```bash
python lstm_ablation_augmentation.py --loss_rate 0.20 --test_loss 0.20
```

### 5. CfC Generalization Stress-Test (Unmodeled Dynamics)

Run the rigorous Out-Of-Distribution (OOD) testing environment that injects unmodeled **Non-Linear Cart/Pole Friction** and **Gaussian Sensor Noise** into the closed-loop evaluation, proving that the CfC interpolates physics accurately regardless of idealized training data.

```bash
python test_generalization_friction.py
```

### 6. Visualizing Temporal Resilience

Read the experimental results and generate the exact vector/PNG plot included in the paper.

```bash
python plot_results.py
python plot_timeseries.py
```

### 7. INT8 Quantization Verification

Run the Dynamic Post-Training Quantization (PTQ) test to verify the Silicon Parity of the FP32 CfC model against typical INT8 deployment targets.

```bash
python test_int8_loss.py
```
