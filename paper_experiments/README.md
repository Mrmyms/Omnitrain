# OmniTrain Paper Experiments

This directory contains the exact code necessary to reproduce the data, training results, and visualizations presented in the paper *"Efficient Closed-Form Continuous-Time Neural Networks on Commodity Microcontrollers"*.

The experiment demonstrates a **Processor-in-the-Loop (PiL)** simulation of an Inverted Pendulum (CartPole), as well as physical **Hardware-in-the-Loop (HIL)** capabilities. It proves that continuous-time neural networks (CfC) can maintain predictive stability over lossy communication channels (temporal jitter), whereas traditional discrete-time RNNs (LSTM, GRU) experience catastrophic failure.

## 1. Environment Setup

To ensure exact reproducibility, install the necessary dependencies in a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Note: The main `omnitrain` package must be present in the parent directory so `BioLiquidCell` can be imported).*

## 2. Generating the Dataset

First, simulate the physics of the Inverted Pendulum to generate the baseline dataset. This script automatically injects artificial packet loss ($20\%$ and $60\%$) to simulate a degraded Serial/WiFi connection.

```bash
python simulate_pendulum.py --samples 5000
```
*Outputs: `data/pendulum_X_*.npy`, `data/pendulum_Y.npy`, `data/pendulum_T.npy`*

## 3. Training the Models

Instantiate the LSTM, GRU, and CfC architectures. The script trains all models under ideal conditions ($0\%$ packet loss) using AdamW and Huber Loss, and then strictly evaluates their Mean Squared Error (MSE) under the $20\%$ and $60\%$ packet loss regimes.

```bash
python train_and_compare.py
```
*Outputs: `data/results_mse.npy`*

## 4. Visualizing Temporal Resilience

Read the experimental results and generate the exact vector/PNG plot included in the paper, contrasting the resilience of the Execute-in-Place (XIP) CfC architecture against the TFLite baselines.

```bash
python plot_results.py
```
*Outputs: `data/temporal_resilience_chart.png`*

## 5. INT8 Quantization Verification & Time-Series Plot

You can also run the Dynamic Post-Training Quantization (PTQ) test to verify the Silicon Parity of the CfC model, and generate the time-series trajectory plot demonstrating temporal divergence:

```bash
# Evaluate PTQ INT8 performance under extreme jitter
python test_int8_loss.py

# Generate qualitative time-series comparison
python plot_timeseries.py
```
*Outputs: `../ai_context/timeseries_comparison.png`*
