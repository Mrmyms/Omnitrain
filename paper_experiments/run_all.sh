#!/bin/bash
set -e

echo "============================================================"
echo " OmniTrain: One-Click 500% Reproducibility Pipeline"
echo "============================================================"
echo "[1/7] Initializing virtual environment and dependencies..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Ensure data directory exists
mkdir -p data

echo -e "\n[2/7] Generating the Processor-in-the-Loop CartPole Dataset..."
python simulate_pendulum.py --samples 5000

echo -e "\n[3/7] Training Baseline and CfC Architectures (Evaluating Temporal Jitter)..."
python train_and_compare.py

echo -e "\n[4/7] Running LSTM Data Augmentation Ablation Study..."
python lstm_ablation_augmentation.py --loss_rate 0.20 --test_loss 0.20

echo -e "\n[5/7] Running CfC Generalization Stress-Test (Friction + Gaussian Noise)..."
python test_generalization_friction.py

echo -e "\n[6/7] Generating Visualizations and Plots..."
python plot_results.py
python plot_timeseries.py

echo -e "\n[7/7] Validating INT8 Quantization Stability (Silicon Parity Test)..."
python test_int8_loss.py

echo "============================================================"
echo " [SUCCESS] All experiments reproduced successfully!"
echo " Outputs have been saved to the 'data/' and '../ai_context/' directories."
echo "============================================================"
