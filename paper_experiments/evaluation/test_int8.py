import numpy as np
import torch
import torch.nn as nn
from train_and_compare import ContinuousCfC, train_model, evaluate_model
import sys

# Support Mac ARM
torch.backends.quantized.engine = 'qnnpack'

# Load data
X_0 = np.load("../data/pendulum_X_0loss.npy")
Y = np.load("../data/pendulum_Y.npy")
T = np.load("../data/pendulum_T.npy")

print("Training CfC model for 10 epochs (fast)...")
torch.manual_seed(42)
hidden_dim = 16
model = ContinuousCfC(4, hidden_dim, 1)
model = train_model(model, X_0, Y, T, epochs=10)

mse_fp32 = evaluate_model(model, X_0, Y, T)
print(f"FP32 MSE: {mse_fp32:.4f}")

print("Applying INT8 Dynamic Quantization...")
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

try:
    mse_int8 = evaluate_model(quantized_model, X_0, Y, T)
    print(f"INT8 MSE: {mse_int8:.4f}")
except Exception as e:
    print(f"Quantization evaluation failed: {e}")
