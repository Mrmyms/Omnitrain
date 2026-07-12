import numpy as np
import torch
import torch.nn as nn
from train_and_compare import ContinuousCfC, train_model, evaluate_model

torch.backends.quantized.engine = 'qnnpack'

X_0 = np.load("data/pendulum_X_0loss.npy")
X_60 = np.load("data/pendulum_X_60loss.npy")
Y = np.load("data/pendulum_Y.npy")
T = np.load("data/pendulum_T.npy")

torch.manual_seed(42)
hidden_dim = 16
model = ContinuousCfC(4, hidden_dim, 1)
model = train_model(model, X_0, Y, T, epochs=10)

mse_60_fp32 = evaluate_model(model, X_60, Y, T)
print(f"FP32 MSE (60% loss): {mse_60_fp32:.4f}")

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
mse_60_int8 = evaluate_model(quantized_model, X_60, Y, T)
print(f"INT8 MSE (60% loss): {mse_60_int8:.4f}")
