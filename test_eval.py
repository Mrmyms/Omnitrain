import numpy as np
import torch
import sys
sys.path.append("src")
from paper_experiments.train_and_compare import ContinuousCfC, evaluate_closed_loop
model = ContinuousCfC(4, 16, 1, 32)
rng = np.random.RandomState(42)
x_mean = np.zeros(4)
x_std = np.ones(4)
res = evaluate_closed_loop(model, True, 0.0, rng, x_mean, x_std, 10)
print("Result:", res)
