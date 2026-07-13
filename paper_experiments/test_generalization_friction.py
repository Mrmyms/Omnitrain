import numpy as np
import torch
import torch.nn as nn
from train_and_compare import ContinuousCfC, train_model, is_failed
import argparse
import os

# --- CartPole Physics Constants (With Friction) ---
_GRAVITY = 9.8
_MASSCART = 1.0
_MASSPOLE = 0.1
_TOTAL_M = _MASSPOLE + _MASSCART
_LENGTH = 0.5
_POLEMASS_L = _MASSPOLE * _LENGTH
_DT = 0.02

# New Constants for Generalization Test
_FRICTION_CART = 0.1
_FRICTION_POLE = 0.05
_NOISE_STD = 0.01  # Gaussian sensor noise

def cartpole_step_complex(state: np.ndarray, force: float) -> np.ndarray:
    """Cartpole with Friction"""
    x, xd, th, thd = state
    cos_th, sin_th = np.cos(th), np.sin(th)
    
    # Friction terms
    force = force - _FRICTION_CART * np.sign(xd)
    pole_friction = _FRICTION_POLE * thd
    
    temp = (force + _POLEMASS_L * thd**2 * sin_th) / _TOTAL_M
    th_acc = ((_GRAVITY * sin_th - cos_th * temp - pole_friction) /
               (_LENGTH * (4.0/3.0 - _MASSPOLE * cos_th**2 / _TOTAL_M)))
    x_acc = temp - _POLEMASS_L * th_acc * cos_th / _TOTAL_M
    
    return np.array([x + _DT*xd, xd + _DT*x_acc, th + _DT*thd, thd + _DT*th_acc])

def evaluate_complex_env(model: nn.Module, drop_prob: float, 
                         x_mean: np.ndarray, x_std: np.ndarray, seeds=30, max_steps=500):
    model.eval()
    ttfs = []
    
    for seed in range(seeds):
        rng = np.random.RandomState(seed)
        # Start state
        state = np.array([0.0, 0.0, rng.uniform(-0.05, 0.05), 0.0], dtype=np.float32)
        held = state.copy()
        
        with torch.no_grad():
            h = torch.zeros(1, model._hsize)
            for step in range(max_steps):
                if rng.rand() >= drop_prob:
                    # Sensor Noise added ONLY on successful packet transmission
                    noise = rng.normal(0, _NOISE_STD, size=4)
                    held = state.copy() + noise
                
                held_n = (held - x_mean) / x_std
                xt = torch.FloatTensor(held_n).unsqueeze(0)
                dt = torch.FloatTensor([[_DT]])
                
                # CfC Forward pass
                bb = model.backbone(torch.cat([xt, h], dim=-1))
                t_gate = torch.sigmoid(-model.f_head(bb) * dt)
                h = t_gate * torch.tanh(model.g_head(bb)) + (1.0 - t_gate) * torch.tanh(model.h_head(bb))
                force = model.fc(h).item()
                
                # Step physics WITH FRICTION
                state = cartpole_step_complex(state, force)
                
                if is_failed(state):
                    ttfs.append(step)
                    break
            else:
                ttfs.append(max_steps)
                
    return np.mean(ttfs), np.std(ttfs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_loss", type=float, default=0.20, help="Packet loss prob during testing")
    args = parser.parse_args()
    
    print("[*] Loading clean cartpole trajectories...")
    X_clean = np.load("data/pendulum_X_0loss.npy")
    Y = np.load("data/pendulum_Y.npy")
    T = np.load("data/pendulum_T.npy")
    
    x_mean = np.mean(X_clean, axis=0)
    x_std = np.std(X_clean, axis=0) + 1e-8
    
    # Train CfC on purely clean, frictionless data
    print("[*] Training CfC on CLEAN data (no friction, no noise, no loss)...")
    model = ContinuousCfC(4, 32, 1, backbone_units=64) # Note: CfC doesn't concatenate DT
    
    # We need to pass NumPy arrays for training (train_model handles conversion)
    X_numpy = (X_clean - x_mean) / x_std
    
    model = train_model(model, X_numpy, Y, T, epochs=250)
    
    # Evaluate on complex environment
    print("\n[*] Evaluating CfC in COMPLEX environment (Friction + Gaussian Noise)...")
    mean_ttf, std_ttf = evaluate_complex_env(model, drop_prob=args.test_loss, x_mean=x_mean, x_std=x_std, seeds=30)
    
    print(f"\n[✓] Evaluation Results (@ {args.test_loss*100:.0f}% Test Packet Loss + Friction + Noise)")
    print(f"    CfC trained on clean data:")
    print(f"    Average Time-To-Failure (TTF): {mean_ttf:.1f} ± {std_ttf:.1f} steps")
