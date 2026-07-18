"""
lstm_ablation_augmentation.py — LSTM Data Augmentation Ablation Study

This script trains the Time-Aware LSTM baseline with injected Zero-Order Hold 
(ZOH) packet loss during the training phase. 
This acts as an ablation study to test if the discrete LSTM can learn to emulate 
the CfC's temporal extrapolation if it explicitly sees jitter during optimization.

Usage:
    python lstm_ablation_augmentation.py --loss_rate 0.20
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

# --- CartPole Physics Constants ---
_GRAVITY = 9.8
_MASSCART = 1.0
_MASSPOLE = 0.1
_TOTAL_M = _MASSPOLE + _MASSCART
_LENGTH = 0.5
_POLEMASS_L = _MASSPOLE * _LENGTH
_DT = 0.02
X_FAIL = 2.4
THETA_FAIL = 12 * 2 * np.pi / 360

def cartpole_step(state: np.ndarray, force: float) -> np.ndarray:
    x, xd, th, thd = state
    cos_th, sin_th = np.cos(th), np.sin(th)
    temp = (force + _POLEMASS_L * thd**2 * sin_th) / _TOTAL_M
    th_acc = ((_GRAVITY * sin_th - cos_th * temp) /
               (_LENGTH * (4.0/3.0 - _MASSPOLE * cos_th**2 / _TOTAL_M)))
    x_acc = temp - _POLEMASS_L * th_acc * cos_th / _TOTAL_M
    return np.array([x + _DT*xd, xd + _DT*x_acc, th + _DT*thd, thd + _DT*th_acc])

def is_failed(state: np.ndarray) -> bool:
    return bool(abs(state[0]) > X_FAIL or abs(state[2]) > THETA_FAIL)

class TimeAwareLSTM(nn.Module):
    def __init__(self, d_in=5, d_model=32, d_out=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_in, hidden_size=d_model, batch_first=True)
        self.fc = nn.Linear(d_model, d_out)
        
    def forward(self, x, h=None):
        out, h = self.lstm(x, h)
        return self.fc(out), h

def inject_zoh_loss(X, loss_prob=0.2):
    mask = np.random.rand(len(X)) > loss_prob
    X_masked = X.copy()
    for i in range(1, len(X)):
        if not mask[i]:
            X_masked[i] = X_masked[i-1]
    return X_masked

def train_augmented_lstm(X_train, Y_train, T_train, augment_loss_prob=0.20, epochs=250):
    print(f"[*] Training LSTM with {augment_loss_prob*100:.0f}% Data Augmentation (ZOH Packet Loss)")
    
    if augment_loss_prob > 0.0:
        X_train_aug = inject_zoh_loss(X_train, loss_prob=augment_loss_prob)
    else:
        X_train_aug = X_train
        
    dT = np.zeros_like(T_train)
    dT[1:] = T_train[1:] - T_train[:-1]
    dT[0] = 0.02
    
    features = np.concatenate([X_train_aug, dT], axis=-1)
    
    X_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    Y_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(0)
    
    model = TimeAwareLSTM(d_in=5, d_model=32, d_out=1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred, _ = model(X_tensor)
        loss = criterion(pred, Y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {loss.item():.5f}")
            
    return model

def evaluate_closed_loop_lstm(model, x_mean, x_std, drop_prob=0.2, seeds=30, max_steps=500):
    model.eval()
    ttfs = []
    
    for seed in range(seeds):
        rng = np.random.RandomState(seed)
        state = np.array([0.0, 0.0, rng.uniform(-0.05, 0.05), 0.0], dtype=np.float32)
        held = state.copy()
        rnn_h = None
        
        with torch.no_grad():
            for step in range(max_steps):
                if rng.rand() >= drop_prob:
                    held = state.copy()
                
                held_n = (held - x_mean) / x_std
                # Concatenate dT=0.02 explicitly
                feat = np.concatenate([held_n, [0.02]])
                xt = torch.FloatTensor(feat).unsqueeze(0).unsqueeze(0)
                
                out, rnn_h = model(xt, rnn_h)
                force = out[0, 0, 0].item()
                
                state = cartpole_step(state, force)
                if is_failed(state):
                    ttfs.append(step)
                    break
            else:
                ttfs.append(max_steps)
                
    return np.mean(ttfs), np.std(ttfs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_rate", type=float, default=0.20, help="Packet loss prob during training")
    parser.add_argument("--test_loss", type=float, default=0.20, help="Packet loss prob during testing")
    args = parser.parse_args()
    
    print("[*] Loading clean cartpole trajectories...")
    X_clean = np.load("../data/pendulum_X_0loss.npy")
    Y = np.load("../data/pendulum_Y.npy")
    T = np.load("../data/pendulum_T.npy")
    
    x_mean = np.mean(X_clean, axis=0)
    x_std = np.std(X_clean, axis=0) + 1e-8
    
    X_norm = (X_clean - x_mean) / x_std
    
    model = train_augmented_lstm(X_norm, Y, T, augment_loss_prob=args.loss_rate, epochs=250)
    
    mean_ttf, std_ttf = evaluate_closed_loop_lstm(model, x_mean, x_std, drop_prob=args.test_loss, seeds=30)
    
    print(f"\n[✓] Evaluation Results (@ {args.test_loss*100:.0f}% Test Packet Loss)")
    print(f"    LSTM trained with {args.loss_rate*100:.0f}% data augmentation:")
    print(f"    Average Time-To-Failure (TTF): {mean_ttf:.1f} ± {std_ttf:.1f} steps")
