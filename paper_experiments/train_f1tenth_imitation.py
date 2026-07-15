import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os

from train_and_compare import ContinuousCfC, DiscreteRNN

def load_dataset():
    data = np.load("data/f1tenth_dataset.npz")
    # lidar: (N, 21), states: (N, 1), actions: (N, 2)
    lidar = data["lidar"]
    states = data["states"]
    actions = data["actions"]
    
    # inputs = [v, ray_1, ..., ray_21]
    inputs = np.hstack([states, lidar])
    
    # Normalize inputs
    mean_X = np.mean(inputs, axis=0)
    std_X = np.std(inputs, axis=0) + 1e-6
    inputs_norm = (inputs - mean_X) / std_X
    
    # Create overlapping sequences
    seq_len = 100
    X, Y, dt = [], [], []
    for i in range(len(inputs_norm) - seq_len):
        X.append(inputs_norm[i:i+seq_len])
        Y.append(actions[i:i+seq_len])
        dt.append(np.full((seq_len, 1), 0.05))
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    dt = torch.tensor(np.array(dt), dtype=torch.float32)
    
    # Train/Val split
    split = int(0.8 * len(X))
    return X[:split], Y[:split], dt[:split], X[split:], Y[split:], dt[split:], mean_X, std_X

def train_model(model, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, epochs=500, is_cfc=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_tr, Y_tr, dt_tr)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for ep in range(epochs):
        model.train()
        for bx, by, bdt in loader:
            optimizer.zero_grad()
            if is_cfc:
                preds = model(bx, bdt)
            else:
                # Provide dt to RNN explicitly by concatenating (Time-Aware RNN)
                bx_with_dt = torch.cat([bx, bdt], dim=-1)
                preds = model(bx_with_dt)
                
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            
        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                if is_cfc:
                    val_preds = model(X_val, dt_val)
                else:
                    bx_val_dt = torch.cat([X_val, dt_val], dim=-1)
                    val_preds = model(bx_val_dt)
                val_loss = criterion(val_preds, Y_val)
            print(f"Epoch {ep+1}/{epochs} | Val MSE: {val_loss.item():.4f}")
            
if __name__ == "__main__":
    X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, mean_X, std_X = load_dataset()
    print(f"Training on sequences of shape {X_tr.shape}")
    
    d_in = X_tr.shape[2]
    d_out = Y_tr.shape[2]
    hidden = 32
    
    print("\n--- Training LSTM ---")
    lstm = DiscreteRNN(d_in + 1, hidden, d_out, rnn_type='lstm')
    train_model(lstm, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, epochs=500)
    
    print("\n--- Training GRU ---")
    gru = DiscreteRNN(d_in + 1, hidden, d_out, rnn_type='gru')
    train_model(gru, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, epochs=500)
    
    print("\n--- Training CfC ---")
    cfc = ContinuousCfC(input_dim=d_in, hidden_dim=hidden, output_dim=d_out, backbone_units=64)
    train_model(cfc, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, epochs=500, is_cfc=True)
    
    torch.save(cfc.state_dict(), "data/f1tenth_cfc.pt")
    np.savez("data/f1tenth_stats.npz", mean=mean_X, std=std_X)
    print("Saved CfC model and stats.")
