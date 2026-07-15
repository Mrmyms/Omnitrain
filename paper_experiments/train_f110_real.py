import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

from train_and_compare import ContinuousCfC, DiscreteRNN

def load_dataset():
    data = np.load("data/f110_real_dataset.npz")
    # lidar: (N, 24), states: (N, 1), actions: (N, 2)
    lidar = data["lidar"]
    states = data["states"]
    actions = data["actions"]
    
    # inputs = [v, ray_1, ..., ray_24]
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

def train_model(model, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, epochs=200, is_cfc=False):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on device: {device}")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_tr, Y_tr, dt_tr)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    dt_val = dt_val.to(device)
    
    for ep in range(epochs):
        model.train()
        for bx, by, bdt in loader:
            bx, by, bdt = bx.to(device), by.to(device), bdt.to(device)
            optimizer.zero_grad()
            if is_cfc:
                preds = model(bx, bdt)
            else:
                bx_with_dt = torch.cat([bx, bdt], dim=-1)
                preds = model(bx_with_dt)
                
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
            
        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                if is_cfc:
                    val_preds = model(X_val, dt_val)
                else:
                    bx_val_dt = torch.cat([X_val, dt_val], dim=-1)
                    val_preds = model(bx_val_dt)
                val_loss = criterion(val_preds, Y_val)
            print(f"Epoch {ep+1}/{epochs} | Val MSE: {val_loss.item():.4f}", flush=True)
            
def run_training_worker(model_name, model, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, is_cfc):
    print(f"\n--- Training {model_name} ---")
    train_model(model, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, epochs=400, is_cfc=is_cfc)
    
    # Save after training
    save_path = f"data/f110_real_{model_name.lower()}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved {model_name} to {save_path}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True) # spawn is required for CUDA/MPS multiprocessing
    
    X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, mean_X, std_X = load_dataset()
    print(f"Training on REAL F1TENTH sequences of shape {X_tr.shape}")
    
    d_in = X_tr.shape[2]
    d_out = Y_tr.shape[2]
    hidden = 32
    
    lstm = DiscreteRNN(d_in + 1, hidden, d_out, rnn_type='lstm')
    gru = DiscreteRNN(d_in + 1, hidden, d_out, rnn_type='gru')
    cfc = ContinuousCfC(input_dim=d_in, hidden_dim=hidden, output_dim=d_out, backbone_units=64)
    
    # Run in true parallel multiprocessing to get 100% CPU utilization
    processes = []
    
    p_lstm = mp.Process(target=run_training_worker, args=("LSTM", lstm, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, False))
    p_gru = mp.Process(target=run_training_worker, args=("GRU", gru, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, False))
    p_cfc = mp.Process(target=run_training_worker, args=("CfC", cfc, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, True))
    
    processes.extend([p_lstm, p_gru, p_cfc])
    
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()
        
    np.savez("data/f110_real_stats.npz", mean=mean_X, std=std_X)
    print("Parallel training complete. Saved all models and stats.")
