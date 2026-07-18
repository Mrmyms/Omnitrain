import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from omnitrain.sparse_cfc import SparseCfC
from omnitrain.esp32_exporter import ESP32Exporter

def train():
    data_dir = "paper_experiments/data"
    X0 = np.load(os.path.join(data_dir, "pendulum_X_0loss.npy"))
    Y  = np.load(os.path.join(data_dir, "pendulum_Y.npy"))
    T  = np.load(os.path.join(data_dir, "pendulum_T.npy"))
    
    X_mean, X_std = X0.mean(0), X0.std(0) + 1e-8
    X0_n = (X0 - X_mean) / X_std
    
    Xt = torch.FloatTensor(X0_n)
    Yt = torch.FloatTensor(Y)
    Tt = torch.FloatTensor(T)
    
    # Create SparseCfC model
    # For cartpole: d_in=4, d_out=1. Let's use 16 hidden units.
    d_in = 4
    d_model = 16
    d_out = 1
    
    # Fully connected mask (density=1.0) to ensure it works easily
    adj = torch.ones((d_model, d_in + d_model))
    
    model = SparseCfC(input_dim=d_in, hidden_dim=d_model, output_dim=d_out, adjacency_matrix=adj)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Training SparseCfC CartPole model for HIL...")
    for ep in range(100):
        optimizer.zero_grad()
        # The forward expects (batch, seq, feat) and (batch, seq, 1)
        pred = model(Xt.unsqueeze(0), Tt.unsqueeze(0))
        loss = criterion(pred, Yt.unsqueeze(0))
        loss.backward()
        optimizer.step()
        if (ep+1) % 20 == 0:
            print(f"Epoch {ep+1} Loss: {loss.item():.4f}")
            
    print("Exporting...")
    exporter = ESP32Exporter(output_dir="hil_test/include")
    exporter.export(model, input_dim=d_in, d_model=d_model, output_dim=d_out, filename="hil_model.omnibit")
    print("Done!")

if __name__ == '__main__':
    train()
