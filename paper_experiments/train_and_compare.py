import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Ensure omnitrain is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from omnitrain.fusion_core import BioLiquidCell

class DiscreteRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rnn_type='lstm'):
        super().__init__()
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)

class ContinuousCfC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.cell = BioLiquidCell(input_size=input_dim, hidden_size=hidden_dim, backbone_units=32)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, times):
        # x: (batch, seq, features)
        # times: (batch, seq, 1)
        batch_size, seq_len, _ = x.shape
        hidden = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            dt = times[:, t, :]
            if t > 0:
                dt = times[:, t, :] - times[:, t-1, :]
            else:
                dt = torch.zeros_like(dt)
                
            hidden = self.cell(xt, hidden, dt)
            outputs.append(hidden.unsqueeze(1))
            
        outputs = torch.cat(outputs, dim=1)
        return self.fc(outputs)

def train_model(model, X, Y, T=None, epochs=50):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss()
    
    X_t = torch.FloatTensor(X).unsqueeze(0) # Batch size 1
    Y_t = torch.FloatTensor(Y).unsqueeze(0)
    if T is not None:
        T_t = torch.FloatTensor(T).unsqueeze(0)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        if T is not None:
            pred = model(X_t, T_t)
        else:
            pred = model(X_t)
            
        loss = criterion(pred, Y_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    return model

def evaluate_model(model, X, Y, T=None):
    model.eval()
    X_t = torch.FloatTensor(X).unsqueeze(0)
    Y_t = torch.FloatTensor(Y).unsqueeze(0)
    if T is not None:
        T_t = torch.FloatTensor(T).unsqueeze(0)
        
    with torch.no_grad():
        if T is not None:
            pred = model(X_t, T_t)
        else:
            pred = model(X_t)
            
        mse = nn.MSELoss()(pred, Y_t).item()
    return mse

if __name__ == "__main__":
    import random
    
    print("Loading CartPole data...")
    X_0 = np.load("data/pendulum_X_0loss.npy")
    X_20 = np.load("data/pendulum_X_20loss.npy")
    X_60 = np.load("data/pendulum_X_60loss.npy")
    Y = np.load("data/pendulum_Y.npy")
    T = np.load("data/pendulum_T.npy")
    
    hidden_dim = 16 # roughly 4000 params for LSTM/GRU/CfC
    seeds = [42, 100, 2026, 777, 999]
    
    raw_results = {"LSTM": {0: [], 20: [], 60: []}, 
                   "GRU": {0: [], 20: [], 60: []}, 
                   "CfC": {0: [], 20: [], 60: []}}
    
    for seed in seeds:
        print(f"\n=====================")
        print(f"Running Seed: {seed}")
        print(f"=====================")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        models = {
            "LSTM": DiscreteRNN(4, hidden_dim, 1, 'lstm'),
            "GRU": DiscreteRNN(4, hidden_dim, 1, 'gru'),
            "CfC": ContinuousCfC(4, hidden_dim, 1)
        }
        
        for name, model in models.items():
            print(f"--- Training {name} ---")
            if name == "CfC":
                model = train_model(model, X_0, Y, T, epochs=50)
            else:
                model = train_model(model, X_0, Y, epochs=50)
                
            mse_0 = evaluate_model(model, X_0, Y, T if name == "CfC" else None)
            mse_20 = evaluate_model(model, X_20, Y, T if name == "CfC" else None)
            mse_60 = evaluate_model(model, X_60, Y, T if name == "CfC" else None)
            
            raw_results[name][0].append(mse_0)
            raw_results[name][20].append(mse_20)
            raw_results[name][60].append(mse_60)
            
    # Calculate Mean and Std
    final_results = {}
    for name in raw_results:
        final_results[name] = {
            "mean": [np.mean(raw_results[name][0]), np.mean(raw_results[name][20]), np.mean(raw_results[name][60])],
            "std": [np.std(raw_results[name][0]), np.std(raw_results[name][20]), np.std(raw_results[name][60])]
        }
        print(f"\n{name} Final Results:")
        print(f"MSE (0% loss): {final_results[name]['mean'][0]:.4f} ± {final_results[name]['std'][0]:.4f}")
        print(f"MSE (20% loss): {final_results[name]['mean'][1]:.4f} ± {final_results[name]['std'][1]:.4f}")
        print(f"MSE (60% loss): {final_results[name]['mean'][2]:.4f} ± {final_results[name]['std'][2]:.4f}")
        
    np.save("data/results_mse.npy", final_results)
    print("\nExperiment complete. Results saved for plotting.")
