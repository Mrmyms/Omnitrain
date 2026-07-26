import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add src to path so we can import omnitrain
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from omnitrain.sparse_cfc import SparseCfC

def load_dataset():
    data = np.load("../data/f110_real_dataset.npz")
    lidar = data["lidar"]
    states = data["states"]
    actions = data["actions"]
    
    inputs = np.hstack([states, lidar])
    mean_X = np.mean(inputs, axis=0)
    std_X = np.std(inputs, axis=0) + 1e-6
    inputs_norm = (inputs - mean_X) / std_X
    
    seq_len = 100
    X, Y, dt = [], [], []
    for i in range(len(inputs_norm) - seq_len):
        X.append(inputs_norm[i:i+seq_len])
        Y.append(actions[i:i+seq_len])
        dt.append(np.full((seq_len, 1), 0.05))
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    dt = torch.tensor(np.array(dt), dtype=torch.float32)
    
    split = int(0.8 * len(X))
    return X[:split], Y[:split], dt[:split], X[split:], Y[split:], dt[split:], mean_X, std_X

def create_20_10_20_mask(input_dim, n_sensory=20, n_process=10, n_header=20, density=0.5):
    """Creates the 20-10-20 advanced layered connectome"""
    hidden_dim = n_sensory + n_process + n_header
    mask = torch.zeros(hidden_dim, input_dim + hidden_dim, dtype=torch.bool)
    
    c_in_start, c_in_end = 0, input_dim
    c_sen_start, c_sen_end = input_dim, input_dim + n_sensory
    c_pro_start, c_pro_end = c_sen_end, c_sen_end + n_process
    c_hdr_start, c_hdr_end = c_pro_end, c_pro_end + n_header
    
    # Sensory Neurons
    mask[0:n_sensory, c_in_start:c_in_end] = torch.rand(n_sensory, input_dim) < density
    
    # Process Neurons
    mask[n_sensory:n_sensory+n_process, c_sen_start:c_sen_end] = torch.rand(n_process, n_sensory) < density
    mask[n_sensory:n_sensory+n_process, c_pro_start:c_pro_end] = torch.rand(n_process, n_process) < density
    
    # Header Neurons
    mask[n_sensory+n_process:, c_pro_start:c_pro_end] = torch.rand(n_header, n_process) < density
    mask[n_sensory+n_process:, c_hdr_start:c_hdr_end] = torch.rand(n_header, n_header) < density
    
    # Ensure connectivity
    for i in range(c_in_start, c_in_end):
        if not mask[0:n_sensory, i].any():
            mask[torch.randint(0, n_sensory, (1,)), i] = True
            
    for i in range(c_sen_start, c_sen_end):
        if not mask[n_sensory:n_sensory+n_process, i].any():
            mask[torch.randint(n_sensory, n_sensory+n_process, (1,)), i] = True
            
    for i in range(c_pro_start, c_pro_end):
        if not mask[n_sensory+n_process:, i].any():
            mask[torch.randint(n_sensory+n_process, hidden_dim, (1,)), i] = True

    return mask.float()

def train_model(model, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, epochs=50):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training NCP on device: {device}")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_tr, Y_tr, dt_tr)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    dt_val = dt_val.to(device)
    
    # Enforce zero weights for Sensory and Process outputs
    with torch.no_grad():
        model.fc.weight[:, :30] = 0.0
        model.fc.bias[:] = 0.0 
    
    for ep in range(epochs):
        model.train()
        for bx, by, bdt in loader:
            bx, by, bdt = bx.to(device), by.to(device), bdt.to(device)
            optimizer.zero_grad()
            preds = model(bx, bdt)
            loss = criterion(preds, by)
            loss.backward()
            
            # Mask gradients so only Header neurons (30-49) update the motor output
            model.fc.weight.grad[:, :30] = 0.0
            
            optimizer.step()
            
            with torch.no_grad():
                model.fc.weight[:, :30] = 0.0
        
        scheduler.step()
            
        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val, dt_val)
                val_loss = criterion(val_preds, Y_val)
            print(f"Epoch {ep+1}/{epochs} | Val MSE: {val_loss.item():.4f}", flush=True)

if __name__ == "__main__":
    X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, mean_X, std_X = load_dataset()
    print(f"Training on REAL F1TENTH sequences of shape {X_tr.shape}")
    
    d_in = X_tr.shape[2]
    d_out = Y_tr.shape[2]
    
    n_sensory = 20
    n_process = 10
    n_header = 20
    hidden = n_sensory + n_process + n_header
    
    # Advanced 20-10-20 Connectome
    adj_matrix = create_20_10_20_mask(d_in, n_sensory, n_process, n_header, density=0.5)
    
    total_synapses = hidden * (d_in + hidden)
    active_synapses = int(adj_matrix.sum().item())
    print(f"Advanced NCP Generated: {active_synapses}/{total_synapses} synapses active ({(1 - active_synapses/total_synapses)*100:.1f}% Sparse)")
    
    ncp = SparseCfC(input_dim=d_in, hidden_dim=hidden, output_dim=d_out, adjacency_matrix=adj_matrix)
    
    # Pre-train for 50 epochs
    train_model(ncp, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, epochs=50)
    
    save_path = "../data/f110_20_10_20_bc.pt"
    torch.save(ncp.state_dict(), save_path)
    print(f"Saved NCP to {save_path}")
