import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import csv
import itertools
from time import time

sys.path.append(os.path.abspath('../src'))
from omnitrain.sparse_cfc import SparseCfC

from train_f110_ncp import load_dataset, create_advanced_layered_mask

def evaluate_architecture(n_sensory, n_process, n_header, density, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, d_in, d_out, device):
    hidden = n_sensory + n_process + n_header
    
    # 1. Create the structured Mask
    adj_matrix = create_advanced_layered_mask(d_in, n_sensory, n_process, n_header, density=density)
    
    total_synapses = hidden * (d_in + hidden)
    active_synapses = int(adj_matrix.sum().item())
    
    # 2. Instantiate SparseCfC
    model = SparseCfC(input_dim=d_in, hidden_dim=hidden, output_dim=d_out, adjacency_matrix=adj_matrix)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_tr, Y_tr, dt_tr)
    # Using 4 workers for speed
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    
    # Pre-masking FC
    with torch.no_grad():
        model.fc.weight[:, :n_sensory + n_process] = 0.0
        model.fc.bias[:] = 0.0
        
    epochs = 30
    
    # 3. Train for exactly 30 epochs
    for ep in range(epochs):
        model.train()
        for bx, by, bdt in loader:
            bx, by, bdt = bx.to(device), by.to(device), bdt.to(device)
            optimizer.zero_grad()
            preds = model(bx, bdt)
            loss = criterion(preds, by)
            loss.backward()
            
            # Mask gradients for non-header neurons
            model.fc.weight.grad[:, :n_sensory + n_process] = 0.0
            
            optimizer.step()
            
            with torch.no_grad():
                model.fc.weight[:, :n_sensory + n_process] = 0.0
                
    # 4. Final Evaluation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val, dt_val)
        val_loss = criterion(val_preds, Y_val)
        
    return val_loss.item(), hidden, active_synapses

def evaluate_worker(args):
    idx, total, combo, keys, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, d_in, d_out, device = args
    config = dict(zip(keys, combo))
    
    print(f"\n[Worker] Started [{idx}/{total}] Config: {config}")
    start_t = time()
    
    try:
        val_mse, total_neurons, active_synapses = evaluate_architecture(
            config['n_sensory'], config['n_process'], config['n_header'], config['density'],
            X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, d_in, d_out, device
        )
        elapsed = time() - start_t
        print(f"[Worker] Finished [{idx}/{total}] -> MSE: {val_mse:.4f} | Neurons: {total_neurons} | Synapses: {active_synapses} | Time: {elapsed:.1f}s")
        
        return (config['n_sensory'], config['n_process'], config['n_header'], config['density'], total_neurons, active_synapses, val_mse)
        
    except Exception as e:
        print(f"[Worker] Error in config {config}: {e}")
        return None

def main():
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True) # Required for CUDA multiprocess
    except RuntimeError:
        pass
        
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Starting MULTI-CORE NCP Architecture Search on device: {device}")
    
    X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, mean_X, std_X = load_dataset()
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    dt_val = dt_val.to(device)
    
    d_in = X_tr.shape[2]
    d_out = Y_tr.shape[2]
    
    # Define hyperparameter grid
    search_space = {
        'n_sensory': [10, 20, 30, 50],
        'n_process': [10, 30, 60, 100],
        'n_header': [5, 10, 20, 50],
        'density': [0.1, 0.25, 0.5]
    }
    
    keys = list(search_space.keys())
    combinations = list(itertools.product(*[search_space[k] for k in keys]))
    
    results_file = "data/ncp_search_results.csv"
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_sensory", "n_process", "n_header", "density", "total_neurons", "active_synapses", "val_mse_epoch30"])
    
    print(f"Total architectures to test: {len(combinations)}")
    
    # Prepare arguments for multiprocessing pool
    pool_args = []
    for idx, combo in enumerate(combinations):
        pool_args.append((idx + 1, len(combinations), combo, keys, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, d_in, d_out, device))
        
    # Use 4 parallel workers (GPUs can multiplex multiple small training runs easily)
    num_workers = min(4, mp.cpu_count())
    print(f"Launching {num_workers} parallel workers...")
    
    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(evaluate_worker, pool_args):
            if result is not None:
                # Save to CSV instantly as results come in
                with open(results_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(result)
                    
    print("Multi-core architecture search completed!")

if __name__ == "__main__":
    main()
