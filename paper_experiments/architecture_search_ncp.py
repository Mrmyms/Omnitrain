import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import csv
import itertools
from time import time
import multiprocessing as mp

sys.path.append(os.path.abspath('../src'))
from omnitrain.sparse_cfc import SparseCfC

from train_f110_ncp import load_dataset, create_advanced_layered_mask

# Global dataset variables for workers to prevent IPC memory explosion
X_tr_w, Y_tr_w, dt_tr_w, X_val_w, Y_val_w, dt_val_w = None, None, None, None, None, None
d_in_w, d_out_w = None, None

def init_worker():
    global X_tr_w, Y_tr_w, dt_tr_w, X_val_w, Y_val_w, dt_val_w, d_in_w, d_out_w
    # Load dataset directly in the worker to avoid pickling 100GB of RAM across IPC pipes
    X_tr_w, Y_tr_w, dt_tr_w, X_val_w, Y_val_w, dt_val_w, _, _ = load_dataset()
    d_in_w = X_tr_w.shape[2]
    d_out_w = Y_tr_w.shape[2]

def evaluate_architecture_worker(config, device_name, results_file):
    device = torch.device(device_name)
    
    n_sensory = config['n_sensory']
    n_process = config['n_process']
    n_header = config['n_header']
    density = config['density']
    
    hidden = n_sensory + n_process + n_header
    
    # 1. Create Mask
    adj_matrix = create_advanced_layered_mask(d_in_w, n_sensory, n_process, n_header, density=density)
    total_synapses = hidden * (d_in_w + hidden)
    active_synapses = int(adj_matrix.sum().item())
    
    # 2. Instantiate Model
    model = SparseCfC(input_dim=d_in_w, hidden_dim=hidden, output_dim=d_out_w, adjacency_matrix=adj_matrix)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_tr_w, Y_tr_w, dt_tr_w)
    
    # We set num_workers=0 because we are already parallelizing the models. 
    # Multiple dataloader threads per model would cause CPU thrashing.
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    with torch.no_grad():
        model.fc.weight[:, :n_sensory + n_process] = 0.0
        model.fc.bias[:] = 0.0
        
    epochs = 30
    start_t = time()
    
    # 3. Train
    for ep in range(epochs):
        model.train()
        for bx, by, bdt in loader:
            bx, by, bdt = bx.to(device), by.to(device), bdt.to(device)
            optimizer.zero_grad()
            preds = model(bx, bdt)
            loss = criterion(preds, by)
            loss.backward()
            model.fc.weight.grad[:, :n_sensory + n_process] = 0.0
            optimizer.step()
            with torch.no_grad():
                model.fc.weight[:, :n_sensory + n_process] = 0.0
                
    # 4. Evaluate
    model.eval()
    with torch.no_grad():
        X_val_dev = X_val_w.to(device)
        dt_val_dev = dt_val_w.to(device)
        Y_val_dev = Y_val_w.to(device)
        val_preds = model(X_val_dev, dt_val_dev)
        val_loss = criterion(val_preds, Y_val_dev).item()
        
    elapsed = time() - start_t
    print(f"✅ Config {config} -> MSE: {val_loss:.4f} | Synapses: {active_synapses} | Time: {elapsed:.1f}s")
    
    # Save instantly to CSV
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            n_sensory, n_process, n_header, density,
            hidden, active_synapses, val_loss
        ])
        
def run_worker_task(args):
    try:
        evaluate_architecture_worker(*args)
    except Exception as e:
        print(f"❌ Error in config {args[0]}: {e}")

def main():
    # Enforce spawn for CUDA/MPS multiprocessing safety
    mp.set_start_method('spawn', force=True)
    
    device_name = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting MULTI-CORE NCP Architecture Search. Accelerator: {device_name}")
    
    # Test the user-defined optimal baseline
    base_architectures = [
        (25, 15, 15),   # 55 neurons
    ]
    
    densities = [0.1, 0.25, 0.5]
    
    combinations = []
    for arch in base_architectures:
        for d in densities:
            combinations.append((arch[0], arch[1], arch[2], d))
            
    keys = ['n_sensory', 'n_process', 'n_header', 'density']
    
    results_file = "data/ncp_search_results.csv"
    
    # Check if we should append or overwrite (don't overwrite what we already computed!)
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["n_sensory", "n_process", "n_header", "density", "total_neurons", "active_synapses", "val_mse_epoch30"])
    
    print(f"Total architectures to test: {len(combinations)}")
    
    # Prepare arguments for multiprocessing pool (Do NOT pass dataset arrays!)
    tasks = []
    for combo in combinations:
        config = dict(zip(keys, combo))
        tasks.append((config, device_name, results_file))
        
    # Launch Process Pool
    # We use 4 parallel processes. 
    num_processes = 4 
    print(f"Launching Pool with {num_processes} parallel workers...")
    
    with mp.Pool(processes=num_processes, initializer=init_worker) as pool:
        pool.map(run_worker_task, tasks)
        
    print("Search Complete! Check data/ncp_search_results.csv")

if __name__ == "__main__":
    main()
