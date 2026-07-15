import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing as mp
import numpy as np
import time
import csv
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from omnitrain.sparse_cfc import SparseCfC

from train_f110_ncp import load_dataset, create_advanced_layered_mask

# --- NEW TOPOLOGIES ---

def create_2d_multisensory_mask(input_dim=25, n_sensory=50, n_process=25, n_header=25, density=0.25):
    """2D Multi-Branch Lobe Topology"""
    hidden_dim = n_sensory + n_process + n_header
    mask = torch.zeros(hidden_dim, input_dim + hidden_dim, dtype=torch.bool)
    
    # 25 Inputs -> Lobe A: 0-7, 17-23 (Side), Lobe B: 8-16 (Front), Lobe C: 24 (Speed)
    side_idx = list(range(0, 8)) + list(range(17, 24))
    front_idx = list(range(8, 17))
    speed_idx = [24]
    
    # Divide 50 sensory into lobes (20, 20, 10)
    sen_side = 20
    sen_front = 20
    sen_speed = 10
    
    c_in = 0
    c_sen = input_dim
    c_pro = input_dim + n_sensory
    c_hdr = input_dim + n_sensory + n_process
    
    # Mask input to specific lobes
    for s_i in range(sen_side):
        mask[s_i, np.random.choice(side_idx, size=max(1, int(len(side_idx)*density)))] = True
    for s_i in range(sen_front):
        mask[sen_side + s_i, np.random.choice(front_idx, size=max(1, int(len(front_idx)*density)))] = True
    for s_i in range(sen_speed):
        mask[sen_side + sen_front + s_i, speed_idx[0]] = True # Speed directly mapped
        
    # Process Layer (split into Spatial and Temporal)
    pro_spatial = 15
    pro_temporal = 10
    
    # Spatial receives from Side and Front
    for p_i in range(pro_spatial):
        mask[n_sensory + p_i, c_sen : c_sen + sen_side + sen_front] = torch.rand(sen_side + sen_front) < density
    
    # Temporal receives from Front and Speed
    for p_i in range(pro_temporal):
        mask[n_sensory + pro_spatial + p_i, c_sen + sen_side : c_sen + n_sensory] = torch.rand(sen_front + sen_speed) < density
        
    # Process recurrent
    mask[n_sensory:n_sensory+n_process, c_pro:c_pro+n_process] = torch.rand(n_process, n_process) < density
    
    # Header receives from Process and recurrent
    mask[n_sensory+n_process:, c_pro:c_pro+n_process] = torch.rand(n_header, n_process) < density
    mask[n_sensory+n_process:, c_hdr:] = torch.rand(n_header, n_header) < density
    
    return mask.float()

def create_constant_loop_mask(input_dim=25, n_sensory=50, n_process=25, n_header=25, density=0.25):
    """Process layer forced into an unbreakable ring topology"""
    hidden_dim = n_sensory + n_process + n_header
    mask = torch.zeros(hidden_dim, input_dim + hidden_dim, dtype=torch.bool)
    
    c_in = 0
    c_sen = input_dim
    c_pro = input_dim + n_sensory
    c_hdr = input_dim + n_sensory + n_process
    
    # Standard input to sensory
    mask[0:n_sensory, c_in:c_in+input_dim] = torch.rand(n_sensory, input_dim) < density
    
    # Sensory to Process
    mask[n_sensory:n_sensory+n_process, c_sen:c_sen+n_sensory] = torch.rand(n_process, n_sensory) < density
    
    # Process Loop (Node i -> Node i+1)
    for i in range(n_process):
        next_i = (i + 1) % n_process
        mask[n_sensory + next_i, c_pro + i] = True
        # Add random sparse connections too
        mask[n_sensory:n_sensory+n_process, c_pro:c_pro+n_process] |= (torch.rand(n_process, n_process) < density)
        
    # Process to Header
    mask[n_sensory+n_process:, c_pro:c_pro+n_process] = torch.rand(n_header, n_process) < density
    mask[n_sensory+n_process:, c_hdr:] = torch.rand(n_header, n_header) < density
    
    return mask.float()

def create_3d_array_mask(input_dim=25, n_sensory=50, n_process=25, n_header=25, density=0.25):
    """3D Grid topology where neurons only connect to spatial neighbors"""
    hidden_dim = n_sensory + n_process + n_header # 100
    mask = torch.zeros(hidden_dim, input_dim + hidden_dim, dtype=torch.bool)
    
    # Map 100 neurons to a 5x5x4 grid
    grid_x, grid_y, grid_z = 5, 5, 4
    
    def get_coords(idx):
        z = idx // (grid_x * grid_y)
        rem = idx % (grid_x * grid_y)
        y = rem // grid_x
        x = rem % grid_x
        return x, y, z
        
    # Input -> Z=0 face (first 25 neurons = sensory)
    # Each input connects to the 5x5 face with some sparsity
    c_sen = input_dim
    mask[0:25, 0:input_dim] = torch.rand(25, input_dim) < density
    # The remaining 25 sensory neurons are at Z=1 (hidden sensory layer)
    mask[25:50, 0:input_dim] = torch.rand(25, input_dim) < (density / 2.0)
    
    # Internal Grid connections (Local neighborhood)
    for i in range(hidden_dim):
        xi, yi, zi = get_coords(i)
        for j in range(hidden_dim):
            xj, yj, zj = get_coords(j)
            # Distance constraint (only connect to neighbors in 3D space, distance <= 2)
            dist = abs(xi - xj) + abs(yi - yj) + abs(zi - zj)
            if dist > 0 and dist <= 2:
                # Flow naturally flows forward in Z, or recurrently in the same Z
                if zj >= zi - 1: # Allow forward and slight backward
                    if np.random.rand() < density:
                        mask[j, c_sen + i] = True
                        
    # Ensure Header (Z=3 face) has recurrent and can output properly
    c_hdr = input_dim + n_sensory + n_process
    # We don't need to manually wire header, the 3D grid naturally puts the last 25 neurons in Z=3
    
    return mask.float()


# --- WORKER SETUP ---

# Global dataset variables for workers to prevent IPC memory explosion
X_tr_w, Y_tr_w, dt_tr_w, X_val_w, Y_val_w, dt_val_w = None, None, None, None, None, None
d_in_w, d_out_w = None, None

def init_worker():
    global X_tr_w, Y_tr_w, dt_tr_w, X_val_w, Y_val_w, dt_val_w, d_in_w, d_out_w
    X_tr_w, Y_tr_w, dt_tr_w, X_val_w, Y_val_w, dt_val_w, _, _ = load_dataset()
    d_in_w = X_tr_w.shape[2]
    d_out_w = Y_tr_w.shape[2]

def evaluate_topology_worker(config, device_name, results_file):
    device = torch.device(device_name)
    
    topology_name = config['topology']
    n_sensory = config['n_sensory']
    n_process = config['n_process']
    n_header = config['n_header']
    density = config['density']
    
    hidden = n_sensory + n_process + n_header
    
    # 1. Create Mask
    if topology_name == "Linear_Baseline":
        adj_matrix = create_advanced_layered_mask(d_in_w, n_sensory, n_process, n_header, density=density)
    elif topology_name == "2D_MultiSensory":
        adj_matrix = create_2d_multisensory_mask(d_in_w, n_sensory, n_process, n_header, density=density)
    elif topology_name == "Constant_Loop":
        adj_matrix = create_constant_loop_mask(d_in_w, n_sensory, n_process, n_header, density=density)
    elif topology_name == "3D_Array":
        adj_matrix = create_3d_array_mask(d_in_w, n_sensory, n_process, n_header, density=density)
    else:
        raise ValueError(f"Unknown topology: {topology_name}")
        
    active_synapses = int(adj_matrix.sum().item())
    
    # 2. Instantiate Model
    model = SparseCfC(input_dim=d_in_w, hidden_dim=hidden, output_dim=d_out_w, adjacency_matrix=adj_matrix)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_tr_w, Y_tr_w, dt_tr_w)
    
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    with torch.no_grad():
        model.fc.weight[:, :n_sensory + n_process] = 0.0
        model.fc.bias[:] = 0.0
        
    epochs = 30
    start_t = time.time()
    
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
        
    elapsed = time.time() - start_t
    print(f"✅ {topology_name} -> MSE: {val_loss:.4f} | Synapses: {active_synapses} | Time: {elapsed:.1f}s")
    
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            topology_name, n_sensory, n_process, n_header, density,
            hidden, active_synapses, val_loss
        ])
        
def run_worker_task(args):
    try:
        evaluate_topology_worker(*args)
    except Exception as e:
        print(f"❌ Error in config {args[0]}: {e}")

def main():
    mp.set_start_method('spawn', force=True)
    
    device_name = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting TOPOLOGY SEARCH. Accelerator: {device_name}")
    
    topologies = ["Linear_Baseline", "2D_MultiSensory", "Constant_Loop", "3D_Array"]
    
    results_file = "data/topology_search_results.csv"
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["topology", "n_sensory", "n_process", "n_header", "density", "total_neurons", "active_synapses", "val_mse_epoch30"])
    
    # Using the 50-25-25 Sweet Spot layout at 25% density
    tasks = []
    for topo in topologies:
        config = {
            'topology': topo,
            'n_sensory': 50,
            'n_process': 25,
            'n_header': 25,
            'density': 0.25
        }
        tasks.append((config, device_name, results_file))
        
    num_processes = 4 
    print(f"Launching Pool with {num_processes} parallel workers...")
    
    with mp.Pool(processes=num_processes, initializer=init_worker) as pool:
        pool.map(run_worker_task, tasks)
        
    print("Search Complete! Check data/topology_search_results.csv")

if __name__ == "__main__":
    main()
