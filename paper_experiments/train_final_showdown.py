import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from omnitrain.sparse_cfc import SparseCfC
from omnitrain.esp32_exporter import ESP32Exporter

# Import generators from our previous scripts
from train_f110_ncp import load_dataset, create_advanced_layered_mask
from topology_search_ncp import create_3d_array_mask, create_reflex_arc_mask

def train_final_model(model, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, epochs=400, n_sensory=25, n_process=50):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_tr, Y_tr, dt_tr)
    # Use standard 0 workers to avoid IPC issues in sequence
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    dt_val = dt_val.to(device)
    
    # Enforce zero weights for Sensory and Process outputs
    with torch.no_grad():
        model.fc.weight[:, :n_sensory + n_process] = 0.0
        model.fc.bias[:] = 0.0
    
    print(f"--- Starting 400-Epoch Training on {device} ---")
    start_t = time.time()
    
    for ep in range(epochs):
        model.train()
        for bx, by, bdt in loader:
            bx, by, bdt = bx.to(device), by.to(device), bdt.to(device)
            optimizer.zero_grad()
            preds = model(bx, bdt)
            loss = criterion(preds, by)
            loss.backward()
            
            # Mask gradients
            model.fc.weight.grad[:, :n_sensory + n_process] = 0.0
            
            optimizer.step()
            
            with torch.no_grad():
                model.fc.weight[:, :n_sensory + n_process] = 0.0
        
        scheduler.step()
            
        if (ep+1) % 50 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val, dt_val)
                val_loss = criterion(val_preds, Y_val)
            print(f"Epoch {ep+1:03d}/{epochs} | Val MSE: {val_loss.item():.4f}", flush=True)

    elapsed = time.time() - start_t
    print(f"--- Training Completed in {elapsed:.1f}s ---")
    return model

if __name__ == "__main__":
    X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, mean_X, std_X = load_dataset()
    print(f"Dataset loaded. Training shape: {X_tr.shape}")
    
    d_in = X_tr.shape[2]
    d_out = Y_tr.shape[2]
    density = 0.25
    
    os.makedirs("data", exist_ok=True)
    exporter = ESP32Exporter(output_dir="data")
    
    # ==========================================
    # 1. LINEAR BASELINE (100 Neurons: 50-25-25)
    # ==========================================
    print("\n" + "="*50)
    print("🥊 CONTENDER 1: LINEAR BASELINE (50-25-25) [ALREADY COMPLETED]")
    print("="*50)
    
    # n_sen_L = 50
    # n_pro_L = 25
    # n_hdr_L = 25
    # hidden_L = n_sen_L + n_pro_L + n_hdr_L
    
    # adj_linear = create_advanced_layered_mask(d_in, n_sen_L, n_pro_L, n_hdr_L, density=density)
    # model_linear = SparseCfC(input_dim=d_in, hidden_dim=hidden_L, output_dim=d_out, adjacency_matrix=adj_linear)
    
    # print(f"Linear Synapses: {int(adj_linear.sum().item())}")
    # model_linear = train_final_model(model_linear, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, epochs=400, 
    #                                  n_sensory=n_sen_L, n_process=n_pro_L)
                                     
    # # Save Model
    # torch.save(model_linear.state_dict(), "data/f110_linear_100.pt")
    # # Export to Omnibit
    # exporter.export(model_linear, input_dim=d_in, d_model=hidden_L, output_dim=d_out, filename="f110_linear_100.omnibit")
    # print("✅ f110_linear_100.omnibit exported successfully!")


    # ==========================================
    # 2. 3D ARRAY CUBE (100 Neurons: 5x5x4)
    # ==========================================
    print("\n" + "="*50)
    print("🥊 CONTENDER 2: 3D ARRAY CUBE (5x5x4) [ALREADY COMPLETED]")
    print("="*50)
    
    # grid_x, grid_y, grid_z = 5, 5, 4
    # hidden_C = grid_x * grid_y * grid_z # 100
    
    # adj_cube = create_3d_array_mask(d_in, grid_x, grid_y, grid_z, density=density)
    # model_cube = SparseCfC(input_dim=d_in, hidden_dim=hidden_C, output_dim=d_out, adjacency_matrix=adj_cube)
    
    # print(f"Cube Synapses: {int(adj_cube.sum().item())}")
    # # For the cube, the first 25 are purely sensory, and the last 25 are header (output capable)
    # n_sen_C = 25
    # n_pro_C = hidden_C - 25 - 25
    
    # model_cube = train_final_model(model_cube, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, epochs=400, 
    #                                  n_sensory=n_sen_C, n_process=n_pro_C)
                                     
    # # Save Model
    # torch.save(model_cube.state_dict(), "data/f110_cube_100.pt")
    # # Export to Omnibit
    # exporter.export(model_cube, input_dim=d_in, d_model=hidden_C, output_dim=d_out, filename="f110_cube_100.omnibit")
    # print("✅ f110_cube_100.omnibit exported successfully!")
    
    # ==========================================
    # 3. REFLEX ARC (100 Neurons: 50-25-25)
    # ==========================================
    print("\n" + "="*50)
    print("🥊 CONTENDER 3: REFLEX ARC (50-25-25)")
    print("="*50)
    
    n_sen_R = 50
    n_pro_R = 25
    n_hdr_R = 25
    hidden_R = n_sen_R + n_pro_R + n_hdr_R
    
    adj_reflex = create_reflex_arc_mask(d_in, n_sen_R, n_pro_R, n_hdr_R, density=density)
    model_reflex = SparseCfC(input_dim=d_in, hidden_dim=hidden_R, output_dim=d_out, adjacency_matrix=adj_reflex)
    
    print(f"Reflex Arc Synapses: {int(adj_reflex.sum().item())}")
    model_reflex = train_final_model(model_reflex, X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, epochs=400, 
                                     n_sensory=n_sen_R, n_process=n_pro_R)
                                     
    # Save Model
    torch.save(model_reflex.state_dict(), "data/f110_reflex_100.pt")
    # Export to Omnibit
    exporter.export(model_reflex, input_dim=d_in, d_model=hidden_R, output_dim=d_out, filename="f110_reflex_100.omnibit")
    print("✅ f110_reflex_100.omnibit exported successfully!")
    
    print("\n🏁 FINAL SHOWDOWN COMPLETE 🏁")
