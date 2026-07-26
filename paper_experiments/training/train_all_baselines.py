import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.utils.prune as prune
import numpy as np
import time
import os
import sys
import concurrent.futures

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from omnitrain.sparse_cfc import SparseCfC
from omnitrain.esp32_exporter import ESP32Exporter
from train_f110_ncp import load_dataset, create_advanced_layered_mask

class BaseRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rnn_type='lstm'):
        super().__init__()
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, times=None):
        out, _ = self.rnn(x)
        return self.fc(out)
        
    def apply_sparsity(self, amount=0.75):
        """Applies magnitude pruning to recurrent weights"""
        for name, module in self.rnn.named_modules():
            if isinstance(module, (nn.LSTM, nn.GRU)):
                prune.l1_unstructured(module, name='weight_ih_l0', amount=amount)
                prune.l1_unstructured(module, name='weight_hh_l0', amount=amount)

def train_and_export_model_worker(args):
    name, model, d_in, d_out, hidden_dim_for_export = args
    
    # Reload dataset in worker to avoid serialization issues
    X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, _, _ = load_dataset()
    epochs = 200
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_tr, Y_tr, dt_tr)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    dt_val = dt_val.to(device)
    
    print(f"--- Starting {name} ---")
    start_t = time.time()
    
    for ep in range(epochs):
        model.train()
        for bx, by, bdt in loader:
            bx, by, bdt = bx.to(device), by.to(device), bdt.to(device)
            optimizer.zero_grad()
            preds = model(bx, bdt) if 'CfC' in name else model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
            
        if (ep+1) % 50 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val, dt_val) if 'CfC' in name else model(X_val)
                val_loss = criterion(val_preds, Y_val)
            print(f"[{name}] Epoch {ep+1:03d}/{epochs} | Val MSE: {val_loss.item():.4f}", flush=True)

    elapsed = time.time() - start_t
    print(f"--- {name} Completed in {elapsed:.1f}s ---")
    
    # Save results to a dedicated directory
    out_dir = "data/paper_baselines"
    os.makedirs(out_dir, exist_ok=True)
    
    model.to("cpu")
    torch.save(model.state_dict(), f"{out_dir}/{name}.pt")
    
    if 'CfC' in name:
        exporter = ESP32Exporter(output_dir=out_dir)
        exporter.export(model, input_dim=d_in, d_model=hidden_dim_for_export, output_dim=d_out, filename=f"{name}.omnibit")
        print(f"[{name}] Exported .omnibit format.")

    return name

if __name__ == "__main__":
    X_tr, Y_tr, dt_tr, X_val, Y_val, dt_val, mean_X, std_X = load_dataset()
    print(f"Dataset loaded. Generating multicore jobs...")
    
    d_in = X_tr.shape[2]
    d_out = Y_tr.shape[2]
    
    os.makedirs("data/paper_baselines", exist_ok=True)
    
    # Define models
    model_lstm_dense = BaseRNN(d_in, hidden_dim=22, output_dim=d_out, rnn_type='lstm')
    model_gru_dense = BaseRNN(d_in, hidden_dim=25, output_dim=d_out, rnn_type='gru')
    
    hidden_dense_cfc = 25
    adj_dense = torch.ones(hidden_dense_cfc, d_in + hidden_dense_cfc)
    model_cfc_dense = SparseCfC(d_in, hidden_dense_cfc, d_out, adj_dense)
    
    model_lstm_sparse = BaseRNN(d_in, hidden_dim=45, output_dim=d_out, rnn_type='lstm')
    model_lstm_sparse.apply_sparsity(0.75)
    
    model_gru_sparse = BaseRNN(d_in, hidden_dim=50, output_dim=d_out, rnn_type='gru')
    model_gru_sparse.apply_sparsity(0.75)
    
    n_sen, n_pro, n_hdr = 50, 25, 25
    hidden_sparse = n_sen + n_pro + n_hdr
    adj_sparse = create_advanced_layered_mask(d_in, n_sen, n_pro, n_hdr, density=0.25)
    model_cfc_sparse = SparseCfC(d_in, hidden_sparse, d_out, adj_sparse)
    
    # Just run the sparse RNNs that failed to serialize in multiprocessing
    models = [
        ("LSTM_Sparse", model_lstm_sparse, d_in, d_out, 0),
        ("GRU_Sparse", model_gru_sparse, d_in, d_out, 0),
    ]
    
    for args in models:
        train_and_export_model_worker(args)
            
    print("All remaining baselines trained and exported.")
