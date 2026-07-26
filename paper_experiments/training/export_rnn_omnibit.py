import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import sys

sys.path.append(os.path.abspath('../../src'))
from omnitrain.esp32_exporter import ESP32Exporter
from train_f110_ncp import load_dataset

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
        
    def apply_sparsity(self, amount=0.75):
        for name, module in self.rnn.named_modules():
            if isinstance(module, (nn.LSTM, nn.GRU)):
                prune.l1_unstructured(module, name='weight_ih_l0', amount=amount)
                prune.l1_unstructured(module, name='weight_hh_l0', amount=amount)

if __name__ == "__main__":
    X_tr, Y_tr, _, _, _, _, _, _ = load_dataset()
    d_in = X_tr.shape[2]
    d_out = Y_tr.shape[2]
    
    exporter = ESP32Exporter(output_dir="data/paper_baselines")
    
    models = [
        ("LSTM_Dense", 22, 'lstm', False),
        ("GRU_Dense", 25, 'gru', False),
        ("LSTM_Sparse", 45, 'lstm', True),
        ("GRU_Sparse", 50, 'gru', True)
    ]
    
    for name, hidden_dim, rnn_type, is_sparse in models:
        model = DiscreteRNN(d_in, hidden_dim, d_out, rnn_type=rnn_type)
        if is_sparse:
            model.apply_sparsity(0.75)
            
        pt_path = f"data/paper_baselines/{name}.pt"
        if os.path.exists(pt_path):
            model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
            exporter.export(model, input_dim=d_in, d_model=hidden_dim, output_dim=d_out, filename=f"{name}.omnibit")
            print(f"Exported {name} to omnibit.")
        else:
            print(f"Missing {pt_path}, skipping.")
