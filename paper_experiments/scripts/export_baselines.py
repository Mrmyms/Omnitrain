import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.abspath('../../src'))
sys.path.append(os.path.abspath('../training'))

from omnitrain.esp32_exporter import ESP32Exporter
from train_and_compare import DiscreteRNN, ContinuousCfC

def main():
    d_in = 25
    d_out = 2
    hidden = 32
    
    # 0. Export Dense CfC
    cfc_model = ContinuousCfC(input_dim=d_in, hidden_dim=hidden, output_dim=d_out, backbone_units=64)
    cfc_path = "../data/f110_real_cfc.pt"
    if os.path.exists(cfc_path):
        cfc_model.load_state_dict(torch.load(cfc_path, map_location='cpu'))
        cfc_model.eval()
        
        exporter = ESP32Exporter(output_dir="../../omnibit_models")
        # ContinuousCfC takes raw d_in, not d_in+1 for the action
        exporter.export(cfc_model, input_dim=d_in, d_model=hidden, output_dim=d_out, filename="f110_real_cfc.omnibit")
        print("✅ Exported Dense CfC to omnibit")
    else:
        print(f"❌ Could not find {cfc_path}")

    # 1. Export GRU
    gru_model = DiscreteRNN(d_in + 1, hidden, d_out, rnn_type='gru')
    gru_path = "../data/f110_real_gru.pt"
    if os.path.exists(gru_path):
        gru_model.load_state_dict(torch.load(gru_path, map_location='cpu'))
        gru_model.eval()
        
        exporter = ESP32Exporter(output_dir="../../omnibit_models")
        exporter.export(gru_model, input_dim=d_in+1, d_model=hidden, output_dim=d_out, filename="f110_real_gru.omnibit")
        print("✅ Exported GRU to omnibit")
    else:
        print(f"❌ Could not find {gru_path}")

    # 2. Export LSTM
    lstm_model = DiscreteRNN(d_in + 1, hidden, d_out, rnn_type='lstm')
    lstm_path = "../data/f110_real_lstm.pt"
    if os.path.exists(lstm_path):
        lstm_model.load_state_dict(torch.load(lstm_path, map_location='cpu'))
        lstm_model.eval()
        
        exporter = ESP32Exporter(output_dir="../../omnibit_models")
        exporter.export(lstm_model, input_dim=d_in+1, d_model=hidden, output_dim=d_out, filename="f110_real_lstm.omnibit")
        print("✅ Exported LSTM to omnibit")
    else:
        print(f"❌ Could not find {lstm_path}")

if __name__ == "__main__":
    main()
