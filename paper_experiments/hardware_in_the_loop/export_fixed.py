import torch
import sys
import os
sys.path.append(os.path.abspath('../../src'))
from omnitrain.sparse_cfc import SparseCfC
from topology_search_ncp import create_reflex_arc_mask
from test_int8_rl import simulate_int8_quantization

d_in = 25
d_out = 2
hidden_R = 100
n_sen_R = 50
n_pro_R = 25
n_hdr_R = 25

base_mask = create_reflex_arc_mask(d_in, n_sen_R, n_pro_R, n_hdr_R, density=0.25)
model = SparseCfC(input_dim=d_in, hidden_dim=hidden_R, output_dim=d_out, adjacency_matrix=base_mask)
model.load_state_dict(torch.load("../data/f110_reflex_qat_champion.pt", map_location='cpu'))

# APPLY INT8 SCALING EXACTLY AS IN TEST SCRIPT!
model = simulate_int8_quantization(model)

# Export to C header
from omnitrain.esp32_exporter import ESP32Exporter
exporter = ESP32Exporter()
exporter.export(model=model, input_dim=d_in, d_model=hidden_R, output_dim=d_out, backbone_units=hidden_R)

# Move the file to include folder
import shutil
shutil.copy("exports/model.h", "../hil_test/include/model.h")
print("Exported fixed model!")
