import torch
import torch.nn as nn
from omnitrain.fusion_core import LiquidFusionCore
from omnitrain.esp32_exporter import ESP32Exporter
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_ablation_study():
    print("========================================")
    print("Omnitrain Architecture Ablation Study")
    print("========================================\n")
    
    # Common hyperparameters
    d_model = 32
    n_latents = 4
    input_dim = 16
    backbone_units = 64
    
    configs = {
        'CfC (BioLiquidCell)': {
            'inputs': [{'id': 'default', 'dim': input_dim}],
            'model': {
                'backbone_units': backbone_units,
                'use_spatial_mixer': False
            }
        },
        'GRU': {
            'inputs': [{'id': 'default', 'dim': input_dim}],
            'model': {
                'ablation': 'gru',
                'backbone_units': backbone_units,
                'use_spatial_mixer': False
            }
        },
        'Transformer': {
            'inputs': [{'id': 'default', 'dim': input_dim}],
            'model': {
                'ablation': 'transformer',
                'backbone_units': backbone_units,
                'use_spatial_mixer': False
            }
        }
    }
    
    dummy_input = torch.randn(2, input_dim) # Batch 2, dim 16
    dummy_dt = torch.tensor([0.1, 0.15]) # Irregular time steps
    
    exporter = ESP32Exporter(output_dir="ablation_exports")
    
    for name, config in configs.items():
        print(f"--- Building {name} ---")
        model = LiquidFusionCore(
            n_latents=n_latents,
            d_model=d_model,
            input_dim=input_dim,
            config=config
        )
        
        # 1. Parameter Count
        params = count_parameters(model)
        print(f"Total Parameters: {params:,}")
        
        # 2. Forward Pass Verification
        # Note: GRU and Transformer will ignore dt, while CfC uses it
        out = model(dummy_input, dummy_dt, modal_id="default")
        print(f"Output Shape: {out.shape} (Expected: 2, {n_latents}, {d_model})")
        
        # 3. Export to OmniEngine V4 format
        filename = f"{name.split(' ')[0].lower()}_ablation.omnibit"
        out_path = exporter.export(
            model=model,
            input_dim=input_dim,
            d_model=d_model,
            output_dim=d_model,
            backbone_units=backbone_units,
            filename=filename
        )
        
        print("\n")

if __name__ == "__main__":
    # Add root src directory to path if run from paper_experiments
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")
    run_ablation_study()
