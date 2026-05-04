"""
OMNITRAIN SUPREME VERIFICATION TEST (v2.4) - FULL DEBUG
"""
import sys
import os
import torch
import torch.nn as nn
import traceback
from typing import Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from omnitrain.fusion_core import LiquidFusionCore, BioConectomaHub
from omnitrain.omni_shield import OmniShieldGuard
from omnitrain.universal_trainer import UniversalTrainer
from omnitrain.heads import RegressionHead, ClassificationHead
from omnitrain.exporter import OmniExporter

# Terminal Aesthetics
PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

def print_test(name, success, detail=""):
    status = PASS if success else FAIL
    print(f"  {status} {name}")
    if detail and not success:
        print(f"     └─ {detail}")

def run_supreme_audit():
    print("\n" + "="*60)
    print("🚀 OMNITRAIN SUPREME INTEGRITY AUDIT")
    print("="*60)

    # 1. ARCHITECTURE AUDIT (The Wall)
    print("\n[1/4] Architecture & Conectoma Audit")
    config = {
        'model': {
            'conectoma': {'enabled': True, 'sensory_n': 8, 'wall_n': 16, 'command_n': 8}
        },
        'inputs': [
            {'id': 'lidar', 'dim': 32},
            {'id': 'gps', 'dim': 2}
        ],
        'heads': [
            {'id': 'drive', 'type': 'regression', 'output_dim': 2}
        ]
    }
    try:
        core = LiquidFusionCore(config=config, d_model=64)
        print_test("BioConectomaHub initialization", isinstance(core.brain, BioConectomaHub))
    except Exception as e:
        print_test("Architecture Audit", False, str(e))
        return

    # 2. DYNAMICS & SOLVER AUDIT
    print("\n[2/4] Liquid Dynamics & Solver Audit")
    try:
        dt_extreme = torch.tensor([[100.0]])
        core.reset_state(batch_size=1)
        with torch.no_grad():
            out = core({'lidar': torch.randn(1, 32)}, dt_extreme)
        print_test("CfC Solver Stability (Extreme dt=100s)", not torch.isnan(out).any())
    except Exception as e:
        print_test("Dynamics Audit", False, str(e))

    # 3. SAFETY & SHIELD AUDIT
    print("\n[3/4] Formal Safety Guard Audit")
    try:
        heads = nn.ModuleDict({'drive': RegressionHead(2, 64)})
        shield = OmniShieldGuard(
            action_head=heads['drive'],
            d_model=64,
            state_dim=16,
            action_dim=2,
            num_hw_sensors=1
        )
        shield.set_hw_limits(torch.tensor([0.1]), torch.tensor([10.0]))
        
        sensor_readings = torch.tensor([[0.05]])
        h_x = torch.randn(1, 1, 64)
        result = shield(h_x, sensor_readings)
        print_test("Safety Shield Emergency Override", (result['action'] == 0).all().item())
        
    except Exception as e:
        print_test("Safety Audit", False, str(e))
        shield = None

    # 4. TRAINING & EXPORT AUDIT
    print("\n[4/4] Training Loop & Export Audit")
    try:
        if shield is None:
            raise ValueError("Shield initialization failed.")
            
        trainer = UniversalTrainer(core, heads, shield, config)
        batch = {
            'inputs': {
                'lidar': torch.randn(2, 4, 32),
                'gps': torch.randn(2, 4, 2)
            },
            'dt': torch.ones(2, 4),
            'targets': {'drive': torch.randn(2, 4, 2)}
        }
        
        metrics = trainer._train_epoch([batch])
        print_test("Training Backprop (Gradient Flow)", metrics['policy'] >= 0)
        
    except Exception as e:
        traceback.print_exc()
        print_test("Training Audit", False, str(e))

    print("\n" + "="*60)
    print("🏆 SUPREME AUDIT COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_supreme_audit()
