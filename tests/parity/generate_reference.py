"""
tests/parity/generate_reference.py
====================================
Generates a reference output file by running a known-weight model through
the Python LiquidFusionCore and saving the exact floating-point outputs.

The C++ parity test (test_parity_check.cpp) loads the same .omnibit and
must reproduce the same outputs within numerical tolerance (1e-4 relative error).

Usage:
    cd /path/to/Omnitrain
    pip install -e .
    python tests/parity/generate_reference.py
"""
import sys
import os
import struct
import torch
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from omnitrain.fusion_core import LiquidFusionCore
from omnitrain.esp32_exporter import ESP32Exporter

PARITY_DIR = os.path.dirname(__file__)
OMNIBIT_PATH = os.path.join(PARITY_DIR, "parity_brain.omnibit")
REFERENCE_PATH = os.path.join(PARITY_DIR, "reference_outputs.json")

INPUT_DIM = 4
D_MODEL   = 16
N_LATENTS = 1
BACKBONE  = 8

# Fixed seed for reproducibility
torch.manual_seed(42)

def main():
    print("[Parity] Creating model with fixed seed (42)...")
    model = LiquidFusionCore(
        n_latents=N_LATENTS,
        d_model=D_MODEL,
        input_dim=INPUT_DIM,
        config={'model': {'backbone_units': BACKBONE, 'use_spatial_mixer': False}}
    )
    model.eval()

    print("[Parity] Exporting to .omnibit...")
    exporter = ESP32Exporter(output_dir=PARITY_DIR)
    exporter.export(
        model,
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        output_dim=2,
        backbone_units=BACKBONE,
        filename="parity_brain.omnibit"
    )

    print("[Parity] Running forward steps...")
    reference = {
        "input_dim": INPUT_DIM,
        "d_model": D_MODEL,
        "n_latents": N_LATENTS,
        "steps": []
    }

    # Fixed sensor inputs for reproducibility
    test_inputs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.1, 0.2, 0.3, 0.4],
    ]

    state = torch.zeros(1, D_MODEL)
    abs_time = 0.0
    dt_val = 0.01

    with torch.no_grad():
        for i, inp in enumerate(test_inputs):
            sensors = torch.tensor([inp])
            # The third parameter of step_stateless is the absolute time input to CTE
            action, state = model.step_stateless(sensors, state, torch.tensor([abs_time]))
            action_sliced = action[:, :2] # Slice to output_dim=2 to match C++ behavior

            reference["steps"].append({
                "step": i,
                "input": inp,
                "dt": dt_val,
                "state_in":  state.tolist()[0],   # Previous state (before this step)
                "action":    action_sliced.tolist()[0],
                "state_out": state.tolist()[0],
            })
            print(f"  Step {i}: action[:4] = {action_sliced[0].tolist()}")
            abs_time += dt_val

    with open(REFERENCE_PATH, 'w') as f:
        json.dump(reference, f, indent=2)

    print(f"\n[Parity] ✅ Reference saved to: {REFERENCE_PATH}")
    print(f"[Parity] ✅ Brain saved to:      {OMNIBIT_PATH}")
    print()
    print("To run the C++ parity check:")
    print("  g++ -std=c++17 -O2 tests/parity/test_parity_check.cpp")
    print("       -I src/cpp_engine/core/include -DOMNI_MAX_DIM=64 -o /tmp/parity_check")
    print("  /tmp/parity_check tests/parity/parity_brain.omnibit tests/parity/reference_outputs.json")


if __name__ == "__main__":
    main()
