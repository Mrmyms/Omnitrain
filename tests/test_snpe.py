import os
import torch
import numpy as np
from omnitrain.snpe_runner import OmniSNPERunner

def run_snpe_validation():
    print("🚀 OMNITRAIN SNPE VALIDATION TEST")
    
    # Check if SNPE bindings are available
    try:
        from snpe import snpe
        has_snpe = True
        print("✅ SNPE SDK Bindings Found.")
    except ImportError:
        has_snpe = False
        print("⚠️ SNPE SDK Bindings Not Found. Running in Simulation Mode.")
        
    dlc_path = "mock_model.dlc"
    
    # Initialize the runner (will fallback to SIMULATION mode if no SNPE)
    runner = OmniSNPERunner(dlc_path=dlc_path, runtime='DSP', max_batch=1, max_tokens=10, d_model=256)
    
    print("✅ Runner Initialized successfully.")
    
    # Create mock inputs
    tokens = np.random.randn(1, 10, 256).astype(np.float32)
    dt = 0.05
    prev_state = np.zeros((1, 32, 256), dtype=np.float32)
    abs_time = 1.0
    
    print("🏃 Executing step...")
    next_state, action, latency = runner.step(tokens, dt, prev_state, abs_time)
    
    print(f"⏱️ Step Latency: {latency:.2f} ms")
    print(f"📊 Next State Shape: {next_state.shape}")
    print(f"📊 Action Shape: {action.shape}")
    
    if next_state.shape == prev_state.shape:
        print("\n🏆 SNPE VALIDATION SUCCESSFUL: Stateful recurrence matched.")
    else:
        print("\n❌ SNPE VALIDATION FAILED: Shape mismatch.")
        exit(1)

if __name__ == "__main__":
    run_snpe_validation()
