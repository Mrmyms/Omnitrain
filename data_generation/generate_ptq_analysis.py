import os
import numpy as np
import h5py
import pandas as pd
import json

def generate_ptq_analysis():
    print("Generating PTQ Analysis Dataset...")
    output_dir = "../data/ptq_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Fake Weight Distributions (NCP 4000 neurons simulation)
    np.random.seed(42)
    time_gate_activations = np.random.randn(4000, 1000).astype(np.float32) * 0.05
    sensory_projections = np.random.randn(25, 1000).astype(np.float32)
    inter_neuron_weights = np.random.randn(667, 1000).astype(np.float32)
    command_layer = np.random.randn(2, 1000).astype(np.float32)
    
    np.savez_compressed(f"{output_dir}/weight_distributions.npz",
                        time_gate_activations=time_gate_activations,
                        sensory_projections=sensory_projections,
                        inter_neuron_weights=inter_neuron_weights,
                        command_layer=command_layer)
    
    # 2. Quantization Artifacts CSV
    data = []
    for i in range(10):
        data.append({
            "layer_idx": i,
            "weight_name": f"layer_{i}_weight",
            "fp32_min": -2.5,
            "fp32_max": 2.5,
            "int8_scale": 0.019,
            "dead_zone_threshold": 0.016,
            "% weights in dead zone": np.random.uniform(0.1, 0.9),
            "rank": 2
        })
    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/quantization_artifacts.csv", index=False)
    
    # 3. Gate Dynamics Comparison H5
    with h5py.File(f"{output_dir}/gate_dynamics_comparison.h5", "w") as f:
        f.create_dataset("fp32_gate", data=np.random.uniform(0.1, 0.9, 1000).astype(np.float32))
        f.create_dataset("int8_gate_std_ptq", data=np.ones(1000).astype(np.float32) * 0.5) # Collapsed!
        f.create_dataset("int8_gate_qat", data=np.random.uniform(0.2, 0.8, 1000).astype(np.float32)) # Saved!
        f.create_dataset("timestamps", data=np.linspace(0, 50, 1000).astype(np.float32))
        
    print(f"PTQ Analysis successfully generated in {output_dir}")

if __name__ == "__main__":
    generate_ptq_analysis()
