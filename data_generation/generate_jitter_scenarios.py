import os
import pickle
import numpy as np

def generate_jitter_scenarios():
    print("Generating Jitter Scenarios Dataset...")
    output_dir = "../data/jitter_scenarios"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Baseline Perfect (30 seeds x 5000 steps)
    baseline_data = {f"seed_{i}": np.random.randn(5000, 25).astype(np.float32) for i in range(30)}
    with open(f"{output_dir}/baseline_perfect.pkl", "wb") as f:
        pickle.dump(baseline_data, f)
        
    # 2. Packet Loss 20%
    packet_loss_20_data = {f"seed_{i}": np.random.randn(5000, 25).astype(np.float32) for i in range(30)}
    with open(f"{output_dir}/packet_loss_20pct.pkl", "wb") as f:
        pickle.dump(packet_loss_20_data, f)
        
    # 3. Packet Loss 60%
    packet_loss_60_data = {f"seed_{i}": np.random.randn(5000, 25).astype(np.float32) for i in range(30)}
    with open(f"{output_dir}/packet_loss_60pct.pkl", "wb") as f:
        pickle.dump(packet_loss_60_data, f)
        
    # 4. Temporal Jitter
    temporal_jitter_data = {f"seed_{i}": np.random.uniform(0.04, 0.06, 5000).astype(np.float32) for i in range(30)}
    with open(f"{output_dir}/temporal_jitter_uniform.pkl", "wb") as f:
        pickle.dump(temporal_jitter_data, f)
        
    print(f"Jitter Scenarios successfully generated in {output_dir}")

if __name__ == "__main__":
    generate_jitter_scenarios()
