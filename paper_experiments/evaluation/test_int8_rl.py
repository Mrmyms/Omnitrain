import gym
import numpy as np
import yaml
import sys
import torch
import os
import time

sys.path.append(os.path.abspath('../f1tenth_gym_repo/examples'))
sys.path.append(os.path.abspath('../../src'))
from omnitrain.sparse_cfc import SparseCfC
from topology_search_ncp import create_reflex_arc_mask

def simulate_int8_quantization(model):
    """
    Simulates INT8 Post-Training Quantization by scaling and rounding all weights
    to 256 discrete levels, then scaling back to float for testing.
    """
    print("Quantizing model parameters to INT8...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Find the max absolute value to determine the dynamic range
            max_val = param.abs().max().item()
            if max_val == 0:
                continue
            
            # Map [-max_val, max_val] to [-127, 127]
            scale = max_val / 127.0
            
            # Quantize: divide by scale, round to nearest integer, clamp to INT8 range
            quantized = torch.round(param / scale).clamp(-127, 127)
            
            # Dequantize: multiply back by scale (this simulates what happens in hardware)
            dequantized = quantized * scale
            
            # Replace weights
            param.copy_(dequantized)
            
    return model

def main():
    d_in = 25
    d_out = 2
    hidden_R = 100
    n_sen_R = 50
    n_pro_R = 25
    n_hdr_R = 25
    
    # Load Stats
    stats = np.load("../data/f110_real_stats.npz")
    mean_X_w = stats["mean"]
    std_X_w = stats["std"]
    
    # Create Model
    base_mask = create_reflex_arc_mask(d_in, n_sen_R, n_pro_R, n_hdr_R, density=0.25)
    model = SparseCfC(input_dim=d_in, hidden_dim=hidden_R, output_dim=d_out, adjacency_matrix=base_mask)
    
    # Load Champion
    model_path = "../data/f110_reflex_qat_champion.pt"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print("✅ Loaded QAT Champion")
    
    # Quantize (Applies the INT8 grid restriction)
    model = simulate_int8_quantization(model)
    model.eval()
    print("✅ Model Quantized (Simulated INT8)")
    
    # Run Simulation
    map_path = "../data/maps/maps/vegas"
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
    
    obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    env.render() # Initialize renderer
    
    obs_window = []
    dt_window = []
    
    steps = 0
    distance_traveled = 0.0
    steering_penalty = 0.0
    
    print("🏁 Starting INT8 Evaluation Race...")
    start_time = time.time()
    
    while not done:
        raw_lidar = obs['scans'][0]
        downsampled_lidar = raw_lidar[::len(raw_lidar)//24][:24]
        state = obs['linear_vels_x'][0]
        
        x = np.hstack([[state], downsampled_lidar])
        x_norm = (x - mean_X_w) / std_X_w
        
        obs_window.append(x_norm)
        dt_window.append([0.05])
        
        if len(obs_window) > 100:
            obs_window.pop(0)
            dt_window.pop(0)
            
        x_tensor = torch.tensor(np.array(obs_window), dtype=torch.float32).unsqueeze(0)
        dt_tensor = torch.tensor(np.array(dt_window), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            out = model(x_tensor, dt_tensor)
        
        action = out[0, -1]
        steer = action[0].item()
        speed = max(1.5, min(action[1].item(), 5.0))
        
        obs, reward, done, info = env.step(np.array([[steer, speed]]))
        env.render(mode='human') # Real-time visualization!
        
        steps += 1
        distance_traveled += speed * 0.05
        steering_penalty += abs(steer) * 2.0
        
        if steps % 1000 == 0:
            print(f"Step {steps}: Distance = {distance_traveled:.1f}m, Laps = {env.lap_counts[0]:.2f}")
            
        if env.lap_counts[0] >= 1.0:
            print("🎉 LAP COMPLETED IN INT8!")
            break
            
        if done:
            print("💥 CRASHED!")
            break
            
    env.close()
    
    # Calculate Fitness
    fitness = (distance_traveled * 10.0) - steering_penalty
    if env.lap_counts[0] >= 1.0:
        lap_time = env.lap_times[0]
        fitness += 10000.0
        fitness += (10000.0 / max(1.0, lap_time))
    
    elapsed = time.time() - start_time
    print("\n--- RESULTS ---")
    print(f"Total Steps: {steps}")
    print(f"Distance Traveled: {distance_traveled:.1f} meters")
    print(f"Laps Completed: {env.lap_counts[0]}")
    print(f"Total Fitness Points: {fitness:.1f}")
    print(f"Time Taken: {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()
