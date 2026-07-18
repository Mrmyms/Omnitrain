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
    with torch.no_grad():
        for name, param in model.named_parameters():
            max_val = param.abs().max().item()
            if max_val == 0: continue
            scale = max_val / 127.0
            quantized = torch.round(param / scale).clamp(-127, 127)
            dequantized = quantized * scale
            param.copy_(dequantized)
    return model

def main():
    d_in = 25
    d_out = 2
    hidden_R = 100
    n_sen_R = 50
    n_pro_R = 25
    n_hdr_R = 25
    
    stats = np.load("../data/f110_real_stats.npz")
    mean_X_w = stats["mean"]
    std_X_w = stats["std"]
    
    base_mask = create_reflex_arc_mask(d_in, n_sen_R, n_pro_R, n_hdr_R, density=0.25)
    model = SparseCfC(input_dim=d_in, hidden_dim=hidden_R, output_dim=d_out, adjacency_matrix=base_mask)
    
    model_path = "../data/f110_reflex_qat_champion.pt"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = simulate_int8_quantization(model)
    model.eval()
    
    map_path = "../data/maps/maps/vegas"
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
    
    fitnesses = []
    distances = []
    
    print("Running 10 evaluations with random starting perturbations...")
    
    for i in range(10):
        # Add random noise to starting position (x, y) and heading (theta)
        # We assume map is wide enough to handle +/- 0.5m safely
        dx = np.random.uniform(-0.5, 0.5)
        dy = np.random.uniform(-0.5, 0.5)
        dtheta = np.random.uniform(-0.2, 0.2)
        
        obs, _, done, _ = env.reset(np.array([[dx, dy, dtheta]]))
        
        obs_window = []
        dt_window = []
        
        steps = 0
        distance_traveled = 0.0
        steering_penalty = 0.0
        
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
            
            steps += 1
            distance_traveled += speed * 0.05
            steering_penalty += abs(steer) * 2.0
            
            if done:
                break
                
        # Calculate Fitness
        fitness = (distance_traveled * 10.0) - steering_penalty
        if env.lap_counts[0] >= 1.0:
            lap_time = env.lap_times[0]
            fitness += 10000.0
            fitness += (10000.0 / max(1.0, lap_time))
            
        fitnesses.append(fitness)
        distances.append(distance_traveled)
        
        print(f"Run {i+1}/10: Fitness = {fitness:.1f}, Distance = {distance_traveled:.1f}m, Laps = {env.lap_counts[0]:.2f}, Start Offset: [{dx:.2f}, {dy:.2f}, {dtheta:.2f}]")
        
    mean_fitness = np.mean(fitnesses)
    std_fitness = np.std(fitnesses)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    print("\n=== FINAL RESULTS (10 RUNS) ===")
    print(f"Mean Fitness: {mean_fitness:.1f} ± {std_fitness:.1f}")
    print(f"Mean Distance: {mean_dist:.1f}m ± {std_dist:.1f}m")
    print(f"Max Fitness: {np.max(fitnesses):.1f}")
    print(f"Min Fitness: {np.min(fitnesses):.1f}")

if __name__ == "__main__":
    main()
