import gym
import numpy as np
import sys
import torch
import os

sys.path.append(os.path.abspath('../f1tenth_gym_repo/examples'))
sys.path.append(os.path.abspath('../../src'))
from omnitrain.sparse_cfc import SparseCfC

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

def run_eval(model, mean_X_w, std_X_w, env):
    obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
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
        
        if env.lap_counts[0] >= 1.0:
            break
            
    fitness = (distance_traveled * 10.0) - steering_penalty
    if env.lap_counts[0] >= 1.0:
        lap_time = env.lap_times[0]
        fitness += 10000.0
        fitness += (10000.0 / max(1.0, lap_time))
    return fitness

def main():
    d_in = 25
    d_out = 2
    
    stats = np.load("../data/f110_real_stats.npz")
    mean_X_w = stats["mean"]
    std_X_w = stats["std"]
    
    model_path = "../data/f110_20_10_20_rl_champion.pt"
    
    # Load exact structure
    dummy = SparseCfC(input_dim=d_in, hidden_dim=50, output_dim=d_out, adjacency_matrix=torch.ones(50, 75))
    dummy.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model_fp32 = SparseCfC(input_dim=d_in, hidden_dim=50, output_dim=d_out, adjacency_matrix=dummy.mask)
    model_fp32.load_state_dict(torch.load(model_path, map_location='cpu'))
    model_fp32.eval()
    
    model_int8 = SparseCfC(input_dim=d_in, hidden_dim=50, output_dim=d_out, adjacency_matrix=dummy.mask)
    model_int8.load_state_dict(torch.load(model_path, map_location='cpu'))
    model_int8 = simulate_int8_quantization(model_int8)
    model_int8.eval()
    
    map_path = "../data/maps/maps/vegas"
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
    
    fp32_fitnesses = []
    int8_fitnesses = []
    
    print("Evaluating FP32 Model...")
    for _ in range(5):
        fp32_fitnesses.append(run_eval(model_fp32, mean_X_w, std_X_w, env))
        
    print("Evaluating INT8 Model...")
    for _ in range(5):
        int8_fitnesses.append(run_eval(model_int8, mean_X_w, std_X_w, env))
        
    print(f"\nFP32 Mean Fitness: {np.mean(fp32_fitnesses):.1f}")
    print(f"INT8 Mean Fitness: {np.mean(int8_fitnesses):.1f}")

if __name__ == "__main__":
    main()
