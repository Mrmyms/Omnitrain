import gym
import numpy as np
import yaml
import sys
import argparse
from argparse import Namespace
import torch
import os

sys.path.append(os.path.abspath('../f1tenth_gym_repo/examples'))
from train_and_compare import ContinuousCfC, DiscreteRNN

# Add src to path so we can import omnitrain
sys.path.append(os.path.abspath('../../src'))
from omnitrain.sparse_cfc import SparseCfC

def evaluate_models():
    stats = np.load("../data/f110_real_stats.npz")
    mean_X = stats["mean"]
    std_X = stats["std"]
    
    d_in = 25 
    d_out = 2
    hidden = 200
    
    # Initialize with dummy mask; load_state_dict will override it with the actual mask
    dummy_adj = torch.ones(200, d_in + 200)
    ncp = SparseCfC(input_dim=d_in, hidden_dim=200, output_dim=d_out, adjacency_matrix=dummy_adj)
    ncp.load_state_dict(torch.load("../data/f110_real_ncp.pt"))
    ncp.eval()
    
    cfc = ContinuousCfC(input_dim=d_in, hidden_dim=32, output_dim=d_out, backbone_units=64)
    cfc.load_state_dict(torch.load("../data/f110_real_cfc.pt"))
    cfc.eval()
    
    map_path = "../data/maps/example_map"
    with open('f1tenth_gym_repo/examples/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
    
    for model_name, model in [("SparseNCP", ncp), ("DenseCfC", cfc)]:
        obs, _, done, _ = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        
        obs_window = []
        dt_window = []
            
        steps = 0
        ttf = 10000
        
        blackout = 0
        
        print(f"\nEvaluating {model_name} on Real F1TENTH Gym with Sensor Noise...")
        while steps < ttf:
            raw_lidar = obs['scans'][0]
            downsampled_lidar = raw_lidar[::len(raw_lidar)//24][:24]
            state = obs['linear_vels_x'][0]
            
            # No artificial blackouts for this pure survival test
            if blackout > 0:
                pass
                
            x = np.hstack([[state], downsampled_lidar])
            x_norm = (x - mean_X) / std_X
            
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
            speed = action[1].item()
            
            obs, reward, done, info = env.step(np.array([[steer, speed]]))
            steps += 1
            
            if done:
                print(f"CRASH! {model_name} collided at step {steps}.")
                break
                
        if steps == ttf:
            print(f"SUCCESS! {model_name} survived the {ttf}-step lethal stress test.")

if __name__ == "__main__":
    evaluate_models()
