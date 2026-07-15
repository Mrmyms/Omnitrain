import gym
import numpy as np
import yaml
import sys
import argparse
from argparse import Namespace
import torch
import os

sys.path.append(os.path.abspath('f1tenth_gym_repo/examples'))
from train_and_compare import ContinuousCfC, DiscreteRNN

def evaluate_models():
    # Load dataset stats for normalization
    stats = np.load("data/f110_real_stats.npz")
    mean_X = stats["mean"]
    std_X = stats["std"]
    
    d_in = 25 # (1 speed + 24 lidar rays)
    d_out = 2
    hidden = 32
    
    cfc = ContinuousCfC(input_dim=d_in, hidden_dim=hidden, output_dim=d_out, backbone_units=64)
    cfc.load_state_dict(torch.load("data/f110_real_cfc.pt"))
    cfc.eval()
    
    lstm = DiscreteRNN(d_in + 1, hidden, d_out, rnn_type='lstm')
    lstm.load_state_dict(torch.load("data/f110_real_lstm.pt"))
    lstm.eval()
    
    map_path = "data/maps/example_map"
    with open('f1tenth_gym_repo/examples/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
    
    for model_name, model in [("CfC", cfc), ("LSTM", lstm)]:
        obs, _, done, _ = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        
        # Maintain sliding windows of max 100 steps
        obs_window = []
        dt_window = []
            
        steps = 0
        ttf = 10000
        
        blackout = 0
        
        print(f"\nEvaluating {model_name} on Real F1TENTH Gym with Lethal Jitter...")
        while steps < ttf:
            raw_lidar = obs['scans'][0]
            downsampled_lidar = raw_lidar[::len(raw_lidar)//24][:24]
            state = obs['linear_vels_x'][0]
            
            # 60% probability of a burst blackout starting (LiDAR dies for 10 steps)
            if blackout == 0 and np.random.rand() < 0.05:
                blackout = np.random.randint(5, 15)
                
            if blackout > 0:
                downsampled_lidar = np.zeros_like(downsampled_lidar)
                blackout -= 1
                
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
                if model_name == "CfC":
                    out = model(x_tensor, dt_tensor)
                else:
                    x_dt = torch.cat([x_tensor, dt_tensor], dim=-1)
                    out = model(x_dt) 
            
            action = out[0, -1] # Take the prediction for the current (last) step
            
            speed = action[0].item()
            steer = action[1].item()
            
            obs, reward, done, info = env.step(np.array([[steer, speed]]))
            steps += 1
            
            if done:
                print(f"CRASH! {model_name} collided at step {steps}.")
                break
                
        if steps == ttf:
            print(f"SUCCESS! {model_name} survived the {ttf}-step lethal stress test.")

if __name__ == "__main__":
    evaluate_models()
