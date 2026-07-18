import gym
import numpy as np
import yaml
import sys
import argparse
from argparse import Namespace
import torch
import os
import time

sys.path.append(os.path.abspath('../f1tenth_gym_repo/examples'))
# Add src to path so we can import omnitrain
sys.path.append(os.path.abspath('../../src'))
from omnitrain.sparse_cfc import SparseCfC
from topology_search_ncp import create_reflex_arc_mask

def spectate_champion():
    stats = np.load("../data/f110_real_stats.npz")
    mean_X = stats["mean"]
    std_X = stats["std"]
    
    d_in = 25
    d_out = 2
    hidden_R = 100
    
    n_sen_R = 50
    n_pro_R = 25
    n_hdr_R = 25
    base_mask = create_reflex_arc_mask(d_in, n_sen_R, n_pro_R, n_hdr_R, density=0.25)
    
    model = SparseCfC(input_dim=d_in, hidden_dim=hidden_R, output_dim=d_out, adjacency_matrix=base_mask)
    
    map_path = "../data/maps/maps/vegas"
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
    
    print("🚗 INICIANDO MODO ESPECTADOR F1TENTH 🚗")
    print("Viendo a la Inteligencia Artificial evolucionar en tiempo real...")
    print("Presiona Ctrl+C en esta terminal para salir.")
    
    last_modified_time = 0
    
    while True:
        model_path = "../data/f110_reflex_rl_champion.pt"
        if not os.path.exists(model_path):
            print("Esperando a que la primera generación termine y guarde a un campeón...")
            time.sleep(2)
            continue
            
        current_mtime = os.path.getmtime(model_path)
        if current_mtime > last_modified_time:
            print("\n🧬 ¡NUEVO CAMPEÓN EVOLUTIVO DETECTADO! Cargando cerebro...")
            try:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                last_modified_time = current_mtime
            except Exception as e:
                time.sleep(0.5) # Prevent reading during write
                continue
                
        model.eval()
        obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
        
        obs_window = []
        dt_window = []
        steps = 0
        
        while not done:
            env.render(mode='human')
            
            raw_lidar = obs['scans'][0]
            downsampled_lidar = raw_lidar[::len(raw_lidar)//24][:24]
            state = obs['linear_vels_x'][0]
            
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
            
            speed = max(1.5, min(speed, 5.0))
            
            obs, reward, done, info = env.step(np.array([[steer, speed]]))
            steps += 1
            
        print(f"🏁 El coche chocó (o completó la carrera) en {steps} pasos.")
        print("Rebobinando y lanzando al siguiente campeón...")
        time.sleep(1)

if __name__ == "__main__":
    try:
        spectate_champion()
    except KeyboardInterrupt:
        print("\nEspectador cerrado.")
