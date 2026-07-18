import gym
import numpy as np
import sys
import torch
import os
import time
import serial

sys.path.append(os.path.abspath('../f1tenth_gym_repo/examples'))
sys.path.append(os.path.abspath('../../src'))
from omnitrain.sparse_cfc import SparseCfC
from topology_search_ncp import create_reflex_arc_mask

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
    model.load_state_dict(torch.load("../data/f110_reflex_qat_champion.pt", map_location='cpu'))
    
    hx = None
    
    print("Connecting to ESP32...")
    ser = serial.Serial('/dev/cu.usbmodemD40592796EE41', 460800, timeout=1)
    
    # Flush buffers
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    env = gym.make('f110_gym:f110-v0', map="../data/maps/maps/vegas", map_ext='.png', num_agents=1)
    obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    
    steps = 0
    print("Starting side-by-side comparison (Infinite Horizon Python vs ESP32)...")
    
    while steps < 20:
        raw_lidar = obs['scans'][0]
        downsampled_lidar = raw_lidar[::len(raw_lidar)//24][:24]
        state = obs['linear_vels_x'][0]
        
        x = np.hstack([[state], downsampled_lidar])
        x_norm = (x - mean_X_w) / std_X_w
        
        sim_time = steps * 0.05
        dt = 0.05
        msg_parts = [f"{val:.4f}" for val in x_norm] + [f"{dt:.4f}", f"{sim_time:.4f}"]
        ser.write((",".join(msg_parts) + "\n").encode('utf-8'))
        
        while True:
            response = ser.readline().decode('utf-8').strip()
            if not response: continue
            if "HEARTBEAT" in response: continue
            if response.startswith("F:"):
                response = response[2:]
            try:
                esp_steer, esp_speed = map(float, response.split(','))
                break
            except:
                pass
                
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
        dt_tensor = torch.tensor([[dt]], dtype=torch.float32).unsqueeze(0) 
        
        with torch.no_grad():
            out, hx = model(x_tensor, dt_tensor, hx)
        
        py_steer = out[0, 0, 0].item()
        py_speed = out[0, 0, 1].item()
        
        print(f"Step {steps:03d} | ESP32: [Steer: {esp_steer: .4f}, Speed: {esp_speed: .4f}] | PY: [Steer: {py_steer: .4f}, Speed: {py_speed: .4f}] | Diff: {abs(esp_steer - py_steer):.4f}")
        
        esp_speed = max(1.5, min(esp_speed, 5.0))
        obs, reward, done, info = env.step(np.array([[esp_steer, esp_speed]]))
        steps += 1
        
    ser.close()

if __name__ == "__main__":
    main()
