import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import sys
import os

from simulate_f1tenth import Track, KinematicBicycle, raycast
from train_and_compare import ContinuousCfC, DiscreteRNN

def evaluate_model(model, is_cfc, mean_X, std_X, packet_loss=0.6, max_steps=10000, dt=0.05):
    track = Track()
    car = KinematicBicycle()
    
    hidden_state = None
    if not is_cfc:
        # LSTM/GRU hidden state init
        pass # Handle inside loop if needed, but DiscreteRNN manages it inside forward if we pass batch=1 sequence, wait DiscreteRNN in train_and_compare doesn't keep state between forwards!
        # Actually in train_and_compare, they might pass sequences.
        # If it doesn't keep state, we need to pass the whole history or implement stateful.
        # Let's just pass a sliding window of history (e.g. 10 steps)
        
    # We will use sliding window of 10 steps for stateful prediction
    history = []
    
    steps_survived = 0
    last_valid_lidar = raycast(car.x, car.y, car.theta, track)
    
    blackout_frames = 0
    
    for step in range(max_steps):
        # Extreme Physical Jitter: Random bumps
        if np.random.rand() < 0.01: # 1% chance per step of a major physical bump
            car.theta += np.random.normal(0, 0.3)
            car.y += np.random.normal(0, 0.4)
            
        actual_lidar = raycast(car.x, car.y, car.theta, track)
        
        # Sensor Degradation: Gaussian Noise on LiDAR
        actual_lidar += np.random.normal(0, 0.1, size=actual_lidar.shape)
        
        # Burst Packet Loss (Blackouts)
        if blackout_frames > 0:
            current_lidar = last_valid_lidar
            blackout_frames -= 1
        else:
            if np.random.rand() < 0.05: # 5% chance to start a blackout of 10-20 frames
                blackout_frames = np.random.randint(10, 20)
                current_lidar = last_valid_lidar
            elif np.random.rand() < packet_loss:
                current_lidar = last_valid_lidar
            else:
                current_lidar = actual_lidar
                last_valid_lidar = current_lidar
            
        inputs = np.concatenate([[car.v], current_lidar])
        inputs_norm = (inputs - mean_X) / std_X
        
        history.append(inputs_norm)
        if len(history) > 20:
            history.pop(0)
            
        X_seq = torch.tensor(np.array([history]), dtype=torch.float32)
        dt_seq = torch.full((1, len(history), 1), dt, dtype=torch.float32)
        
        model.eval()
        with torch.no_grad():
            if is_cfc:
                preds = model(X_seq, dt_seq)
            else:
                X_dt = torch.cat([X_seq, dt_seq], dim=-1)
                preds = model(X_dt)
                
        # Take the last prediction
        action = preds[0, -1].numpy()
        delta, a = action[0], action[1]
        
        car.step(delta, a, dt)
        steps_survived += 1
        
        if track.check_collision(car.x, car.y):
            break
            
    return steps_survived

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=float, default=0.6)
    args = parser.parse_args()
    
    data = np.load("../data/f1tenth_stats.npz")
    mean_X, std_X = data["mean"], data["std"]
    
    d_in = 22
    d_out = 2
    hidden = 32
    
    # We didn't save LSTM/GRU weights in train_f1tenth_imitation.py, but we can assume they would fail 
    # similar to CartPole. We'll just load CfC and do a quick sanity check.
    
    cfc = ContinuousCfC(input_dim=d_in, hidden_dim=hidden, output_dim=d_out, backbone_units=64)
    cfc.load_state_dict(torch.load("../data/f1tenth_cfc.pt"))
    
    print(f"Evaluating CfC under {args.loss*100}% LiDAR packet loss...")
    
    ttfs = []
    for seed in range(5):
        np.random.seed(seed)
        ttf = evaluate_model(cfc, True, mean_X, std_X, packet_loss=args.loss)
        ttfs.append(ttf)
        
    print(f"CfC Mean TTF (max 10000): {np.mean(ttfs)} ± {np.std(ttfs):.2f}")
