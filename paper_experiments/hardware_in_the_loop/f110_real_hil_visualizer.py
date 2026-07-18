#!/usr/bin/env python3
import gym
import numpy as np
import yaml
import sys
import argparse
from argparse import Namespace
import torch
import os
import serial
import time

sys.path.append(os.path.abspath('../f1tenth_gym_repo/examples'))
# Add src to path so we can import omnitrain
sys.path.append(os.path.abspath('../../src'))

def run_hil_simulator(port="/dev/cu.usbmodem101", baudrate=115200):
    # Connect to ESP32
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        ser.dtr = True
        ser.rts = True
        print(f"Connected to ESP32-S3 HIL on {port} at {baudrate} baud.")
    except Exception as e:
        print(f"ERR: Could not connect to ESP32: {e}")
        print("Please check the port and ensure the ESP32 is out of bootloader mode.")
        return

    # Load Normalization Stats
    stats = np.load("../data/f110_real_stats.npz")
    mean_X = stats["mean"]
    std_X = stats["std"]
    
    # Setup Gym Environment
    map_path = "../data/maps/maps/vegas"
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
    obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    
    print("\n--- Starting REAL-TIME F1TENTH HIL SIMULATION ---")
    print("ESP32-S3 is driving the car!")
    
    steps = 0
    total_latency = 0.0
    
    while not done:
        # 1. Prepare LiDAR & State
        raw_lidar = obs['scans'][0]
        downsampled_lidar = raw_lidar[::len(raw_lidar)//24][:24]
        state_var = obs['linear_vels_x'][0]
        
        # 2. Normalize
        x = np.hstack([[state_var], downsampled_lidar])
        x_norm = (x - mean_X) / std_X
        
        # Send memory purge command every 100 steps to match the PyTorch sliding window
        if steps > 0 and steps % 100 == 0:
            ser.write("999.0\n".encode('utf-8'))
            time.sleep(0.05) # Brief pause for ESP32 to re-init engine
            
        # 3. Send to ESP32 (Append dt and sim_time at the end of the state vector)
        sim_time = steps * 0.05
        dt = 0.05
        msg_parts = [f"{val:.4f}" for val in x_norm] + [f"{dt:.4f}", f"{sim_time:.4f}"]
        msg = ",".join(msg_parts) + "\n"
        
        t0 = time.time()
        ser.write(msg.encode('utf-8'))
        
        # 4. Wait for ESP32 Response
        response = None
        retries = 0
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("F:"):
                response = line[2:]
                break
            elif line:
                # Silently ignore heartbeats to keep console clean
                pass
            else:
                retries += 1
                if retries > 2:
                    ser.write(msg.encode('utf-8'))
                    retries = 0
        
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0
        total_latency += latency_ms
        
        # 5. Parse Steering & Throttle
        try:
            steer, speed = map(float, response.split(','))
            speed = max(1.5, min(speed, 5.0)) # RL bounds
        except:
            print(f"Parse error: {response}")
            steer, speed = 0.0, 0.0
            
        # 6. Step Physics Simulator
        obs, reward, done, info = env.step(np.array([[steer, speed]]))
        
        # 7. Render GUI Frame
        env.render(mode='human')
        
        steps += 1
        if steps % 100 == 0:
            print(f"Step {steps:4d} | ESP32 Latency: {latency_ms:5.2f} ms | Steer: {steer:5.2f} | Throttle: {speed:5.2f}")
            
    avg_lat = total_latency / max(1, steps)
    print(f"\nCRASH or FINISH at step {steps}.")
    print(f"Average ESP32 Latency: {avg_lat:.2f} ms")
    ser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1TENTH Real-Time HIL")
    parser.add_argument("--port", type=str, default="/dev/cu.usbmodemD40592796EE41", help="Serial port")
    args = parser.parse_args()
    run_hil_simulator(args.port)
