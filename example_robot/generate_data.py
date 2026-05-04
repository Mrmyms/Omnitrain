import numpy as np
import csv
import time
import os

"""
SafeDelivery-Bot Dataset Generator (High-Fidelity Version)
Simulates a robot moving in an environment with high-dim sensors:
1. 32-beam Lidar (expanded columns).
2. 128-dim Vision embeddings (expanded columns).
3. Boolean Battery flag.
"""

def generate_robot_data(num_samples=1000):
    print(f"📡 Generating {num_samples} frames of INDUSTRIAL robot sensor logs...")
    
    file_path = 'robot_logs.csv'
    
    # Define Dimensions from config.yaml
    LIDAR_DIM = 32
    VISION_DIM = 128
    DRIVE_DIM = 2
    
    # Build Headers
    headers = ['timestamp']
    for i in range(LIDAR_DIM): headers.append(f'lidar_front_{i}')
    for i in range(VISION_DIM): headers.append(f'vision_embed_{i}')
    headers.append('battery_state')
    for i in range(DRIVE_DIM): headers.append(f'drive_control_{i}')
    headers.append('safety')
    
    with open(file_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for i in range(num_samples):
            timestamp = time.time() + (i * 0.1)
            
            # 1. Lidar (simulated 32 beams with spatial gradient)
            base_dist = np.random.uniform(0.1, 5.0)
            lidar_beams = base_dist + np.random.normal(0, 0.05, LIDAR_DIM)
            
            # 2. Vision Embed (simulated 128-dim vector)
            vision_vec = np.random.uniform(-1.0, 1.0, VISION_DIM)

            # 3. Battery (Boolean)
            battery_low = 1.0 if np.random.random() < 0.05 else 0.0
            
            # 4. Logic for Labels
            if base_dist < 0.2 or battery_low > 0.5:
                target_speed = [0.0, 0.0]
                safety_label = 1 # EMERGENCY
            else:
                target_speed = [0.5, np.random.uniform(-0.1, 0.1)] # Moving forward
                safety_label = 0 # SAFE
            
            # Build Row
            row = {'timestamp': timestamp}
            for j in range(LIDAR_DIM): row[f'lidar_front_{j}'] = lidar_beams[j]
            for j in range(VISION_DIM): row[f'vision_embed_{j}'] = vision_vec[j]
            row['battery_state'] = battery_low
            for j in range(DRIVE_DIM): row[f'drive_control_{j}'] = target_speed[j]
            row['safety'] = safety_label
            
            writer.writerow(row)
            
    print(f"✅ High-Fidelity Dataset saved to '{file_path}'")

if __name__ == "__main__":
    generate_robot_data()
