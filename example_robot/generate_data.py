import numpy as np
import csv
import time
import os

"""
SafeDelivery-Bot Dataset Generator (Zero-Dependency Version)
Simulates a robot moving in an environment:
1. Random Lidar distances.
2. Random Vision embeddings.
3. Random Battery flag.
"""

def generate_robot_data(num_samples=1000):
    print(f"📡 Generating {num_samples} frames of robot sensor logs...")
    
    file_path = '/Users/mr.myms/Omnitrain/example_robot/robot_logs.csv'
    
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['timestamp', 'lidar_min', 'battery', 'target_v', 'target_w', 'safety_label'])
        
        for i in range(num_samples):
            timestamp = time.time() + (i * 0.1)
            
            # 1. Lidar (32 beams)
            lidar = np.random.uniform(0.1, 5.0, 32).astype('float32')
            min_dist = np.min(lidar)
            
            # 2. Battery (Boolean)
            battery_low = 1.0 if np.random.random() < 0.05 else 0.0
            
            # 3. Logic for Labels
            if min_dist < 0.2 or battery_low > 0.5:
                target_speed = [0.0, 0.0]
                safety_label = 1 # EMERGENCY
            else:
                target_speed = [0.5, np.random.uniform(-0.1, 0.1)] # Moving forward
                safety_label = 0 # SAFE
            
            writer.writerow([timestamp, min_dist, battery_low, target_speed[0], target_speed[1], safety_label])
            
    print(f"✅ Dataset saved to '{file_path}'")

if __name__ == "__main__":
    generate_robot_data()
