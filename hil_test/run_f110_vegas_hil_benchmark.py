import gym
import numpy as np
import sys
import os
import time
import serial
import argparse
import csv

sys.path.append(os.path.abspath('../paper_experiments/f1tenth_gym_repo/examples'))

def run_evaluation(ser, env, mean_X_w, std_X_w, episode_idx, model_path):
    obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    
    steps = 0
    distance_traveled = 0.0
    steering_penalty = 0.0
    
    start_time = time.time()
    sim_time = 0.0
    dt = 0.05
    
    with open(model_path, "rb") as f:
        payload = f.read()
    
    size = len(payload)
    ser.reset_input_buffer()
    ser.write(f"LOAD:{size}\n".encode('utf-8'))
    
    ack_found = False
    for _ in range(10):
        ack = ser.readline().decode('utf-8', errors='ignore').strip()
        if ack == f"ACK_LOAD:{size}":
            ack_found = True
            break
            
    if ack_found:
        ser.write(payload)
        status_found = False
        for _ in range(20):
            status = ser.readline().decode('utf-8', errors='ignore').strip()
            if status == "LOAD_OK":
                status_found = True
                break
    
    while not done:
        raw_lidar = obs['scans'][0]
        downsampled_lidar = raw_lidar[::len(raw_lidar)//24][:24]
        state = obs['linear_vels_x'][0]
        
        x = np.hstack([[state], downsampled_lidar])
        x_norm = (x - mean_X_w) / std_X_w
        
        sim_time += dt
        
        msg_parts = [f"{val:.4f}" for val in x_norm] + [f"{dt:.4f}", f"{sim_time:.4f}"]
        msg = ",".join(msg_parts) + "\n"
        
        ser.write(msg.encode('utf-8'))
        
        response = None
        retries = 0
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("F:"):
                response = line[2:]
                break
            elif line:
                pass
            else:
                retries += 1
                if retries > 2:
                    ser.write(msg.encode('utf-8'))
                    retries = 0
                    
        try:
            steer, speed = map(float, response.split(','))
        except:
            break
            
        speed = max(1.5, min(speed, 5.0))
        
        obs, reward, done, info = env.step(np.array([[steer, speed]]))
        
        steps += 1
        distance_traveled += speed * dt
        steering_penalty += abs(steer) * 2.0
            
        if env.lap_counts[0] >= 1.0:
            break
            
        if done:
            break
            
    fitness = (distance_traveled * 10.0) - steering_penalty
    completed = False
    if env.lap_counts[0] >= 1.0:
        lap_time = env.lap_times[0]
        fitness += 10000.0
        fitness += (10000.0 / max(1.0, lap_time))
        completed = True
        
    return fitness, completed, steps, distance_traveled

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--csv", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    print("Connecting to ESP32...")
    ser = serial.Serial('/dev/cu.usbmodemD40592796EE41', 115200, timeout=2)
    ser.dtr = True
    ser.rts = True
    time.sleep(2.0)
    
    stats = np.load("../paper_experiments/data/f110_real_stats.npz")
    mean_X_w = stats["mean"]
    std_X_w = stats["std"]
    
    map_path = "../paper_experiments/data/maps/maps/vegas"
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
    
    results = []
    
    print(f"🏁 Starting ESP32 HIL Benchmark ({args.episodes} episodes)...")
    print(f"Model: {args.model}")
    
    for i in range(args.episodes):
        fit, comp, steps, dist = run_evaluation(ser, env, mean_X_w, std_X_w, i+1, args.model)
        status = "COMPLETED" if comp else "CRASHED"
        results.append({
            "Episode": i+1,
            "Status": status,
            "Fitness": round(fit, 2),
            "Distance(m)": round(dist, 2),
            "Steps": steps
        })
        icon = "✅" if comp else "💥"
        print(f"Episode {i+1:02d}: {icon} {status} | Fitness: {fit:8.1f} | Dist: {dist:6.1f}m | Steps: {steps}")
        
    env.close()
    ser.close()
    
    # Save CSV
    keys = results[0].keys()
    with open(args.csv, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
        
    fitnesses = [r["Fitness"] for r in results]
    completions = [1 for r in results if r["Status"] == "COMPLETED"]
    
    print("\n" + "="*40)
    print(f"🏆 FINAL BENCHMARK RESULTS ({os.path.basename(args.model)})")
    print("="*40)
    print(f"Episodes:      {args.episodes}")
    print(f"Success Rate:  {(sum(completions)/args.episodes)*100:.1f}%")
    print(f"Mean Fitness:  {np.mean(fitnesses):.1f} ± {np.std(fitnesses):.1f}")
    print(f"Max Fitness:   {np.max(fitnesses):.1f}")
    print(f"Min Fitness:   {np.min(fitnesses):.1f}")
    print(f"Data saved to: {args.csv}")
    print("="*40)

if __name__ == "__main__":
    main()
