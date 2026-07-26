import gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import serial
import argparse

sys.path.append(os.path.abspath('../paper_experiments/f1tenth_gym_repo/examples'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help="Path to .omnibit model to load dynamically")
    parser.add_argument('--episodes', type=int, default=1, help="Number of episodes to run for averaging")
    parser.add_argument('--log-csv', action='store_true', help="Append results to master_hil_summary.csv")
    parser.add_argument('--render', action='store_true', help="Save trajectory plot (Mac compatible)")
    args = parser.parse_args()

    import glob
    ports = glob.glob('/dev/cu.usbmodem*')
    if not ports:
        print("ERR: No ESP32 found on /dev/cu.usbmodem*")
        return
    # Prefer normal mode over bootloader mode (which might be a stale macOS file)
    port = next((p for p in ports if 'D40592796EE41' in p), ports[0])
    
    print(f"Connecting to ESP32 on {port}...")
    # DTR must be true for Native USB CDC, but RTS must be False to avoid bootloader
    ser = serial.Serial(port, 115200, timeout=2)
    ser.setDTR(True)
    ser.setRTS(False)
    time.sleep(1.0)
    
    # Force ESP32 to clear any pending buffers/crashed loads
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write(b"\n\n999.0\n")
    ser.flush()
    time.sleep(0.5)
    ser.reset_input_buffer()
    
    if args.model and os.path.exists(args.model):
        size = os.path.getsize(args.model)
        with open(args.model, 'rb') as f:
            payload = f.read()
        print(f"Loading dynamic model from {args.model}...")
        ser.reset_input_buffer()
        ser.write(f"LOAD:{size}\n".encode('utf-8'))
        
        start_wait = time.time()
        ack_received = False
        while time.time() - start_wait < 2.0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("ACK_LOAD:"):
                ack_received = True
                break
        
        if ack_received:
            print(f"ESP32 acknowledged load command. Sending {size} bytes...")
            ser.write(payload)
            start_wait = time.time()
            success = False
            while time.time() - start_wait < 5.0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line == "LOAD_OK":
                    success = True
                    break
                elif line.startswith("LOAD_ERR"):
                    print(f"ESP32 reported error: {line}")
                    break
            if success:
                print("Model loaded successfully onto ESP32.")
            else:
                print("Failed to load model. Exiting.")
                return
        else:
            print("ERR: Did not receive ACK from ESP32")
            return
        time.sleep(0.5)

    # Flush buffers
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    # Send reset command to trigger re-initialization if no dynamic model was loaded
    if not args.model:
        ser.write(b"999.0\n")
        time.sleep(0.5)
        ser.reset_input_buffer()
    
    # Load Stats
    stats = np.load("../paper_experiments/data/f110_real_stats.npz")
    mean_X_w = stats["mean"]
    std_X_w = stats["std"]
    
    # Run Simulation
    map_path = "../paper_experiments/data/maps/maps/vegas"
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
    
    # Initialize global accumulators
    total_avg_fitness = 0.0
    total_avg_distance = 0.0
    total_success_rate = 0.0

    plt.figure(figsize=(8,8))

    print(f"🏁 Starting ESP32 HIL Evaluation Race on Vegas ({args.episodes} episodes)...")
    
    for episode in range(args.episodes):
        obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
        
        steps = 0
        distance_traveled = 0.0
        steering_penalty = 0.0
        traj_x = []
        traj_y = []
        
        with open("hil_realtime.log", "a") as flog:
            flog.write(f"--- Episode {episode + 1}/{args.episodes} ---\n")
        print(f"--- Episode {episode + 1}/{args.episodes} ---")
        
        # Reset ESP32 internal memory (hidden states) for the new episode
        ser.reset_input_buffer()
        ser.write(b"999.0\n")
        ser.flush()
        
        # Wait for reset confirmation
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line == "F:0.0,0.0":
                break
            elif not line:
                ser.write(b"999.0\n")
                ser.flush()
        
        
        start_time = time.time()
        sim_time = 0.0
        dt = 0.05
        action = np.array([[0.0, 0.0]])
        trajectory_x = []
        trajectory_y = []
        
        while not done:
            trajectory_x.append(obs['poses_x'][0])
            trajectory_y.append(obs['poses_y'][0])
            
            raw_lidar = obs['scans'][0]
            downsampled_lidar = raw_lidar[::len(raw_lidar)//24][:24]
            state = obs['linear_vels_x'][0]
            
            # Include prev_steering as the 26th sensor
            prev_steering = float(action[0][0])
            x = np.hstack([[state], downsampled_lidar, [prev_steering]])
            if len(x) == len(mean_X_w) + 1:
                mean = np.append(mean_X_w, 0.0)
                std = np.append(std_X_w, 1.0)
                x_norm = (x - mean) / np.maximum(std, 1e-8)
            else:
                x_norm = (x - mean_X_w) / np.maximum(std_X_w, 1e-8)       
            sim_time += dt
            
            # Format as CSV and append dt and sim_time
            msg_parts = [f"{val:.4f}" for val in x_norm] + [f"{dt:.4f}", f"{sim_time:.4f}"]
            msg = ",".join(msg_parts) + "\n"
            
            ser.write(msg.encode('utf-8'))
            
            # Wait for response
            response = None
            retries = 0
            heartbeats = 0
            garbage = 0
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("F:"):
                    response = line[2:]
                    break
                elif line:
                    if not line.startswith("HEARTBEAT"):
                        with open("hil_realtime.log", "a") as flog:
                            flog.write(f"ESP32: {line}\n")
                        garbage += 1
                        if garbage > 50:
                            with open("hil_realtime.log", "a") as flog:
                                flog.write("ERR: Too much garbage. Aborting episode.\n")
                            done = True
                            break
                    else:
                        heartbeats += 1
                        if heartbeats > 15:
                            with open("hil_realtime.log", "a") as flog:
                                flog.write("ERR: ESP32 Deadlocked in HEARTBEAT loop. Aborting episode.\n")
                            done = True
                            break
                else:
                    retries += 1
                    if retries > 5:
                        print("ERR: ESP32 Deadlocked. Aborting episode.")
                        done = True
                        break
                    if retries > 2:
                        ser.write(msg.encode('utf-8'))
                        
            # Parse output (Steering, Throttle)
            try:
                steer, speed = map(float, response.split(','))
                action = np.array([[steer, speed]])
            except:
                print(f"Failed to parse: {response}")
                break
                
            speed = max(1.5, min(speed, 5.0))
            
            obs, reward, done, info = env.step(np.array([[steer, speed]]))
            
            
            steps += 1
            distance_traveled += speed * 0.05
            
            if args.render:
                traj_x.append(obs['poses_x'][0])
                traj_y.append(obs['poses_y'][0])
                
            steering_penalty += abs(steer) * 2.0
            
            if steps % 500 == 0:
                pass # print(f"Step {steps}: Distance = {distance_traveled:.1f}m, Laps = {env.lap_counts[0]:.2f}")
                
            if env.lap_counts[0] >= 5.0:
                print("🎉 5 LAPS COMPLETED BY ESP32!")
                break
                
            if done:
                print("💥 CRASHED!")
                break
                
        if args.render and len(traj_x) > 0:
            plt.figure(figsize=(8, 8))
            plt.plot(traj_x, traj_y, 'r-', linewidth=2, label='Trajectory')
            plt.plot(traj_x[0], traj_y[0], 'go', markersize=8, label='Start')
            if done:
                plt.plot(traj_x[-1], traj_y[-1], 'kx', markersize=12, label='Crash')
            plt.title(f"HIL Trajectory: {os.path.basename(args.model)}")
            plt.legend()
            out_name = f"traj_{os.path.basename(args.model)}.png"
            plt.savefig(out_name)
            print(f"📸 Saved trajectory plot to {out_name}")
            plt.close()
                
        # Calculate Fitness for episode
        fitness = (distance_traveled * 10.0) - steering_penalty
        if env.lap_counts[0] >= 1.0:
            lap_time = env.lap_times[0]
            fitness += 10000.0
            fitness += (10000.0 / max(1.0, lap_time))
        
        success_rate = min(500.0, env.lap_counts[0] * 100.0)
        print(f"  Result: Fitness={fitness:.1f}, Dist={distance_traveled:.1f}m, Laps={env.lap_counts[0]:.2f}")
        total_avg_fitness += fitness
        total_avg_distance += distance_traveled
        total_success_rate += success_rate
        
        # Save trajectory plot for the episode
        plt.plot(trajectory_x, trajectory_y, 'r-', linewidth=1.5, alpha=0.3)
        if episode == 0:
            plt.scatter([trajectory_x[0]], [trajectory_y[0]], c='green', marker='o', s=100, label='Start')
            plt.scatter([trajectory_x[-1]], [trajectory_y[-1]], c='blue', marker='x', s=100, label='End')
        else:
            plt.scatter([trajectory_x[-1]], [trajectory_y[-1]], c='blue', marker='x', s=30, alpha=0.5)

    # Save final overlaid plot
    plt.title(f"Trajectory ({args.model.split('/')[-1]}) - 20 Episodes")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"trajectory_{args.model.split('/')[-1]}.png", dpi=300, bbox_inches='tight')
    plt.close()

    env.close()
    
    # Calculate global averages
    avg_fitness = total_avg_fitness / args.episodes
    avg_distance = total_avg_distance / args.episodes
    avg_success = total_success_rate / args.episodes
    
    print("\n=== FINAL HIL AVERAGES (OVER 20 EPISODES) ===")
    print(f"Model: {args.model if args.model else 'Native C++ Hardcoded'}")
    print(f"Mean Fitness: {avg_fitness:.1f}")
    print(f"Mean Distance: {avg_distance:.1f} meters")
    print(f"Success Rate: {avg_success:.1f}%")

    if args.log_csv:
        csv_path = "master_hil_summary.csv"
        model_name = os.path.basename(args.model) if args.model else "UNKNOWN"
        line = f"{model_name},{avg_success:.1f},{avg_fitness:.1f},{avg_distance:.1f}\n"
        with open(csv_path, "a") as f:
            f.write(line)
        print(f"Logged to {csv_path}")

if __name__ == "__main__":
    main()
