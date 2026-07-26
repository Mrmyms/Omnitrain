import gym
import numpy as np
import sys
import os
import time
import serial
import argparse

sys.path.append(os.path.abspath('../paper_experiments/f1tenth_gym_repo/examples'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help="Path to .omnibit model to load dynamically")
    args = parser.parse_args()

    print("Connecting to ESP32...")
    # DTR and RTS must be true for Native USB CDC
    ser = serial.Serial('/dev/cu.usbmodemD40592796EE41', 115200, timeout=2)
    ser.dtr = True
    ser.rts = True
    time.sleep(2.0)
    
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
    
    obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    
    steps = 0
    distance_traveled = 0.0
    steering_penalty = 0.0
    
    print("🏁 Starting ESP32 HIL Evaluation Race on Vegas...")
    start_time = time.time()
    sim_time = 0.0
    dt = 0.05
    action = np.array([[0.0, 0.0]])
    
    while not done:
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
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("F:"):
                response = line[2:]
                break
            elif line:
                pass # ignore HEARTBEAT or boot logs
            else:
                retries += 1
                if retries > 2:
                    ser.write(msg.encode('utf-8'))
                    retries = 0
                    
        # Parse output (Steering, Throttle)
        try:
            steer, speed = map(float, response.split(','))
        except:
            print(f"Failed to parse: {response}")
            break
            
        speed = max(1.5, min(speed, 5.0))
        
        obs, reward, done, info = env.step(np.array([[steer, speed]]))
        
        steps += 1
        distance_traveled += speed * 0.05
        steering_penalty += abs(steer) * 2.0
        
        if steps % 500 == 0:
            print(f"Step {steps}: Distance = {distance_traveled:.1f}m, Laps = {env.lap_counts[0]:.2f}")
            
        if env.lap_counts[0] >= 1.0:
            print("🎉 LAP COMPLETED BY ESP32!")
            break
            
        if done:
            print("💥 CRASHED!")
            break
            
    env.close()
    
    # Calculate Fitness
    fitness = (distance_traveled * 10.0) - steering_penalty
    if env.lap_counts[0] >= 1.0:
        lap_time = env.lap_times[0]
        fitness += 10000.0
        fitness += (10000.0 / max(1.0, lap_time))
    
    elapsed = time.time() - start_time
    print("\n--- ESP32 HIL RESULTS ---")
    print(f"Total Steps: {steps}")
    print(f"Distance Traveled: {distance_traveled:.1f} meters")
    print(f"Laps Completed: {env.lap_counts[0]:.4f}")
    success_rate = min(100.0, env.lap_counts[0] * 100.0)
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Fitness Points: {fitness:.1f}")
    print(f"Time Taken: {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()
