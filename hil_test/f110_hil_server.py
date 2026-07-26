#!/usr/bin/env python3
import serial
import time
import argparse
import numpy as np

def run_f110_hil(port, baudrate, num_samples, model_path=None, dt=0.05):
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        # ESP32 Native USB CDC requires DTR to be false to avoid reset
        ser.dtr = False
        ser.rts = False
        print(f"Connected to ESP32-S3 on {port} at {baudrate} baud.")
    except Exception as e:
        print(f"Failed to connect to Serial port: {e}")
        return

    # Wait for ESP32 to boot up
    time.sleep(2.0)
    
    if model_path:
        print(f"Loading dynamic model from {model_path}...")
        try:
            with open(model_path, "rb") as f:
                payload = f.read()
            size = len(payload)
            
            # Send LOAD command
            ser.reset_input_buffer()
            ser.write(f"LOAD:{size}\n".encode('utf-8'))
            
            # Wait for ACK
            ack_found = False
            for _ in range(10): # retry a few times
                ack = ser.readline().decode('utf-8', errors='ignore').strip()
                if ack == f"ACK_LOAD:{size}":
                    ack_found = True
                    break
                elif ack:
                    print(f"[ESP32] {ack}")
                    
            if ack_found:
                print(f"ESP32 acknowledged load command. Sending {size} bytes...")
                ser.write(payload)
                
                # Wait for OK
                status_found = False
                for _ in range(20):
                    status = ser.readline().decode('utf-8', errors='ignore').strip()
                    if status == "LOAD_OK":
                        status_found = True
                        break
                    elif status:
                        print(f"[ESP32] {status}")
                        
                if status_found:
                    print("Model loaded successfully onto ESP32.")
                else:
                    print("ERR: Failed to load model or timed out.")
                    return
            else:
                print("ERR: Did not receive ACK from ESP32")
                return
                
        except Exception as e:
            print(f"ERR: Could not read or send model payload: {e}")
            return

    print("Assuming ESP32-S3 is ready on native USB. Starting F1TENTH HIL Simulation.")

    latencies = []
    
    # 25 inputs: 21 LiDAR rays + 4 state variables (e.g. speed, steering_angle, etc.)
    input_dim = 25
    
    start_time = time.time()
    sim_time = 0.0
    
    for step in range(num_samples):
        # Use fixed physics timestep dt instead of serial latency
        sim_time += dt

        # Generate random dummy state (normalized)
        state = np.random.uniform(low=-1.0, high=1.0, size=(input_dim,))
        
        # Format as CSV and append dt and sim_time
        msg_parts = [f"{x:.4f}" for x in state]
        msg_parts.append(f"{dt:.5f}")
        msg_parts.append(f"{sim_time:.5f}")
        msg = ",".join(msg_parts) + "\n"
        
        t0 = time.time()
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
                print(f"[ESP32] {line}")
            else:
                retries += 1
                if retries > 2:
                    ser.write(msg.encode('utf-8'))
                    retries = 0
                    
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)
        
        # Parse output (Steering, Throttle)
        try:
            steering, throttle = map(float, response.split(','))
            if step % 100 == 0:
                print(f"Step {step:4d} | Latency: {latency_ms:5.2f} ms | Steering: {steering:6.3f} | Throttle: {throttle:6.3f}")
        except:
            print(f"Failed to parse response: {response}")

    total_time = time.time() - start_time
    ser.close()
    
    print("\n--- F1TENTH HIL Test Results ---")
    print(f"Total steps run: {num_samples}")
    print(f"Total time elapsed: {total_time:.2f} s")
    print(f"Average Frequency:  {num_samples/total_time:.2f} Hz")
    print(f"Mean Latency: {np.mean(latencies):.2f} ms")
    print(f"99th Pctl Latency: {np.percentile(latencies, 99):.2f} ms")
    print(f"Max Latency:  {np.max(latencies):.2f} ms")
    print(f"Flash Payload Size: ~14 KB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1TENTH HIL Serial Server")
    parser.add_argument("--port", type=str, default="/dev/cu.usbmodem101", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--samples", type=int, default=1000, help="Number of inferences to test")
    parser.add_argument("--model", type=str, default=None, help="Path to .omnibit file to load dynamically")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulated physics timestep (e.g. 0.05 for 20Hz LiDAR)")
    args = parser.parse_args()
    
    run_f110_hil(args.port, args.baud, args.samples, args.model, args.dt)
