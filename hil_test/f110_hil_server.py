#!/usr/bin/env python3
import serial
import time
import argparse
import numpy as np

def run_f110_hil(port, baudrate, num_samples):
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        ser.dtr = True
        ser.rts = True
        print(f"Connected to ESP32-S3 on {port} at {baudrate} baud.")
    except Exception as e:
        print(f"Failed to connect to Serial port: {e}")
        return

    print("Assuming ESP32-S3 is ready on native USB. Starting F1TENTH HIL Simulation.")

    latencies = []
    
    # 25 inputs: 21 LiDAR rays + 4 state variables (e.g. speed, steering_angle, etc.)
    input_dim = 25
    
    start_time = time.time()
    
    for step in range(num_samples):
        # Generate random dummy state (normalized)
        state = np.random.uniform(low=-1.0, high=1.0, size=(input_dim,))
        
        # Format as CSV
        msg = ",".join([f"{x:.4f}" for x in state]) + "\n"
        
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
    args = parser.parse_args()
    
    run_f110_hil(args.port, args.baud, args.samples)
