#!/usr/bin/env python3
"""
pc_hil_server.py — PC-side Hardware-in-the-Loop Simulation Runner

This script runs the CartPole physics engine on the PC and streams the
state variables via Serial to the ESP32-S3 microcontroller. The ESP32-S3
executes the continuous-time CfC neural network and returns the control force.
"""

import serial
import time
import argparse
import numpy as np
import sys
import os

# Add parent directory to path to import simulate_pendulum logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from paper_experiments.simulate_pendulum import generate_cartpole_data
except ImportError:
    print("ERR: Could not import physics engine from paper_experiments")
    sys.exit(1)

def run_hil_test(port, baudrate, num_samples, loss_prob, model_path=None):
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        # ESP32 Native USB CDC requires DTR to be asserted to know a terminal is connected
        ser.dtr = True
        ser.rts = True
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
            ser.write(f"LOAD:{size}\n".encode('utf-8'))
            
            # Wait for ACK
            ack = ser.readline().decode('utf-8').strip()
            if ack == f"ACK_LOAD:{size}":
                print(f"ESP32 acknowledged load command. Sending {size} bytes...")
                ser.write(payload)
                
                # Wait for OK
                status = ser.readline().decode('utf-8').strip()
                if status == "LOAD_OK":
                    print("Model loaded successfully onto ESP32.")
                else:
                    print(f"ERR: Failed to load model. ESP32 reported: {status}")
                    return
            else:
                print(f"ERR: Unexpected ACK from ESP32: {ack}")
                return
                
        except Exception as e:
            print(f"ERR: Could not read or send model payload: {e}")
            return

    # Wait for ESP32 to be READY (Disabled for native USB compatibility)
    # print("Waiting for ESP32 to initialize...")
    # while True:
    #     line = ser.readline().decode('utf-8').strip()
    #     if "READY" in line:
    #         print("ESP32-S3 Ready! Starting HIL Simulation.")
    #         break
    #     elif line:
    #         print(f"[ESP32] {line}")
    print("Assuming ESP32-S3 is ready on native USB. Starting HIL Simulation.")

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = (masspole + masscart)
    length = 0.5
    polemass_length = (masspole * length)
    dt = 0.02
    
    state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
    
    ttf = 0
    start_time = time.time()
    sim_time = 0.0
    force = 0.0
    
    for step in range(num_samples):
        x, x_dot, theta, theta_dot = state
        
        # Simulate packet loss (Zero-Order Hold)
        if np.random.rand() < loss_prob:
            # Packet dropped: ESP32 maintains its internal state. We don't send anything.
            pass
        else:
            # Send state to ESP32-S3, appending dt and sim_time
            msg = f"{x:.5f},{x_dot:.5f},{theta:.5f},{theta_dot:.5f},{dt:.5f},{sim_time:.5f}\n"
            ser.write(msg.encode('utf-8'))
            
            # Wait for ESP32-S3 to compute force
            force = 0.0
            retries = 0
            while True:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith("F:"):
                    force = float(line[2:])
                    break
                elif line:
                    print(f"[ESP32] {line}")
                else:
                    retries += 1
                    if retries > 2:
                        print("Resending lost serial packet...")
                        ser.write(msg.encode('utf-8'))
                        retries = 0
                
        force = np.clip(force, -10.0, 10.0)
        
        # Step physics
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
        xacc  = temp - polemass_length * thetaacc * costheta / total_mass
        
        x  = x + dt * x_dot
        x_dot = x_dot + dt * xacc
        theta = theta + dt * theta_dot
        theta_dot = theta_dot + dt * thetaacc
        
        sim_time += dt
        state = np.array([x, x_dot, theta, theta_dot])
        ttf += 1
        
        # Safety bounds check
        if abs(x) > 2.4 or abs(theta) > 0.2095:
            print(f"PENDULUM FAILED at step {ttf}.")
            break
            
    total_time = time.time() - start_time
    print(f"HIL Test Complete. Time-To-Failure (TTF): {ttf} steps.")
    print(f"Total physical time elapsed: {total_time:.2f}s")
    ser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HIL Serial Server for OmniTrain")
    parser.add_argument("--port", type=str, default="/dev/cu.usbserial-0001", help="Serial port of ESP32-S3")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--samples", type=int, default=5000, help="Max steps to simulate")
    parser.add_argument("--loss", type=float, default=0.20, help="Simulated Packet Loss (0.0 to 1.0)")
    parser.add_argument("--model", type=str, default=None, help="Path to .omnibit file to load dynamically")
    args = parser.parse_args()
    
    run_hil_test(args.port, args.baud, args.samples, args.loss, args.model)
