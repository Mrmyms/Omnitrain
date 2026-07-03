import serial
import struct
import numpy as np
import time
import os

PORT = "/dev/cu.usbserial-10" # Hardcoded based on earlier ls, could be passed as arg
BAUD = 115200

def run_hil(port, X, Y, T, title):
    print(f"\n--- Running HIL Test: {title} ---")
    
    try:
        ser = serial.Serial(port, BAUD, timeout=2.0)
    except Exception as e:
        print(f"Failed to open port {port}: {e}")
        return
        
    time.sleep(2) # Wait for ESP32 reset
    
    # Wait for ready signal
    ser.write(b'\n')
    ready = False
    for _ in range(20):
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if "OMNI_READY" in line:
            ready = True
            break
        elif "OMNI_ERROR" in line:
            print("ESP32 reported an error loading the model.")
            return
            
    if not ready:
        print("ESP32 did not send OMNI_READY. Is the firmware running?")
        return
        
    print("ESP32 Ready! Blasting dataset...")
    
    predictions = []
    start_t = time.time()
    
    # We will compute dt on the fly just like in train_and_compare.py
    for i in range(len(X)):
        if i == 0:
            dt = 0.0
        else:
            dt = float(T[i][0] - T[i-1][0])
            
        state = X[i] # 4 floats
        
        # Pack: dt, x1, x2, x3, x4
        packet = struct.pack('<fffff', dt, state[0], state[1], state[2], state[3])
        ser.write(packet)
        
        # Read prediction
        resp = ser.read(4)
        if len(resp) == 4:
            pred = struct.unpack('<f', resp)[0]
            predictions.append(pred)
        else:
            print(f"Failed to read prediction at step {i}")
            break
            
    end_t = time.time()
    ser.close()
    
    if len(predictions) > 0:
        mse = np.mean((np.array(predictions) - Y[:len(predictions), 0])**2)
        print(f"HIL MSE: {mse:.4f}")
        print(f"Inference FPS: {len(predictions)/(end_t - start_t):.2f}")
    else:
        print("No predictions received.")
    return mse

if __name__ == "__main__":
    print("Loading datasets...")
    # Go up one directory to find data
    os.chdir("..")
    X_0 = np.load("data/pendulum_X_0loss.npy")
    X_20 = np.load("data/pendulum_X_20loss.npy")
    X_60 = np.load("data/pendulum_X_60loss.npy")
    Y = np.load("data/pendulum_Y.npy")
    T = np.load("data/pendulum_T.npy")
    
    run_hil(PORT, X_0, Y, T, "0% Packet Loss")
    run_hil(PORT, X_20, Y, T, "20% Packet Loss")
    run_hil(PORT, X_60, Y, T, "60% Packet Loss")
