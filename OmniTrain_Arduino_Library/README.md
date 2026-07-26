# OmniTrainEngine

OmniTrainEngine is a lightweight, bare-metal neural network inference engine written in C++ and optimized for microcontrollers like the ESP32. It was designed to run sequence models (LSTMs, GRUs, Continuous-time recurrent neural networks like CfC) with absolute determinism and zero-copy `.omnibit` payload parsing.

## Features
- **Dynamic HIL Loading**: Load new neural network weights over the Serial port dynamically without re-flashing the firmware.
- **Memory Safe**: Utilizes `std::vector` inside the execution pipeline to bypass RTOS task stack limits (e.g. 8KB limit in FreeRTOS `loop()` tasks), preventing Stack Overflows on large matrices.
- **Alignment Safe**: Forces 4-byte memory alignment for the payload buffers (`__attribute__((aligned(4)))`), preventing fatal `LoadStoreError` exceptions on Xtensa processors when casting bytes to `float` pointers.
- **Multipurpose**: The library is entirely decoupled from F1Tenth. The core inference engines (`OmniEngineLSTM`, `OmniEngineGRU`, `OmniEngineCFC`) simply take an array of `float` inputs and return an array of `float` outputs.

## Getting Started

### 1. Uploading the Weights (Dynamic Loading)
The library allows you to upload `.omnibit` payload files directly over Serial. To do this using Python, you can use the `pyserial` library. 

**Important for MacOS users**: When opening the serial port with `serial.Serial(..., timeout=2)`, PySerial will assert `DTR` by default. On some ESP32 setups (especially Native USB CDC), this causes the ESP32 to reboot! 
To safely upload weights, you should structure your Python code to either leave the port open during inference, or implement a wait loop:

```python
import serial
import time
import os

ser = serial.Serial('/dev/cu.usbmodem...', 115200, timeout=2)
ser.dtr = True
ser.rts = True
time.sleep(2.0) # Wait for ESP32 to recover if it reset

model_path = 'model.omnibit'
size = os.path.getsize(model_path)
with open(model_path, 'rb') as f:
    payload = f.read()

ser.reset_input_buffer()
ser.write(f"LOAD:{size}\n".encode('utf-8'))

# Wait for ACK
line = ser.readline().decode('utf-8').strip()
if line.startswith("ACK_LOAD:"):
    ser.write(payload)
    print("Model Sent!")
```

### 2. Running Inference in Arduino

```cpp
#include "OmniEngineLSTM.hpp"

OmniEngineLSTM engine_lstm;
__attribute__((aligned(4))) uint8_t dynamic_model_buffer[65536]; // 64KB

void setup() {
    Serial.begin(115200);
}

void loop() {
    // 1. Receive weights into dynamic_model_buffer (using your preferred serial logic)
    // 2. Load the model
    engine_lstm.Load(dynamic_model_buffer, payload_size);
    
    // 3. Run Inference
    float inputs[26] = {0.0}; // Your sensor data
    std::vector<float> action = engine_lstm.Step(inputs);
    
    // 4. Use action vector
    Serial.println(action[0]);
}
```

## Authors
- **Manuel Yobani Martinez Sanchez (Mr.Myms)**
