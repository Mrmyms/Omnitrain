# OmniTrain Arduino Library — Zero-Allocation Architecture

## 🚀 Overview
The OmniTrain inference engine has been fully refactored into a **Zero-Allocation** architecture. 
This means that **`std::vector` and dynamic memory (Heap) are completely eliminated** during the high-frequency inference loop (`Step()`).

By avoiding dynamic memory allocation, the ESP32 (and other microcontrollers) are completely immune to Heap fragmentation, preventing "Kernel Panic" crashes during 1000 Hz control loops.

---

## 🛠️ How to Port and Use the New API

Instead of the engine dynamically allocating an array and returning it, you must now pass a statically allocated buffer (array) to the engine where it will write the output forces/actions.

### ❌ The Old Way (DEPRECATED)
```cpp
// This causes heap fragmentation and crashes!
std::vector<float> action = engine_cfc.Step(state_vector, dt, sim_time);
```

### ✅ The New Way (Zero-Allocation)
```cpp
// 1. Pre-allocate a static array for the output (e.g. 2 forces: steering, speed)
float action[2] = {0.0f, 0.0f};

// 2. Pass the array as a pointer to the engine
engine_cfc.Step(state_vector, dt, sim_time, action);

// 3. Read the results directly from your array!
Serial.print(action[0]); // Steering
Serial.print(action[1]); // Speed
```

### Supported Engines:
The new signature has been applied to all three engines:
- **CfC / SparseCfC**: `engine_cfc.Step(sensors, dt, abs_time, action_array);`
- **GRU**: `engine_gru.Step(sensors, action_array);`
- **LSTM**: `engine_lstm.Step(sensors, action_array);`

## 🧠 Platform Support
Because it no longer relies heavily on the STL `std::vector`, this library is now dramatically more compatible with bare-metal C++ compilers, including:
- Espressif ESP32, ESP32-S3 (Xtensa & RISC-V)
- Raspberry Pi Pico (RP2040)
- STM32 Series (ARM Cortex-M)
- Arduino Nano 33 BLE

Enjoy your memory-safe, ultra-fast neural control!
