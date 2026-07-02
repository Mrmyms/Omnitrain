# ESP32 Edge Port Guide (Zero-Copy Architecture)

Welcome to the **Omnitrain ESP32 Port Guide**. 
Deploying complex Continuous-Time Recurrent Neural Networks (like Liquid Time-Constant Networks or CfCs) onto 512KB microcontrollers is historically challenging.

Omnitrain v1.1.0 introduces a **Zero-Copy Engine** designed explicitly for low-power edge devices like the ESP32.

## The Memory Challenge
An ESP32 typically has:
- **SRAM**: ~520 KB (Where your variables live)
- **Flash (DROM)**: 4 MB to 16 MB (Where your code and constants live)

If you use standard ML libraries (like TensorFlow Lite for Microcontrollers), model weights are often copied or unpacked into SRAM. A moderate CfC architecture can consume 300KB of weights, causing an instant Out-Of-Memory (OOM) crash in SRAM.

## Zero-Copy Engine Solution
Our `ESPOmniEngine` completely bypasses SRAM for weight storage.
When you export your trained `LiquidFusionCore` via Python:
```python
exporter = ESP32Exporter(output_dir="exports")
exporter.export(model, input_dim=8, d_model=128, output_dim=4, filename="bot_brain.omnibit")
```
The exporter generates a `.omnibit` (V2 Structured) binary file.
This binary file maps the exact matrix shapes and offsets for the BioLiquidCell.

In your ESP32 (Arduino or ESP-IDF), you upload this file to the **SPIFFS / LittleFS** partition, or embed it directly as a C-array using `xxd`. 
When `ESPOmniEngine::Load()` is called, it does **not** copy the weights. It simply sets raw pointers to the memory-mapped Flash addresses.

## Core Features

### 1. Deterministic SRAM Allocation
The engine uses bounded, pre-allocated C-arrays (e.g., `float latents_[256];`). It guarantees:
- 0% Heap fragmentation (no `malloc` or `new` in the inference loop)
- Constant microsecond latency.

### 2. Ping-Pong Buffer (ESPTokenBus)
Since the ESP32 is dual-core, you can assign:
- **Core 0**: Reads I2C sensors (IMU, Encoders) at high frequency.
- **Core 1**: Runs `ESPOmniEngine::Step()` and `OmniShieldGuard::Enforce()`.

The `ESPTokenBus` uses a lock-free Ping-Pong Buffer architecture (`std::atomic`). This allows Core 0 to dump fresh telemetry without blocking Core 1, eliminating jitter.

### 3. OmniShield (Hardware Failsafe)
`OmniShieldGuard` is a deterministic wrapper that sits between the neural output and the actual motor drivers. It performs Tier-1 checks (absolute max/min RPMs) and overrides the neural network with a fallback action if boundary conditions are breached.

## Getting Started

1. Place `esp_omni_engine.hpp` and `esp_omni_engine.cpp` in your ESP32 project's `src` or `include` folder.
2. Export your `.omnibit` file from Python.
3. Use the `ESPTokenBus` to connect your sensor loops to the AI loop.

```cpp
#include "esp_omni_engine.hpp"
#include "OmniShield.hpp"

ESPOmniEngine engine;
OmniShieldGuard shield(8, 4);

void setup() {
    // Mount SPIFFS and get raw pointer, then:
    engine.Load(mapped_flash_pointer, file_size);
    shield.SetHardwareLimits(min_vec, max_vec);
}

void loop() {
    // 1. Get sensors (via TokenBus)
    float* sensors = bus.ReadSensors();
    
    // 2. Inference
    std::vector<float> intent = engine.Step(sensors, dt, abs_time);
    
    // 3. Shield
    bool intervened;
    std::vector<float> safe_action = shield.Enforce(sensors, intent, intervened);
    
    // 4. Actuate
    DriveMotors(safe_action);
}
```
