# NVIDIA Jetson Port Guide (Linux + CUDA)

## Overview
NVIDIA Jetson boards are purpose-built for AI Edge workloads. They are the recommended platform when you need real-time **visual perception** (cameras + LiDAR fusion) alongside the CfC brain.

## Supported Boards

| Board | CPU | GPU | RAM | Recommended Use |
| :--- | :--- | :--- | :--- | :--- |
| Jetson Nano | Cortex-A57 | 128-core Maxwell | 4 GB | Entry-level AI robots |
| Jetson Orin Nano | Cortex-A78 | 1024-core Ampere | 8 GB | Professional robots |
| Jetson Xavier NX | Cortex-A57×6 | 384-core Volta + DLA | 8 GB | Autonomous vehicles |
| Jetson AGX Orin | Cortex-A78×12 | 2048-core Ampere + DLA | 64 GB | Full autonomous systems |

## Inference Modes

### Mode 1: CPU-Only (Recommended for <500Hz control loops)
Use `OmniEngineNEON.hpp` with the Jetson HAL. The ARM NEON SIMD path gives excellent performance for most robotics applications with no CUDA overhead.

### Mode 2: CUDA-Accelerated (Future — for video fusion)
A CUDA-accelerated `OmniEngineCUDA.hpp` with `cuBLAS` matmul is planned for the next release. This will be needed for real-time multi-camera fusion at >60Hz.

## Setup (JetPack 5.x / Ubuntu 20.04)
```bash
# Install Python package
pip3 install omnitrain

# Build the C++ engine natively with NEON
g++ -std=c++17 -O3 -march=native \
    -o jetson_robot main.cpp OmniEngine.cpp \
    -I /path/to/omnitrain/src/cpp_engine/core/include
```

## `main.cpp` Example
```cpp
#include "hal/jetson/OmniHAL.hpp"
#include "OmniEngineNEON.hpp"   // ARM NEON matmul on Jetson's Cortex-A
#include "OmniTokenBus.hpp"
#include "OmniShield.hpp"

OmniEngineTarget engine;
OmniTokenBus<12> bus;           // 12-axis IMU + odometry
OmniShieldGuard shield(12, 6);  // 6 motor outputs

int main() {
    // Load brain — mmap() on Linux, Zero-Copy
    OmniHALResult brain = OmniHAL_LoadBrain("/opt/omnibot/brain.omnibit");
    if (!brain.ok || !engine.Load(brain.data, brain.length)) return 1;

    // Thread 0: Sensor acquisition (camera / LiDAR / IMU)
    std::thread sensor_thread([&]() {
        while (true) {
            float sensor_data[12] = { read_all_sensors() };
            bus.WriteSensors(sensor_data);
        }
    });

    // Thread 1: AI inference + safety + actuation
    float abs_time = 0.0f;
    while (true) {
        const float* sensors = bus.ReadSensors();
        auto intent  = engine.Step(sensors, 0.01f, abs_time);

        bool intervened;
        auto action  = shield.Enforce(sensors, intent, intervened);

        drive_motors(action);
        abs_time += 0.01f;
    }

    OmniHAL_Unload(brain);
}
```

## Training Directly on Jetson (Recommended for Sim-to-Real)
The Jetson has a CUDA GPU, which means you can run the full OmniTrain training loop directly on the robot without a separate PC:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"  # Should print True
python3 train_my_robot.py   # Full 5-phase curriculum on GPU
```
This is ideal for **Sim-to-Real transfer**: train in Isaac Sim on a PC, copy the `.omni` bundle to the Jetson, fine-tune for a few hours directly on the hardware, and deploy.
