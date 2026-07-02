# Raspberry Pi Port Guide (Linux / ARM)

## Overview
The Raspberry Pi runs a full Linux OS, which means it can:
- Run the full OmniTrain Python trainer directly (`pip install omnitrain`)
- Use the C++ `OmniEngine` for real-time low-latency control loops
- Use `mmap()` for **true Zero-Copy** brain loading (better than any MCU)

## Supported Boards

| Board | CPU | RAM | Recommended Use |
| :--- | :--- | :--- | :--- |
| Pi Zero 2W | Cortex-A53 @ 1GHz | 512 MB | Nano drones, wearables |
| Pi 3B+ | Cortex-A53 @ 1.4GHz | 1 GB | Light robotics |
| Pi 4 | Cortex-A72 @ 1.8GHz | 2–8 GB | Full robot brains |
| Pi 5 | Cortex-A76 @ 2.4GHz | 4–8 GB | High-performance robots |

## Loading Strategy: POSIX `mmap()` (True Zero-Copy)
On Linux, the OS memory-maps the `.omnibit` file. The weights are **never** copied into application memory — the virtual memory system handles everything transparently.

## Build Instructions
```bash
# Install on the Pi directly
pip install omnitrain

# Or compile the C++ engine natively
g++ -std=c++17 -O3 -march=native \
    -o my_robot main.cpp OmniEngine.cpp \
    -I /path/to/omnitrain/src/cpp_engine/core/include
```

## `main.cpp` Example (Python process + C++ control loop)
```cpp
#include "hal/raspi/OmniHAL.hpp"
#include "OmniEngineNEON.hpp"   // ARM NEON vectorized matmul
#include "OmniShield.hpp"
#include <chrono>
#include <thread>

OmniEngineTarget engine;
OmniShieldGuard  shield(8, 4);

int main() {
    OmniHALResult brain = OmniHAL_LoadBrain("exports/bot_brain.omnibit");
    if (!brain.ok || !engine.Load(brain.data, brain.length)) {
        fprintf(stderr, "Failed to load brain!\n");
        return 1;
    }

    float abs_time = 0.0f;
    const float dt = 0.01f; // 100Hz

    while (true) {
        float sensors[8] = { read_all_sensors() };

        auto intent = engine.Step(sensors, dt, abs_time);

        bool intervened;
        auto action = shield.Enforce(sensors, intent, intervened);

        drive_actuators(action);

        abs_time += dt;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    OmniHAL_Unload(brain);
    return 0;
}
```

## Using the Full Python Trainer on the Pi 4/5
The Raspberry Pi 4/5 has enough RAM to run the Python training pipeline:
```bash
pip install omnitrain
python3 train_my_robot.py
```
This is particularly powerful for **incremental on-device learning**: after a session, the robot can retrain itself overnight on the Pi and wake up smarter.
