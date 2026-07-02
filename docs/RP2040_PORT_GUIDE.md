# Raspberry Pi Pico (RP2040) Port Guide

This guide explains how to deploy an OmniTrain-trained `.omnibit` brain on the **Raspberry Pi Pico / Pico W** (RP2040 dual-core Cortex-M0+).

## Prerequisites
- Raspberry Pi Pico or Pico W
- [PlatformIO](https://platformio.org/) with the `earle-philhower/raspberry-pi-pico` platform
- A trained `.omnibit` file exported from Python

## Memory Strategy (Buffered Load)
Unlike the ESP32, the RP2040 does **not** support arbitrary memory-mapped Flash access.
Instead, `OmniHAL.hpp` reads the `.omnibit` from a LittleFS partition into a **single static SRAM buffer** at startup.

| Resource | Usage |
| :--- | :--- |
| Flash (LittleFS) | Stores the `.omnibit` file |
| SRAM (Static Buffer) | One-time copy of weights (default: 192 KB) |
| SRAM (Inference Buffers) | ~8 KB for `OmniEngine` state/latent vectors |
| Core 0 | Sensor reading via `OmniTokenBus` |
| Core 1 | AI inference via `OmniEngine::Step()` |

## Folder Structure
```
your_pico_project/
├── platformio.ini
├── data/
│   └── bot_brain.omnibit      ← Upload to LittleFS
└── src/
    ├── main.cpp
    ├── OmniEngine.hpp          ← From src/cpp_engine/core/include/
    ├── OmniEngine.cpp          ← From src/cpp_engine/core/src/
    ├── OmniTokenBus.hpp        ← From src/cpp_engine/core/include/
    ├── OmniShield.hpp          ← From src/cpp_engine/core/include/ (optional)
    └── OmniHAL.hpp             ← From src/cpp_engine/hal/rp2040/
```

## `platformio.ini`
```ini
[env:pico]
platform  = https://github.com/maxgerhardt/platform-raspberrypi.git
board     = pico
framework = arduino
board_build.filesystem = littlefs
lib_deps  = 
    https://github.com/earlephilhower/arduino-littlefs-upload

build_flags = -DOMNI_MAX_DIM=128  ; Smaller model for RP2040's 264KB SRAM
```

## Upload the Brain to LittleFS
1. Place your `bot_brain.omnibit` in the `data/` folder.
2. In PlatformIO, run: **"Upload Filesystem Image"** (`pio run -t uploadfs`).

## `main.cpp` Example
```cpp
#include "OmniHAL.hpp"
#include "OmniEngine.hpp"
#include "OmniTokenBus.hpp"

OmniEngine engine;
OmniTokenBus<6> bus;  // e.g., 6-axis IMU

void setup() {
    // Core 0: Load brain from LittleFS into static SRAM buffer
    OmniHALResult brain = OmniHAL_LoadBrain("/bot_brain.omnibit");
    if (!brain.ok || !engine.Load(brain.data, brain.length)) {
        while (true); // Halt if brain is invalid
    }
    
    // Core 1: Run the AI inference loop
    multicore_launch_core1([]() {
        float abs_time = 0.0f;
        const float dt = 0.01f; // 100Hz
        
        while (true) {
            const float* sensors = bus.ReadSensors();
            auto action = engine.Step(sensors, dt, abs_time);
            abs_time += dt;
            
            // --- Apply action to motors here ---
            sleep_us(10000);
        }
    });
}

void loop() {
    // Core 0: Read sensors and push to the token bus
    float imu_data[6] = { read_ax(), read_ay(), read_az(),
                           read_gx(), read_gy(), read_gz() };
    bus.WriteSensors(imu_data);
    delay(5);
}
```
