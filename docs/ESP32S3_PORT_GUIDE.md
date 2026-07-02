# ESP32-S3 Port Guide

## Overview
The ESP32-S3 is the recommended microcontroller for OmniTrain Edge deployment.

| Feature | ESP32 | ESP32-S3 |
| :--- | :--- | :--- |
| Core | Xtensa LX6 | **Xtensa LX7 @ 240 MHz** |
| SRAM | ~320 KB | **512 KB** |
| SIMD | ❌ | ✅ PIE Vector Extensions |
| USB-OTG | ❌ | ✅ Native USB (no CH340!) |
| AI Acceleration | ❌ | ✅ via ESP-DSP library |

## HAL: `hal/esp32s3/OmniHAL.hpp`
Identical loading strategy to ESP32 (SPIFFS buffered read), but the larger SRAM allows models up to **400 KB** by default.

## Accelerated Matmul: `OmniEngineS3.hpp`
Replace the standard `OmniEngine.hpp` with `OmniEngineS3.hpp` to enable SIMD-vectorized matrix multiplication via the ESP-DSP library.

### `platformio.ini`
```ini
[env:esp32s3]
platform  = espressif32
board     = esp32-s3-devkitc-1
framework = arduino
lib_deps  = 
    esp-dsp
build_flags = 
    -DOMNI_MAX_DIM=256
    -DARDUINO_USB_MODE=1    ; Enable native USB
board_build.flash_mode = qio
board_build.psram_type = opi
```

### `main.cpp`
```cpp
#include "hal/esp32s3/OmniHAL.hpp"
#include "OmniEngineS3.hpp"     // SIMD-accelerated
#include "OmniTokenBus.hpp"
#include "OmniShield.hpp"

OmniEngineTarget engine;        // Resolves to OmniEngineS3 on S3 builds
OmniTokenBus<6> bus;
OmniShieldGuard shield(6, 2);

void setup() {
    OmniHALResult brain = OmniHAL_LoadBrain("/bot_brain.omnibit");
    if (!brain.ok || !engine.Load(brain.data, brain.length)) {
        while (true) delay(500); // Halt
    }
}

void loop() {
    float sensors[6] = { read_imu() };
    bus.WriteSensors(sensors);

    const float* s = bus.ReadSensors();
    auto intent = engine.Step(s, 0.01f, millis() / 1000.0f);

    bool intervened;
    auto action = shield.Enforce(s, intent, intervened);

    set_motors(action[0], action[1]);
}
```
