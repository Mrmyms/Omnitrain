# STM32 Port Guide (Cortex-M4/M7)

This guide explains how to deploy an OmniTrain-trained `.omnibit` brain on **STM32** microcontrollers (e.g., STM32F4, STM32H7, STM32G4).

## Prerequisites
- STM32 board (STM32F4xx recommended, 1MB+ Flash, 192KB+ SRAM)
- [PlatformIO](https://platformio.org/) with the `ststm32` platform, or STM32CubeIDE
- `arm-none-eabi-objcopy` (included in your ARM GCC toolchain)
- A trained `.omnibit` file from Python

## Memory Strategy (Zero-Copy via Linker Section)
STM32 has the most efficient deployment: the `.omnibit` is converted into a C object file at compile time via `arm-none-eabi-objcopy`. The weights are stored directly in Flash's `.rodata` section and the engine reads them via linker-generated symbols — **no runtime loading, no SRAM copy, no filesystem required.**

| Resource | Usage |
| :--- | :--- |
| Flash (.rodata) | `.omnibit` weights embedded at link time |
| SRAM (Inference Buffers) | ~8 KB for `OmniEngine` state/latent vectors |
| SRAM (Dynamic) | 0 bytes (no heap, no malloc) |

## Step 1: Convert `.omnibit` to a linkable object file
Run this once on your PC after training. Only needs to be re-run when you retrain the model.

```bash
arm-none-eabi-objcopy \
  -I binary \
  -O elf32-littlearm \
  --rename-section .data=.rodata,alloc,load,readonly,data,contents \
  bot_brain.omnibit bot_brain.o
```

Add `bot_brain.o` to your CMakeLists.txt or PlatformIO `build_src_filter`.

## Step 2: Folder Structure
```
your_stm32_project/
├── platformio.ini
└── src/
    ├── main.cpp
    ├── bot_brain.o             ← Compiled from .omnibit via objcopy
    ├── OmniEngine.hpp          ← From src/cpp_engine/core/include/
    ├── OmniEngine.cpp          ← From src/cpp_engine/core/src/
    ├── OmniTokenBus.hpp        ← From src/cpp_engine/core/include/
    └── OmniHAL.hpp             ← From src/cpp_engine/hal/stm32/
```

## `platformio.ini`
```ini
[env:nucleo_f446re]
platform  = ststm32
board     = nucleo_f446re
framework = arduino
build_flags = -DOMNI_MAX_DIM=256
```

## `main.cpp` Example
```cpp
#include "OmniHAL.hpp"
#include "OmniEngine.hpp"

OmniEngine engine;

void setup() {
    // Zero-Copy: OmniHAL returns a pointer directly into Flash
    OmniHALResult brain = OmniHAL_LoadBrain();
    
    if (!brain.ok || !engine.Load(brain.data, brain.length)) {
        // Blink error LED and halt
        while (true) { digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); delay(200); }
    }
}

void loop() {
    float sensors[8] = { read_all_sensors() };  // Replace with your sensors
    float dt         = 0.01f;
    float abs_time   = millis() / 1000.0f;
    
    auto action = engine.Step(sensors, dt, abs_time);
    
    // Apply actions to actuators
    set_motor_pwm(action[0], action[1]);
}
```

## Supported STM32 Families

| Family | Recommended Board | Flash | SRAM | Max Model Size |
| :--- | :--- | :--- | :--- | :--- |
| STM32F4 | Nucleo-F446RE | 512 KB | 128 KB | ~80 KB |
| STM32H7 | Nucleo-H743ZI | 2 MB | 1 MB | ~400 KB |
| STM32G4 | Nucleo-G474RE | 512 KB | 128 KB | ~80 KB |
| STM32F7 | Nucleo-F746ZG | 1 MB | 320 KB | ~200 KB |

> **Tip:** For `OMNI_MAX_DIM`, choose the smallest value that fits your model's `d_model` parameter. Every halving of `OMNI_MAX_DIM` saves ~16 KB of SRAM.
