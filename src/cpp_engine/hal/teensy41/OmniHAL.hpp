#ifndef OMNI_HAL_HPP
#define OMNI_HAL_HPP

// ============================================================
// OmniHAL: Teensy 4.1 (iMXRT1062) Hardware Abstraction Layer
// ============================================================
// The Teensy 4.1 is the most powerful hobbyist MCU available:
//   - ARM Cortex-M7 @ 600 MHz (with FPU and DSP instructions)
//   - 1 MB SRAM (512 KB DTCM + 512 KB OCRAM)
//   - 8 MB Flash on-board (optional 64 MB PSRAM/Flash via expansion pads)
//   - Built-in SD card slot (via SdFat library)
//
// Strategy: Load .omnibit from the onboard SD card into SRAM.
//
// Why SD card?
//   A 256-dim CfC model with backbone=128 requires ~1.5 MB of weights,
//   which exceeds Teensy's 1 MB SRAM. The SD card provides virtually
//   unlimited model size. The HAL reads the file into a pre-allocated
//   static buffer of configurable size.
//
// For models smaller than ~900 KB, you can skip the SD card entirely and
// embed the .omnibit as a C-array in program Flash (16 MB total Flash on Teensy 4.1).
// Use the STM32 HAL's objcopy technique for that approach.
//
// Toolchain: Teensyduino + PlatformIO (platform = teensy)
// Requires: SdFat library (included in Teensyduino)
// ============================================================

#include <SD.h>
#include <cstddef>
#include <cstdint>

// Teensy 4.1 built-in SD uses pin 10 by default in Arduino/SdFat
#ifndef OMNI_TEENSY_SD_CS_PIN
#define OMNI_TEENSY_SD_CS_PIN BUILTIN_SDCARD
#endif

#ifndef OMNI_TEENSY41_MAX_BRAIN_BYTES
#define OMNI_TEENSY41_MAX_BRAIN_BYTES (800 * 1024) // 800 KB — fits in 1MB SRAM
#endif

// Placed in EXTMEM (PSRAM) if available, otherwise DMAMEM (OCRAM)
DMAMEM static uint8_t _omni_teensy41_brain_buf[OMNI_TEENSY41_MAX_BRAIN_BYTES];

struct OmniHALResult {
    const unsigned char* data;
    size_t               length;
    bool                 ok;
};

inline OmniHALResult OmniHAL_LoadBrain(const char* path = "/omnibot/brain.omnibit") {
    OmniHALResult result = {nullptr, 0, false};

    if (!SD.begin(OMNI_TEENSY_SD_CS_PIN)) {
        Serial.println("[OmniHAL/Teensy] SD card mount failed.");
        return result;
    }

    File f = SD.open(path, FILE_READ);
    if (!f) {
        Serial.printf("[OmniHAL/Teensy] File not found: %s\n", path);
        return result;
    }

    size_t file_size = f.size();
    size_t max_size  = sizeof(_omni_teensy41_brain_buf);

    if (file_size > max_size || file_size < 28) {
        Serial.printf("[OmniHAL/Teensy] Invalid size: %zu bytes\n", file_size);
        f.close();
        return result;
    }

    size_t bytes_read = f.read(_omni_teensy41_brain_buf, file_size);
    f.close();

    if (bytes_read != file_size) {
        Serial.println("[OmniHAL/Teensy] SD read error — partial read.");
        return result;
    }

    Serial.printf("[OmniHAL/Teensy] Brain loaded: %.1f KB\n", file_size / 1024.0f);

    result.data   = _omni_teensy41_brain_buf;
    result.length = file_size;
    result.ok     = true;
    return result;
}

#endif // OMNI_HAL_HPP
