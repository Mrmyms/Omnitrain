#ifndef OMNI_HAL_HPP
#define OMNI_HAL_HPP

// ============================================================
// OmniHAL: ESP32 Hardware Abstraction Layer
// ============================================================
// Strategy: BUFFERED LOAD via SPIFFS/LittleFS.
//
// NOTE: The Arduino SPIFFS/LittleFS API on ESP32 does NOT expose
// a raw pointer to the Flash address space (DROM). The filesystem
// layer copies data through an internal buffer. The only way to
// achieve true Zero-Copy on ESP32 is to embed the .omnibit as a
// C-array in Flash (see STM32 HAL for that technique).
//
// For most use-cases, this SRAM buffer approach is completely fine:
// a 200KB model will consume 200KB of SRAM. If you need to save SRAM,
// use the STM32-style objcopy technique with the ESP-IDF's
// DRAM_ATTR / RODATA_ATTR linker attributes instead.
//
// Toolchain: Arduino ESP32 core (PlatformIO: espressif32)
// ============================================================

#include <SPIFFS.h>
#include <cstddef>

// Maximum model size in bytes (default: 200 KB).
// Reduce this to save SRAM if your model is smaller.
#ifndef OMNI_ESP32_MAX_BRAIN_BYTES
#define OMNI_ESP32_MAX_BRAIN_BYTES (200 * 1024)
#endif

// Static buffer to hold the loaded .omnibit weights in SRAM.
static uint8_t _omni_esp32_brain_buf[OMNI_ESP32_MAX_BRAIN_BYTES];

struct OmniHALResult {
    const unsigned char* data;
    size_t               length;
    bool                 ok;
};

inline OmniHALResult OmniHAL_LoadBrain(const char* path = "/bot_brain.omnibit") {
    OmniHALResult result = {nullptr, 0, false};

    if (!SPIFFS.begin(true)) {
        return result; // SPIFFS mount failed
    }

    File f = SPIFFS.open(path, "r");
    if (!f) {
        return result; // File not found in SPIFFS partition
    }

    size_t file_size = f.size();
    size_t max_size  = sizeof(_omni_esp32_brain_buf);

    if (file_size > max_size || file_size < 28) {
        f.close();
        return result; // File too large or too small to be a valid .omnibit
    }

    // Read the .omnibit into the static SRAM buffer
    size_t bytes_read = f.read(_omni_esp32_brain_buf, file_size);
    f.close();

    if (bytes_read != file_size) {
        return result; // Partial read error
    }

    result.data   = _omni_esp32_brain_buf;
    result.length = file_size;
    result.ok     = true;
    return result;
}

#endif // OMNI_HAL_HPP
