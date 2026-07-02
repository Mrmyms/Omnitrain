#ifndef OMNI_HAL_HPP
#define OMNI_HAL_HPP

// ============================================================
// OmniHAL: Raspberry Pi Pico (RP2040) Hardware Abstraction Layer
// ============================================================
// Strategy: BUFFERED LOAD via LittleFS.
// The RP2040 does NOT support memory-mapped arbitrary file access.
// Its Flash is XIP (Execute-In-Place) but only for code sections.
//
// Instead, we use the Pico SDK's LittleFS to read the .omnibit file
// into a single STATIC buffer in Flash-adjacent SRAM (SRAM5 bank).
//
// Toolchain: Raspberry Pi Pico SDK + littlefs-lib (PlatformIO: earle-philhower/RPi-Pico)
// ============================================================

#include <LittleFS.h>
#include <cstddef>
#include <cstring>

// Static buffer: OMNI_RP2040_MAX_BRAIN_KB controls the max model size in KB.
// Default: 192KB. Adjust downward to save SRAM for your application.
#ifndef OMNI_RP2040_MAX_BRAIN_KB
#define OMNI_RP2040_MAX_BRAIN_KB 192
#endif

static uint8_t _omni_rp2040_brain_buf[OMNI_RP2040_MAX_BRAIN_KB * 1024];

struct OmniHALResult {
    const unsigned char* data;
    size_t               length;
    bool                 ok;
};

inline OmniHALResult OmniHAL_LoadBrain(const char* path = "/bot_brain.omnibit") {
    OmniHALResult result = {nullptr, 0, false};

    if (!LittleFS.begin()) {
        return result; // LittleFS mount failed
    }

    File f = LittleFS.open(path, "r");
    if (!f) {
        return result; // File not found in flash partition
    }

    size_t file_size = f.size();
    size_t max_size  = sizeof(_omni_rp2040_brain_buf);

    if (file_size > max_size || file_size < 28) {
        f.close();
        return result; // File too large or too small for a valid .omnibit
    }

    // Copy the .omnibit bytes from LittleFS into the static SRAM buffer
    size_t bytes_read = f.read(_omni_rp2040_brain_buf, file_size);
    f.close();

    if (bytes_read != file_size) {
        return result; // Read error
    }

    result.data   = _omni_rp2040_brain_buf;
    result.length = file_size;
    result.ok     = true;
    return result;
}

#endif // OMNI_HAL_HPP
