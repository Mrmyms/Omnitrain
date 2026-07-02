#ifndef OMNI_HAL_HPP
#define OMNI_HAL_HPP

// ============================================================
// OmniHAL: ESP32-S3 Hardware Abstraction Layer
// ============================================================
// The ESP32-S3 (Xtensa LX7) is a significant upgrade over the original
// ESP32 (LX6). Key improvements relevant to OmniTrain:
//
//  - 512 KB SRAM (vs. ~320 KB on ESP32)                 → Larger models fit
//  - Xtensa LX7 @ 240 MHz with PIE Vector Extensions    → Faster matmul
//  - Optional ESP-DSP library: dsps_dotprod_f32()       → SIMD dot-product
//
// This HAL is identical to the ESP32 HAL (SPIFFS buffered load).
// For vectorized inference, include OmniEngineS3.hpp instead of
// OmniEngine.hpp — it overrides matmul() with ESP-DSP SIMD calls.
//
// Toolchain: Arduino ESP32 core >= 2.0 (PlatformIO: espressif32)
// Board: esp32s3dev, waveshare_esp32_s3_lcd_147, etc.
// ============================================================

#include <SPIFFS.h>
#include <cstddef>

#ifndef OMNI_ESP32S3_MAX_BRAIN_BYTES
#define OMNI_ESP32S3_MAX_BRAIN_BYTES (400 * 1024) // 400 KB — safe for 512 KB SRAM
#endif

static uint8_t _omni_esp32s3_brain_buf[OMNI_ESP32S3_MAX_BRAIN_BYTES];

struct OmniHALResult {
    const unsigned char* data;
    size_t               length;
    bool                 ok;
};

inline OmniHALResult OmniHAL_LoadBrain(const char* path = "/bot_brain.omnibit") {
    OmniHALResult result = {nullptr, 0, false};

    if (!SPIFFS.begin(true)) {
        return result;
    }

    File f = SPIFFS.open(path, "r");
    if (!f) return result;

    size_t file_size = f.size();
    if (file_size > sizeof(_omni_esp32s3_brain_buf) || file_size < 28) {
        f.close();
        return result;
    }

    size_t bytes_read = f.read(_omni_esp32s3_brain_buf, file_size);
    f.close();

    if (bytes_read != file_size) return result;

    result.data   = _omni_esp32s3_brain_buf;
    result.length = file_size;
    result.ok     = true;
    return result;
}

#endif // OMNI_HAL_HPP
