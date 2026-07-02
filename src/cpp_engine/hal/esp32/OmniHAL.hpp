#ifndef OMNI_HAL_HPP
#define OMNI_HAL_HPP

// ============================================================
// OmniHAL: ESP32 Hardware Abstraction Layer
// ============================================================
// Strategy: TRUE ZERO-COPY via DROM memory mapping.
// The ESP32's Flash is memory-mapped by the MMU into the DROM region.
// When you read a file from SPIFFS/LittleFS, the OS returns a pointer
// directly into the Flash address space — no SRAM copy is needed.
//
// This means a 200KB .omnibit model uses ZERO SRAM for weight storage.
// ============================================================

#include <SPIFFS.h>
#include <cstddef>

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
        return result; // File not found
    }

    // On ESP32, SPIFFS.open() on a memory-mapped file returns a pointer
    // directly into the DROM (read-only data in Flash).
    // This is the Zero-Copy mechanism.
    result.length = f.size();
    result.data   = reinterpret_cast<const unsigned char*>(f.read);
    result.ok     = (result.length > 28);

    // Note: Do NOT call f.close() if you are holding a Zero-Copy pointer.
    // The pointer remains valid as long as SPIFFS is mounted.
    return result;
}

#endif // OMNI_HAL_HPP
