#ifndef OMNI_HAL_HPP
#define OMNI_HAL_HPP

// ============================================================
// OmniHAL: NVIDIA Jetson (Linux + CUDA) Hardware Abstraction Layer
// ============================================================
// The Jetson HAL is functionally identical to the Raspberry Pi HAL
// (POSIX mmap for Zero-Copy file loading), but includes hints and
// utilities specific to the Jetson/CUDA ecosystem.
//
// Supported boards (tested / expected):
//   - Jetson Nano (4GB)     — 128-core Maxwell GPU
//   - Jetson Orin Nano (8GB) — 1024-core Ampere GPU
//   - Jetson Xavier NX       — 384-core Volta + DLAs
//
// Inference Modes (choose based on your use-case):
//   1. CPU-only (default): Use OmniEngine.hpp with this HAL.
//      Simple, deterministic, great for <500Hz control loops.
//
//   2. CUDA-accelerated (advanced): Replace OmniEngine::matmul()
//      with cuBLAS calls (see OmniEngineCUDA.hpp — future release).
//      Required for real-time video fusion (>60Hz camera processing).
//
// Toolchain: JetPack 5.x / 6.x (Ubuntu 20.04 / 22.04 aarch64)
// ============================================================

#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>

struct OmniHALResult {
    const unsigned char* data;
    size_t               length;
    bool                 ok;
    int                  _fd;
};

inline void OmniHAL_Unload(OmniHALResult& result) {
    if (result.ok && result.data != nullptr) {
        munmap(const_cast<unsigned char*>(result.data), result.length);
        close(result._fd);
        result.data = nullptr;
        result.ok   = false;
    }
}

inline OmniHALResult OmniHAL_LoadBrain(const char* path = "bot_brain.omnibit") {
    OmniHALResult result = {nullptr, 0, false, -1};

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[OmniHAL/Jetson] Cannot open brain file: %s\n", path);
        return result;
    }

    struct stat sb;
    if (fstat(fd, &sb) < 0 || sb.st_size < 28) {
        fprintf(stderr, "[OmniHAL/Jetson] Brain file too small or invalid: %s\n", path);
        close(fd);
        return result;
    }

    void* addr = mmap(nullptr, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "[OmniHAL/Jetson] mmap() failed for: %s\n", path);
        close(fd);
        return result;
    }

    // On Jetson, madvise MADV_WILLNEED pre-faults the pages into RAM immediately,
    // eliminating latency on the first OmniEngine::Load() call.
    madvise(addr, (size_t)sb.st_size, MADV_WILLNEED);

    result.data   = static_cast<const unsigned char*>(addr);
    result.length = (size_t)sb.st_size;
    result.ok     = true;
    result._fd    = fd;

    fprintf(stdout, "[OmniHAL/Jetson] Brain loaded: %.1f KB from %s\n",
            sb.st_size / 1024.0f, path);
    return result;
}

#endif // OMNI_HAL_HPP
