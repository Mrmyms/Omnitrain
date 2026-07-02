#ifndef OMNI_HAL_HPP
#define OMNI_HAL_HPP

// ============================================================
// OmniHAL: Raspberry Pi (Linux) Hardware Abstraction Layer
// ============================================================
// Strategy: TRUE ZERO-COPY via POSIX mmap().
//
// On Linux (Raspberry Pi, Ubuntu, etc.), the OS supports memory-mapped
// file I/O natively. mmap() maps the .omnibit file directly into the
// process's virtual address space. The CPU reads weights from disk
// cache pages — there is NO explicit copy into application memory.
//
// This is the most efficient loading strategy available on any platform:
//   - No SRAM buffer required (unlike RP2040/ESP32)
//   - OS handles caching and prefetching automatically
//   - Works on: Raspberry Pi Zero 2W, Pi 3, Pi 4, Pi 5, Orange Pi, Rock Pi
//
// Compatible with: any ARM Linux board (aarch64 or armv7l)
// Requires: POSIX-compliant OS (Linux, macOS for testing)
// ============================================================

#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct OmniHALResult {
    const unsigned char* data;
    size_t               length;
    bool                 ok;
    int                  _fd;    // Internal: keep fd open for mmap lifetime
};

// Call OmniHAL_Unload() when you no longer need the brain to free resources.
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
    if (fd < 0) return result; // File not found

    struct stat sb;
    if (fstat(fd, &sb) < 0 || sb.st_size < 28) {
        close(fd);
        return result;
    }

    // mmap: map the file into virtual memory. PROT_READ = read-only.
    // MAP_PRIVATE: changes don't affect the underlying file (safe for AI weights).
    void* addr = mmap(nullptr, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (addr == MAP_FAILED) {
        close(fd);
        return result;
    }

    // Hint to the OS that we will access this data sequentially (prefetch pages)
    madvise(addr, (size_t)sb.st_size, MADV_SEQUENTIAL);

    result.data   = static_cast<const unsigned char*>(addr);
    result.length = (size_t)sb.st_size;
    result.ok     = true;
    result._fd    = fd;
    return result;
}

#endif // OMNI_HAL_HPP
