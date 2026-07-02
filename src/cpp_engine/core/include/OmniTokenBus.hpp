#ifndef OMNI_TOKEN_BUS_HPP
#define OMNI_TOKEN_BUS_HPP

#include <cstdint>
#include <atomic>
#include <array>
#include <cstring>

// OmniTokenBus: Lock-Free Dual-Core Ping-Pong Buffer
// Designed for multi-core microcontrollers. Allows one core (sensor reader)
// to write continuously while the other core (AI inference) reads, with
// zero blocking and zero jitter via std::atomic operations.
//
// Platform compatibility:
//   ESP32 (Xtensa LX6, FreeRTOS):  ✅ Full Dual-Core (Core 0 writes, Core 1 infers)
//   RP2040 (ARM Cortex-M0+):       ✅ Full Dual-Core (Core 0 writes, Core 1 infers)
//   STM32 (Cortex-M4/M7):          ⚠️  Single-Core, safe for ISR-to-main-loop use
//   Desktop (x86/ARM64):           ✅ Full multi-thread (used in simulation/testing)
//
// Requirements: C++11 or higher (std::atomic).
template <size_t SENSOR_DIM>
class OmniTokenBus {
public:
    OmniTokenBus() : active_buffer_(0), latest_ready_(0) {
        buffer_a_.fill(0.0f);
        buffer_b_.fill(0.0f);
    }

    // [Core 0 / Sensor ISR] Writes new sensor data to the inactive buffer,
    // then atomically signals the AI core that it's ready.
    void WriteSensors(const float* new_data) {
        uint8_t inactive = 1u - active_buffer_.load(std::memory_order_acquire);
        float* target = (inactive == 0) ? buffer_a_.data() : buffer_b_.data();

        for (size_t i = 0; i < SENSOR_DIM; ++i) {
            target[i] = new_data[i];
        }

        latest_ready_.store(inactive, std::memory_order_release);
    }

    // [Core 1 / AI Loop] Atomically swaps to the freshest sensor buffer
    // and returns a pointer to it. No copy, no mutex, no blocking.
    const float* ReadSensors() {
        uint8_t ready = latest_ready_.load(std::memory_order_acquire);
        active_buffer_.store(ready, std::memory_order_release);
        return (ready == 0) ? buffer_a_.data() : buffer_b_.data();
    }

    // Utility: Returns the number of sensor channels this bus is configured for.
    static constexpr size_t GetDim() { return SENSOR_DIM; }

private:
    std::array<float, SENSOR_DIM> buffer_a_;
    std::array<float, SENSOR_DIM> buffer_b_;

    // std::atomic guarantees cross-core visibility without OS primitives.
    std::atomic<uint8_t> active_buffer_;
    std::atomic<uint8_t> latest_ready_;
};

#endif // OMNI_TOKEN_BUS_HPP
