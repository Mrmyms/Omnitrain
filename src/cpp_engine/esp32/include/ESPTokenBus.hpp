#ifndef ESP_TOKEN_BUS_HPP
#define ESP_TOKEN_BUS_HPP

#include <cstdint>
#include <atomic>
#include <array>

// ESPTokenBus: Asynchronous Inter-Core Communication Architecture (Ping-Pong Buffers)
// Designed for ESP32. Allows Core 0 (I2C/Sensors) to write while Core 1 (AI) reads
// without relying on FreeRTOS Semaphores which would introduce jitter/latency.
template <size_t SENSOR_DIM>
class ESPTokenBus {
public:
    ESPTokenBus() : active_buffer_(0) {
        for (int i = 0; i < SENSOR_DIM; ++i) {
            buffer_a_[i] = 0.0f;
            buffer_b_[i] = 0.0f;
        }
    }

    // Called by Core 0 (I2C/Sensors). Writes to the currently inactive buffer.
    void WriteSensors(const float* new_data) {
        // Read which buffer the AI is NOT using
        uint8_t inactive = 1 - active_buffer_.load(std::memory_order_acquire);
        
        float* target_buffer = (inactive == 0) ? buffer_a_.data() : buffer_b_.data();
        
        for (size_t i = 0; i < SENSOR_DIM; ++i) {
            target_buffer[i] = new_data[i];
        }
        
        // Signal the AI that the inactive buffer now has the freshest data
        latest_ready_.store(inactive, std::memory_order_release);
    }

    // Called by Core 1 (AI Engine). Swaps active buffer to the freshest data.
    const float* ReadSensors() {
        uint8_t ready = latest_ready_.load(std::memory_order_acquire);
        
        // Lock the freshest buffer for AI use
        active_buffer_.store(ready, std::memory_order_release);
        
        return (ready == 0) ? buffer_a_.data() : buffer_b_.data();
    }

private:
    std::array<float, SENSOR_DIM> buffer_a_;
    std::array<float, SENSOR_DIM> buffer_b_;
    
    // std::atomic guarantees thread-safety across dual-cores without Mutex locks
    std::atomic<uint8_t> active_buffer_;
    std::atomic<uint8_t> latest_ready_;
};

#endif // ESP_TOKEN_BUS_HPP
