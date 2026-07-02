#ifndef ESP_TOKEN_BUS_HPP
#define ESP_TOKEN_BUS_HPP

#include <cstring>
#include <cstdint>

#define MAX_TOKEN_DIM 32

// Estructura aliniada a 4 bytes para DMA y caché de CPU
struct alignas(4) TokenFrame {
    float data[MAX_TOKEN_DIM];
    float timestamp;
    bool ready;
};

// ESPTokenBus implementando Ping-Pong Buffer (Doble Búfer)
// Esto permite que el Core 0 escriba por I2C/DMA al mismo tiempo
// que el Core 1 evalúa la inferencia, sin usar FreeRTOS Mutex (cero bloqueos).
class ESPTokenBus {
public:
    ESPTokenBus() : write_index_(0), read_index_(1) {
        buffers_[0].ready = false;
        buffers_[1].ready = false;
    }

    // Core 0 (Productor): Publicar nuevos sensores
    void Publish(const float* sensors, uint32_t dim, float timestamp) {
        if (dim > MAX_TOKEN_DIM) dim = MAX_TOKEN_DIM;
        
        TokenFrame& write_buf = buffers_[write_index_];
        std::memcpy(write_buf.data, sensors, dim * sizeof(float));
        write_buf.timestamp = timestamp;
        
        // Memory barrier implícita o semántica atómica de flag
        write_buf.ready = true;
        
        // Intercambio de búferes
        write_index_ = 1 - write_index_;
    }

    // Core 1 (Consumidor): Consumir marco para inferencia
    // Devuelve true si había un frame nuevo.
    bool Consume(float* out_sensors, uint32_t dim, float& out_timestamp) {
        // En una arquitectura ping-pong, leemos del búfer que el escritor NO está tocando.
        // Como el escritor siempre escribe en `write_index_`, leemos del opuesto.
        read_index_ = 1 - write_index_;
        TokenFrame& read_buf = buffers_[read_index_];

        if (read_buf.ready) {
            if (dim > MAX_TOKEN_DIM) dim = MAX_TOKEN_DIM;
            std::memcpy(out_sensors, read_buf.data, dim * sizeof(float));
            out_timestamp = read_buf.timestamp;
            read_buf.ready = false;
            return true;
        }
        
        return false;
    }

private:
    volatile uint8_t write_index_;
    uint8_t read_index_;
    TokenFrame buffers_[2];
};

#endif // ESP_TOKEN_BUS_HPP
