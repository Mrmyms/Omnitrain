#ifndef ESP_OMNI_ENGINE_HPP
#define ESP_OMNI_ENGINE_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstddef>

class ESPOmniEngine {
public:
    ESPOmniEngine() : is_loaded(false), weights_ptr_(nullptr) {}

    // Carga los pesos mapeados en memoria Flash sin copiarlos a SRAM (Zero-Copy)
    // Devuelve true si la carga fue exitosa.
    bool Load(const unsigned char* omnibit_data, size_t length);

    // Bucle de inferencia. Devuelve la acción procesada por la red.
    std::vector<float> Step(const float* sensors, float dt, float abs_time);

    // Getters
    uint32_t GetInputDim() const { return input_dim_; }
    uint32_t GetModelDim() const { return d_model_; }
    uint32_t GetOutputDim() const { return output_dim_; }

private:
    bool is_loaded;
    
    // Dimensiones
    uint32_t input_dim_;
    uint32_t d_model_;
    uint32_t output_dim_;
    uint32_t backbone_units_;
    uint32_t total_weights_;
    
    // Búferes asignados estáticamente en clase (previene fragmentación OOM)
    // Máximos razonables (256) garantizan SRAM determinista constante
    float latents_[256];
    float state_buffer_[256]; 
    float b_state_[256];
    float b_time_[256];
    float x_in_[256 + 256]; // input_dim + d_model
    
    // Punteros a la Flash
    const float* weights_ptr_;

    // Punteros exactos a las matrices de la BioLiquidCell
    const float* sensory_w_;
    const float* sensory_b_;
    const float* state_w_;
    const float* state_b_;
    const float* time_w_;
    const float* time_b_;
    const float* ff1_w_;
    const float* ff1_b_;
    const float* ff2_w_;
    const float* ff2_b_;
    const float* time_a_w_;
    const float* time_a_b_;
    const float* time_b_w_;
    const float* time_b_b_;
    const float* time_scale_;

    // Tubería de Procesamiento Neuronal
    void apply_input_projection(const float* sensors);
    void add_temporal_encoding(float abs_time);
    void apply_bio_liquid_cell(float dt);
    
    // Utilidades matemáticas
    float lecun_activation(float x) const { return 1.7159f * std::tanh(0.666f * x); }
    float sigmoid(float x) const { return 1.0f / (1.0f + std::exp(-x)); }
    void matmul(const float* W, const float* b, const float* x, float* out, int rows, int cols);
};

#endif // ESP_OMNI_ENGINE_HPP
