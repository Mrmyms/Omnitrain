#ifndef ESP_OMNI_ENGINE_HPP
#define ESP_OMNI_ENGINE_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstddef>

class ESPOmniEngine {
public:
    ESPOmniEngine() : is_loaded(false), weights_ptr_(nullptr) {}

    // Loads the weights mapped in Flash memory without copying them to SRAM (Zero-Copy)
    // Returns true if the load was successful.
    bool Load(const unsigned char* omnibit_data, size_t length);

    // Inference loop. Returns the action processed by the network.
    std::vector<float> Step(const float* sensors, float dt, float abs_time);

    // Getters
    uint32_t GetInputDim() const { return input_dim_; }
    uint32_t GetModelDim() const { return d_model_; }
    uint32_t GetOutputDim() const { return output_dim_; }

private:
    bool is_loaded;
    
    // Dimensions
    uint32_t input_dim_;
    uint32_t d_model_;
    uint32_t output_dim_;
    uint32_t backbone_units_;
    uint32_t total_weights_;
    uint8_t architecture_type_; // 0: CfC, 1: GRU, 2: Transformer
    
    // Statically allocated buffers (prevents OOM fragmentation)
    // Reasonable maximums (256) guarantee constant deterministic SRAM
    float latents_[256];
    float state_buffer_[256]; 
    float b_state_[256];
    float b_time_[256];
    float x_in_[256 + 256]; // input_dim + d_model
    
    // Pointers to Flash (DROM)
    const float* weights_ptr_;

    // Exact pointers to BioLiquidCell matrices
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

    // Pointers for GRU (Arch 1)
    const float* gru_w_ih_;
    const float* gru_w_hh_;
    const float* gru_b_ih_;
    const float* gru_b_hh_;

    // Pointers for Transformer (Arch 2)
    const float* trf_input_proj_w_;
    const float* trf_input_proj_b_;
    const float* trf_wq_w_;
    const float* trf_wq_b_;
    const float* trf_wk_w_;
    const float* trf_wk_b_;
    const float* trf_wv_w_;
    const float* trf_wv_b_;
    const float* trf_wo_w_;
    const float* trf_wo_b_;
    const float* trf_ffn1_w_;
    const float* trf_ffn1_b_;
    const float* trf_ffn2_w_;
    const float* trf_ffn2_b_;
    const float* trf_norm1_w_;
    const float* trf_norm1_b_;
    const float* trf_norm2_w_;
    const float* trf_norm2_b_;

    // Pointers for Full CfC (Arch 3)
    const float* cfc_bb_w_;
    const float* cfc_bb_b_;
    const float* cfc_f_w_;
    const float* cfc_f_b_;
    const float* cfc_g_w_;
    const float* cfc_g_b_;
    const float* cfc_h_w_;
    const float* cfc_h_b_;
    const float* cfc_fc_w_;
    const float* cfc_fc_b_;

    // Neural Processing Pipeline
    void apply_input_projection(const float* sensors);
    void add_temporal_encoding(float abs_time);
    void apply_bio_liquid_cell(float dt);
    void apply_gru_cell();
    void apply_transformer_layer();
    void apply_cfc_full_cell(const float* sensors, float dt);
    
    // Math Utilities
    float lecun_activation(float x) const { return 1.7159f * std::tanh(0.666f * x); }
    float sigmoid(float x) const { return 1.0f / (1.0f + std::exp(-x)); }
    void matmul(const float* W, const float* b, const float* x, float* out, int rows, int cols);
};

#endif // ESP_OMNI_ENGINE_HPP
