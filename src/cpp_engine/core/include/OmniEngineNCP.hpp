#ifndef OMNI_ENGINE_NCP_HPP
#define OMNI_ENGINE_NCP_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstddef>

#ifndef OMNI_MAX_DIM
#define OMNI_MAX_DIM 256
#endif

// OmniEngineNCP: Inference engine for Universal Sparse Connectomes (CSR format)
class OmniEngineNCP {
public:
    OmniEngineNCP() : is_loaded_(false) {}

    // Load the network from a .omnibit binary blob (Arch Flag 4)
    bool Load(const unsigned char* omnibit_data, size_t length);

    // Runs one inference step
    std::vector<float> Step(const float* sensors, float dt, float abs_time);

    uint32_t GetInputDim()  const { return input_dim_; }
    uint32_t GetModelDim()  const { return d_model_; }
    uint32_t GetOutputDim() const { return output_dim_; }
    bool     IsLoaded()     const { return is_loaded_; }

private:
    bool is_loaded_;
    uint32_t input_dim_;
    uint32_t d_model_;
    uint32_t output_dim_;

    float h_state_[OMNI_MAX_DIM];

    // Pointers to the CSR arrays for the backbone (Zero-Copy mapped from Flash)
    const float* cfc_bb_val_;
    const uint32_t* cfc_bb_col_;
    const uint32_t* cfc_bb_row_;

    // Dense vectors for element-wise heads
    const float* cfc_bb_b_;
    const float* cfc_f_w_;
    const float* cfc_f_b_;
    const float* cfc_g_w_;
    const float* cfc_g_b_;
    const float* cfc_h_w_;
    const float* cfc_h_b_;

    const float* fc_w_;
    const float* fc_b_;

    // Compressed Sparse Row (CSR) matrix-vector multiplication
    void sparse_mat_vec_mul(const float* val, const uint32_t* col, const uint32_t* row,
                            const float* x, float* out, int rows);
};

#endif // OMNI_ENGINE_NCP_HPP
