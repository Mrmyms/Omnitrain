#ifndef OMNI_ENGINE_MIXED_HPP
#define OMNI_ENGINE_MIXED_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  OmniEngineMixed: Connectome-Guided Mixed-Precision Inference Engine
// ═══════════════════════════════════════════════════════════════════════════
//
//  Reads .omnibit V5 (OMNI\x05) binaries exported by the QAT-ES pipeline.
//  Each functional core of the NCP connectome runs at its own precision:
//
//    Core        | Params              | Typical Precision
//    ------------|---------------------|------------------
//    Sensory     | backbone[:, 0:I]    | INT4  (noise filter)
//    Inter       | g_w, g_b, h_w, h_b | INT8  (state memory)
//    Command     | fc_w, fc_b          | INT8  (motor output)
//    Timegate    | f_w, f_b            | FP16  (ODE solver)
//
//  Key design decisions for beating FedCFC:
//
//  1. ZERO-COPY DEQUANTIZATION: Compressed weights stay in Flash (DROM).
//     We dequantize on-the-fly into SRAM scratch buffers only during the
//     hot loop. This means the INT4 nibble-packed weights consume half the
//     Flash of INT8, and the total SRAM footprint is constant regardless
//     of model precision (only the scratch buffer size matters).
//
//  2. FUSED TIMEGATE PIPELINE: The f_weight/f_bias → sigmoid(-f*dt) path
//     is computed entirely in float32 arithmetic. Even when stored as FP16,
//     we upcast ONCE at load time into a dedicated SRAM buffer, eliminating
//     per-step conversion overhead. This guarantees zero temporal drift.
//
//  3. SENSORY NOISE GATE: INT4 dequantization of sensory weights acts as
//     a hardware-level noise filter. Values below the quantization threshold
//     (|x| < scale/16) are effectively zeroed, suppressing LiDAR noise
//     without any additional computation.
//
//  4. NO HEAP ALLOCATION: Every buffer is statically sized. The inference
//     hot loop triggers exactly 0 calls to malloc/new, preventing the
//     heap fragmentation that killed the GRU architecture.
//
//  Compatible with: ESP32-S3, ESP32-C3, RP2040, STM32F4+, any C++11 target.
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstddef>

// Maximum hidden dimension. Override in your build for smaller chips.
#ifndef OMNI_MAX_DIM
#define OMNI_MAX_DIM 64
#endif

// Maximum sensor input dimension
#ifndef OMNI_MAX_INPUT
#define OMNI_MAX_INPUT 32
#endif

// Maximum output dimension (e.g., [steering, throttle])
#ifndef OMNI_MAX_OUTPUT
#define OMNI_MAX_OUTPUT 4
#endif

// ─────────────────────────────────────────────────────────────────────────
//  Precision Types (matches Python PRECISION_LEVELS indices)
// ─────────────────────────────────────────────────────────────────────────
enum OmniDType : uint8_t {
    OMNI_INT4  = 0,
    OMNI_INT8  = 1,
    OMNI_FP16  = 2,
    OMNI_FP32  = 3,
};

// ─────────────────────────────────────────────────────────────────────────
//  Table-of-Contents Entry (matches Python exporter's per-tensor TOC)
// ─────────────────────────────────────────────────────────────────────────
struct OmniTOCEntry {
    uint32_t  data_size;   // Size in bytes of the packed data
    uint8_t   dtype_flag;  // OmniDType
    float     scale;       // Quantization scale (for INT4/INT8)
} __attribute__((packed));

// ─────────────────────────────────────────────────────────────────────────
//  Precision Map (4 genes packed into 2 bytes by the Python exporter)
// ─────────────────────────────────────────────────────────────────────────
struct OmniPrecisionMap {
    OmniDType sensory;
    OmniDType inter;
    OmniDType command;
    OmniDType timegate;
};

// ─────────────────────────────────────────────────────────────────────────
//  Mixed-Precision Tensor Descriptor
// ─────────────────────────────────────────────────────────────────────────
//  Points directly into Flash. Dequantization happens on-the-fly.
struct MixedTensor {
    const uint8_t* data;   // Raw pointer into Flash (DROM)
    float          scale;  // Quantization scale factor
    OmniDType      dtype;  // Precision type
    uint32_t       count;  // Number of logical elements
};


class OmniEngineMixed {
public:
    OmniEngineMixed() : is_loaded_(false) {
        memset(h_state_, 0, sizeof(h_state_));
    }

    // ─── Load ────────────────────────────────────────────────────────
    //  Parses the V5 .omnibit header and maps tensor pointers into Flash.
    //  For timegate tensors stored as FP16, immediately upcasts to the
    //  dedicated f32 SRAM buffer (one-time cost, ~60 bytes for 15 neurons).
    bool Load(const unsigned char* omnibit_data, size_t length);

    // ─── Step ────────────────────────────────────────────────────────
    //  Runs one inference step at your control loop frequency.
    //  sensors: raw float array of size GetInputDim()
    //  dt:      time delta in seconds since last Step() call
    //  output:  pre-allocated float array of size GetOutputDim()
    //
    //  Returns true on success. Writes action to output[].
    //  This API avoids std::vector to prevent any heap allocation.
    bool Step(const float* sensors, float dt, float* output);

    // ─── Getters ─────────────────────────────────────────────────────
    uint32_t GetInputDim()  const { return input_dim_; }
    uint32_t GetModelDim()  const { return hidden_dim_; }
    uint32_t GetOutputDim() const { return output_dim_; }
    bool     IsLoaded()     const { return is_loaded_; }

    // ─── Precision Introspection ─────────────────────────────────────
    OmniPrecisionMap GetPrecisionMap() const { return prec_map_; }

    // ─── Diagnostics ─────────────────────────────────────────────────
    //  Returns the SRAM usage in bytes (static allocation, constant).
    uint32_t GetSRAMUsage() const {
        return sizeof(h_state_) + sizeof(scratch_a_) + sizeof(scratch_b_)
             + sizeof(x_in_) + sizeof(f_w_f32_) + sizeof(f_b_f32_)
             + sizeof(output_buf_);
    }

private:
    bool is_loaded_;

    // ─── Dimensions ──────────────────────────────────────────────────
    uint32_t input_dim_;
    uint32_t hidden_dim_;
    uint32_t output_dim_;

    // ─── Precision Map ───────────────────────────────────────────────
    OmniPrecisionMap prec_map_;

    // ─── Tensor Descriptors (Zero-Copy pointers into Flash) ──────────
    //  Sensory backbone partition
    MixedTensor bb_sensory_;   // shape: [hidden, input]
    MixedTensor bb_recurrent_; // shape: [hidden, hidden]
    MixedTensor bb_bias_;      // shape: [hidden]

    //  Timegate (ODE solver)
    MixedTensor f_w_desc_;     // shape: [hidden]
    MixedTensor f_b_desc_;     // shape: [hidden]

    //  Inter-neuron state (memory)
    MixedTensor g_w_;          // shape: [hidden]
    MixedTensor g_b_;          // shape: [hidden]
    MixedTensor h_w_;          // shape: [hidden]
    MixedTensor h_b_;          // shape: [hidden]

    //  Command output
    MixedTensor fc_w_;         // shape: [output, hidden]
    MixedTensor fc_b_;         // shape: [output]

    // ─── SRAM Buffers (statically allocated, zero heap) ──────────────
    float h_state_[OMNI_MAX_DIM];                          // Hidden state (persistent)
    float scratch_a_[OMNI_MAX_DIM];                        // Backbone output buffer
    float scratch_b_[OMNI_MAX_DIM];                        // Temp computation buffer
    float x_in_[OMNI_MAX_INPUT + OMNI_MAX_DIM];           // Concatenated input
    float f_w_f32_[OMNI_MAX_DIM];                          // Timegate weights (pre-upcast)
    float f_b_f32_[OMNI_MAX_DIM];                          // Timegate bias (pre-upcast)
    float output_buf_[OMNI_MAX_OUTPUT];                    // Output buffer

    // ─── Dequantization Primitives ───────────────────────────────────

    // Dequantize a single element from a MixedTensor at index i
    inline float dequant(const MixedTensor& t, uint32_t i) const;

    // Dequantize a contiguous block into a float buffer
    void dequant_block(const MixedTensor& t, float* out, uint32_t start, uint32_t count) const;

    // ─── Math Primitives ─────────────────────────────────────────────

    // Mixed-precision dot product: dequantizes from tensor t on-the-fly
    float mixed_dot(const MixedTensor& t, uint32_t row, const float* x, uint32_t cols) const;

    // Mixed-precision matrix-vector multiply (dequantizes per row)
    void mixed_matvec(const MixedTensor& t, const MixedTensor& bias,
                      const float* x, float* out,
                      uint32_t rows, uint32_t cols) const;

    // Fused backbone computation: sensory + recurrent partitions
    void fused_backbone(const float* x_in, float* bb_out);

    // Standard sigmoid
    inline float sigmoid(float x) const {
        return 1.0f / (1.0f + expf(-x));
    }

    // IEEE 754 half-precision to float conversion (no FPU needed)
    static float fp16_to_f32(uint16_t h);
};

#endif // OMNI_ENGINE_MIXED_HPP
