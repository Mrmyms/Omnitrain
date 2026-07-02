#ifndef OMNI_ENGINE_NEON_HPP
#define OMNI_ENGINE_NEON_HPP

// ============================================================
// OmniEngineNEON: ARM NEON SIMD Inference Engine
// ============================================================
// Accelerated matrix multiplication for ARM Cortex-A processors:
//   - Raspberry Pi 3/4/5 (Cortex-A53/A72/A76)
//   - NVIDIA Jetson Nano/Orin (Cortex-A57/A78)
//   - BeagleBone AI-64, Orange Pi 5, Rock Pi, etc.
//
// ARM NEON processes 4 floats simultaneously per instruction (128-bit).
// This gives a theoretical 4x speedup on the matmul hot loop.
//
// Usage: Replace #include "OmniEngine.hpp" with this file in your
//        Raspberry Pi / Jetson main.cpp.
// ============================================================

#include "OmniEngine.hpp"

#ifdef __ARM_NEON
#include <arm_neon.h>

class OmniEngineNEON : public OmniEngine {
protected:
    // NEON-vectorized dot product for a single row of W against vector x
    inline float dot_neon(const float* row, const float* x, int cols) const {
        float32x4_t acc = vdupq_n_f32(0.0f);
        int i = 0;

        // Process 4 floats at a time using 128-bit NEON registers
        for (; i <= cols - 4; i += 4) {
            float32x4_t a = vld1q_f32(row + i);
            float32x4_t b = vld1q_f32(x + i);
            acc = vmlaq_f32(acc, a, b); // acc += a * b (fused multiply-add)
        }

        // Horizontal sum: reduce float32x4 to scalar
        float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        float result = vget_lane_f32(vpadd_f32(sum2, sum2), 0);

        // Handle remaining elements (cols not divisible by 4)
        for (; i < cols; ++i) {
            result += row[i] * x[i];
        }
        return result;
    }

    void matmul_neon(const float* W, const float* b,
                     const float* x, float* out, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
            out[i] = dot_neon(&W[i * cols], x, cols)
                   + ((b != nullptr) ? b[i] : 0.0f);
        }
    }
};

using OmniEngineTarget = OmniEngineNEON;

#else
// Fallback on non-ARM platforms (x86 desktop simulation, etc.)
using OmniEngineTarget = OmniEngine;
#endif

#endif // OMNI_ENGINE_NEON_HPP
