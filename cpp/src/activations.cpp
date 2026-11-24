#include "activations.h"
#include <cmath>
#include <algorithm>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define HAS_WASM_SIMD 1
#else
#define HAS_WASM_SIMD 0
#endif

void Activations::apply(float* data, size_t size, ActivationType type) {
    switch (type) {
        case ActivationType::Sigmoid: {
            for (size_t i = 0; i < size; ++i) {
                // Clamped to avoid overflow/underflow
                float x = data[i];
                if (x > 45.0f) data[i] = 1.0f;
                else if (x < -45.0f) data[i] = 0.0f;
                else data[i] = 1.0f / (1.0f + std::exp(-x));
            }
            break;
        }
        case ActivationType::ReLU: {
            size_t i = 0;
#if HAS_WASM_SIMD
            v128_t vzero = wasm_f32x4_splat(0.0f);
            for (; i + 3 < size; i += 4) {
                v128_t val = wasm_v128_load(&data[i]);
                wasm_v128_store(&data[i], wasm_f32x4_max(val, vzero));
            }
#endif
            for (; i < size; ++i) {
                if (data[i] < 0.0f) data[i] = 0.0f;
            }
            break;
        }
        case ActivationType::LeakyReLU: {
            const float alpha = 0.01f;
            for (size_t i = 0; i < size; ++i) {
                if (data[i] < 0.0f) data[i] *= alpha;
            }
            break;
        }
        case ActivationType::Tanh: {
            for (size_t i = 0; i < size; ++i) {
                data[i] = std::tanh(data[i]);
            }
            break;
        }
        case ActivationType::Softmax: {
            if (size == 0) return;
            // Find max for numerical stability
            float maxVal = data[0];
            for (size_t i = 1; i < size; ++i) {
                if (data[i] > maxVal) maxVal = data[i];
            }
            float sum = 0.0f;
            for (size_t i = 0; i < size; ++i) {
                data[i] = std::exp(data[i] - maxVal);
                sum += data[i];
            }
            if (sum > 1e-7f) {
                float invSum = 1.0f / sum;
                for (size_t i = 0; i < size; ++i) {
                    data[i] *= invSum;
                }
            }
            break;
        }
    }
}

// Multiplies gradientInOut[i] *= derivative(activation(z)[i]) in-place
void Activations::applyDerivative(const float* activatedData, float* gradientInOut, size_t size, ActivationType type) {
    switch (type) {
        case ActivationType::Sigmoid: {
            // derivative = a * (1 - a)
            size_t i = 0;
#if HAS_WASM_SIMD
            v128_t vone = wasm_f32x4_splat(1.0f);
            for (; i + 3 < size; i += 4) {
                v128_t va = wasm_v128_load(&activatedData[i]);
                v128_t vg = wasm_v128_load(&gradientInOut[i]);
                v128_t vderiv = wasm_f32x4_mul(va, wasm_f32x4_sub(vone, va));
                wasm_v128_store(&gradientInOut[i], wasm_f32x4_mul(vg, vderiv));
            }
#endif
            for (; i < size; ++i) {
                float a = activatedData[i];
                gradientInOut[i] *= a * (1.0f - a);
            }
            break;
        }
        case ActivationType::ReLU: {
            // derivative = a > 0 ? 1 : 0
            for (size_t i = 0; i < size; ++i) {
                if (activatedData[i] <= 0.0f) {
                    gradientInOut[i] = 0.0f;
                }
            }
            break;
        }
        case ActivationType::LeakyReLU: {
            // derivative = a > 0 ? 1 : 0.01
            for (size_t i = 0; i < size; ++i) {
                if (activatedData[i] <= 0.0f) {
                    gradientInOut[i] *= 0.01f;
                }
            }
            break;
        }
        case ActivationType::Tanh: {
            // derivative = 1 - a^2
            for (size_t i = 0; i < size; ++i) {
                float a = activatedData[i];
                gradientInOut[i] *= (1.0f - a * a);
            }
            break;
        }
        case ActivationType::Softmax: {
            // Softmax derivative combined with Cross Entropy loss gives (output - target) directly,
            // handled cleanly in output error computation. If called directly:
            break;
        }
    }
}

ActivationType Activations::fromString(const std::string& name) {
    if (name == "relu" || name == "ReLU") return ActivationType::ReLU;
    if (name == "leaky_relu" || name == "LeakyReLU") return ActivationType::LeakyReLU;
    if (name == "tanh" || name == "Tanh") return ActivationType::Tanh;
    if (name == "softmax" || name == "Softmax") return ActivationType::Softmax;
    return ActivationType::Sigmoid;
}

std::string Activations::toString(ActivationType type) {
    switch (type) {
        case ActivationType::ReLU: return "relu";
        case ActivationType::LeakyReLU: return "leaky_relu";
        case ActivationType::Tanh: return "tanh";
        case ActivationType::Softmax: return "softmax";
        default: return "sigmoid";
    }
}
