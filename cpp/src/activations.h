#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cstddef>
#include <string>

enum class ActivationType : int {
    Sigmoid = 0,
    ReLU = 1,
    LeakyReLU = 2,
    Tanh = 3,
    Softmax = 4
};

class Activations {
public:
    static void apply(float* data, size_t size, ActivationType type);
    static void applyDerivative(const float* activatedData, float* gradientInOut, size_t size, ActivationType type);

    static ActivationType fromString(const std::string& name);
    static std::string toString(ActivationType type);
};

#endif // ACTIVATIONS_H
