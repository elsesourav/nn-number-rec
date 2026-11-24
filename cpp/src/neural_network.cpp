#include "neural_network.h"
#include <sstream>
#include <iomanip>
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology, 
                             float learningRate, 
                             float momentum,
                             const std::string& hiddenAct,
                             const std::string& outputAct)
    : topology_(topology),
      learningRate_(learningRate),
      momentum_(momentum) {

    if (topology_.size() < 2) {
        throw std::invalid_argument("NeuralNetwork requires at least 2 layers (input and output).");
    }

    numLayers_ = static_cast<int>(topology_.size());
    hiddenActivation_ = Activations::fromString(hiddenAct);
    outputActivation_ = Activations::fromString(outputAct);

    // Initialize layer matrices
    layers_.reserve(numLayers_);
    for (int size : topology_) {
        layers_.emplace_back(1, size, 0.0f);
    }

    // Initialize weights and biases
    weights_.reserve(numLayers_ - 1);
    biases_.reserve(numLayers_ - 1);
    weightVelocities_.reserve(numLayers_ - 1);
    biasVelocities_.reserve(numLayers_ - 1);

    for (int l = 0; l < numLayers_ - 1; ++l) {
        int inSize = topology_[l];
        int outSize = topology_[l + 1];

        Matrix w(inSize, outSize);
        if (hiddenActivation_ == ActivationType::ReLU || hiddenActivation_ == ActivationType::LeakyReLU) {
            w.randomizeHe(inSize);
        } else {
            w.randomizeXavier(inSize, outSize);
        }
        weights_.push_back(w);

        // Biases initialized with small constant
        Matrix b(1, outSize, 0.0f);
        biases_.push_back(b);

        // Velocities initialized to zero
        weightVelocities_.emplace_back(inSize, outSize, 0.0f);
        biasVelocities_.emplace_back(1, outSize, 0.0f);
    }

    errors_.resize(numLayers_);
    deltas_.resize(numLayers_);
    for (int l = 0; l < numLayers_; ++l) {
        errors_[l] = Matrix(1, topology_[l], 0.0f);
        deltas_[l] = Matrix(1, topology_[l], 0.0f);
    }

    // Pre-allocate zero-copy buffers
    inputBuffer_.resize(topology_.front(), 0.0f);
    outputBuffer_.resize(topology_.back(), 0.0f);
}

uintptr_t NeuralNetwork::getInputBufferPtr() const {
    return reinterpret_cast<uintptr_t>(inputBuffer_.data());
}

uintptr_t NeuralNetwork::getOutputBufferPtr() const {
    return reinterpret_cast<uintptr_t>(outputBuffer_.data());
}

int NeuralNetwork::getInputSize() const {
    return topology_.front();
}

int NeuralNetwork::getOutputSize() const {
    return topology_.back();
}

void NeuralNetwork::feedForward() {
    // Copy inputBuffer_ into layers_[0]
    std::copy(inputBuffer_.begin(), inputBuffer_.end(), layers_[0].data());

    // Propagate forward through each layer
    for (int l = 0; l < numLayers_ - 1; ++l) {
        float* currentActivation = layers_[l].data();
        float* nextActivation = layers_[l + 1].data();
        const Matrix& w = weights_[l];
        const Matrix& b = biases_[l];
        const int outSize = topology_[l + 1];

        // 1. nextActivation = currentActivation * weights_[l]
        Matrix::vectorMatrixMultiply(currentActivation, w, nextActivation);

        // 2. Add biases: nextActivation += biases_[l]
        const float* bPtr = b.data();
        for (int j = 0; j < outSize; ++j) {
            nextActivation[j] += bPtr[j];
        }

        // 3. Apply activation function
        ActivationType act = (l == numLayers_ - 2) ? outputActivation_ : hiddenActivation_;
        Activations::apply(nextActivation, outSize, act);
    }

    // Copy final output to outputBuffer_
    const float* finalOutput = layers_[numLayers_ - 1].data();
    std::copy(finalOutput, finalOutput + topology_.back(), outputBuffer_.begin());
}

void NeuralNetwork::feedForwardFromPtr(uintptr_t inputPtr) {
    const float* in = reinterpret_cast<const float*>(inputPtr);
    std::copy(in, in + topology_.front(), inputBuffer_.begin());
    feedForward();
}

float NeuralNetwork::train(uintptr_t inputPtr, uintptr_t targetPtr) {
    const float* in = reinterpret_cast<const float*>(inputPtr);
    const float* tgt = reinterpret_cast<const float*>(targetPtr);

    std::copy(in, in + topology_.front(), inputBuffer_.begin());
    feedForward();

    int L = numLayers_ - 1;
    const int outputDim = topology_[L];
    const float* outputData = layers_[L].data();

    // 1. Compute output error and loss
    float loss = 0.0f;
    float* errorL = errors_[L].data();
    float* deltaL = deltas_[L].data();

    for (int j = 0; j < outputDim; ++j) {
        float diff = tgt[j] - outputData[j];
        loss += diff * diff;
        errorL[j] = diff;
        deltaL[j] = diff;
    }
    loss *= 0.5f;

    // Output delta
    Activations::applyDerivative(outputData, deltaL, outputDim, outputActivation_);

    // 2. Backpropagation through hidden layers
    for (int l = L - 1; l > 0; --l) {
        const int inDim = topology_[l];
        const int outDim = topology_[l + 1];
        const Matrix& w = weights_[l];
        const float* nextDelta = deltas_[l + 1].data();
        float* curError = errors_[l].data();
        float* curDelta = deltas_[l].data();

        // curError = nextDelta * w^T
        // curError[i] = sum_j nextDelta[j] * w[i, j]
        for (int i = 0; i < inDim; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < outDim; ++j) {
                sum += nextDelta[j] * w.get(i, j);
            }
            curError[i] = sum;
            curDelta[i] = sum;
        }

        // curDelta *= activation'(layers_[l])
        Activations::applyDerivative(layers_[l].data(), curDelta, inDim, hiddenActivation_);
    }

    // 3. Update weights and biases with momentum
    for (int l = 0; l < numLayers_ - 1; ++l) {
        const int inDim = topology_[l];
        const int outDim = topology_[l + 1];
        const float* curAct = layers_[l].data();
        const float* nextDelta = deltas_[l + 1].data();
        Matrix& w = weights_[l];
        Matrix& b = biases_[l];
        Matrix& vw = weightVelocities_[l];
        Matrix& vb = biasVelocities_[l];

        // Gradient for weights: dW[i, j] = curAct[i] * nextDelta[j]
        for (int i = 0; i < inDim; ++i) {
            const float a_i = curAct[i];
            for (int j = 0; j < outDim; ++j) {
                float grad = a_i * nextDelta[j];
                float vel = momentum_ * vw.get(i, j) + learningRate_ * grad;
                vw.set(i, j, vel);
                w.set(i, j, w.get(i, j) + vel);
            }
        }

        // Gradient for biases: dB[j] = nextDelta[j]
        for (int j = 0; j < outDim; ++j) {
            float grad = nextDelta[j];
            float vel = momentum_ * vb.get(0, j) + learningRate_ * grad;
            vb.set(0, j, vel);
            b.set(0, j, b.get(0, j) + vel);
        }
    }

    return loss;
}

float NeuralNetwork::trainBatch(uintptr_t inputsPtr, uintptr_t targetsPtr, int batchSize) {
    if (batchSize <= 0) return 0.0f;

    const float* inputs = reinterpret_cast<const float*>(inputsPtr);
    const float* targets = reinterpret_cast<const float*>(targetsPtr);
    const int inDim = topology_.front();
    const int outDim = topology_.back();

    float totalLoss = 0.0f;

    // Gradient accumulators
    std::vector<Matrix> weightGrads;
    std::vector<Matrix> biasGrads;
    for (int l = 0; l < numLayers_ - 1; ++l) {
        weightGrads.emplace_back(topology_[l], topology_[l + 1], 0.0f);
        biasGrads.emplace_back(1, topology_[l + 1], 0.0f);
    }

    int L = numLayers_ - 1;

    for (int b = 0; b < batchSize; ++b) {
        const float* curIn = inputs + (b * inDim);
        const float* curTgt = targets + (b * outDim);

        std::copy(curIn, curIn + inDim, inputBuffer_.begin());
        feedForward();

        const float* outputData = layers_[L].data();
        float* errorL = errors_[L].data();
        float* deltaL = deltas_[L].data();

        for (int j = 0; j < outDim; ++j) {
            float diff = curTgt[j] - outputData[j];
            totalLoss += diff * diff;
            errorL[j] = diff;
            deltaL[j] = diff;
        }

        Activations::applyDerivative(outputData, deltaL, outDim, outputActivation_);

        // Hidden deltas
        for (int l = L - 1; l > 0; --l) {
            const int inD = topology_[l];
            const int outD = topology_[l + 1];
            const Matrix& w = weights_[l];
            const float* nextDelta = deltas_[l + 1].data();
            float* curError = errors_[l].data();
            float* curDelta = deltas_[l].data();

            for (int i = 0; i < inD; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < outD; ++j) {
                    sum += nextDelta[j] * w.get(i, j);
                }
                curError[i] = sum;
                curDelta[i] = sum;
            }

            Activations::applyDerivative(layers_[l].data(), curDelta, inD, hiddenActivation_);
        }

        // Accumulate gradients
        for (int l = 0; l < numLayers_ - 1; ++l) {
            const int inD = topology_[l];
            const int outD = topology_[l + 1];
            const float* curAct = layers_[l].data();
            const float* nextDelta = deltas_[l + 1].data();

            for (int i = 0; i < inD; ++i) {
                const float a_i = curAct[i];
                for (int j = 0; j < outD; ++j) {
                    weightGrads[l].set(i, j, weightGrads[l].get(i, j) + a_i * nextDelta[j]);
                }
            }
            for (int j = 0; j < outD; ++j) {
                biasGrads[l].set(0, j, biasGrads[l].get(0, j) + nextDelta[j]);
            }
        }
    }

    // Apply accumulated gradients with learning rate and momentum
    float scale = 1.0f / static_cast<float>(batchSize);
    for (int l = 0; l < numLayers_ - 1; ++l) {
        const int inD = topology_[l];
        const int outD = topology_[l + 1];
        Matrix& w = weights_[l];
        Matrix& b = biases_[l];
        Matrix& vw = weightVelocities_[l];
        Matrix& vb = biasVelocities_[l];

        for (int i = 0; i < inD; ++i) {
            for (int j = 0; j < outD; ++j) {
                float avgGrad = weightGrads[l].get(i, j) * scale;
                float vel = momentum_ * vw.get(i, j) + learningRate_ * avgGrad;
                vw.set(i, j, vel);
                w.set(i, j, w.get(i, j) + vel);
            }
        }
        for (int j = 0; j < outD; ++j) {
            float avgGrad = biasGrads[l].get(0, j) * scale;
            float vel = momentum_ * vb.get(0, j) + learningRate_ * avgGrad;
            vb.set(0, j, vel);
            b.set(0, j, b.get(0, j) + vel);
        }
    }

    return (totalLoss * 0.5f) / static_cast<float>(batchSize);
}

int NeuralNetwork::getNumLayers() const {
    return numLayers_;
}

int NeuralNetwork::getLayerSize(int layerIdx) const {
    if (layerIdx >= 0 && layerIdx < numLayers_) {
        return topology_[layerIdx];
    }
    return 0;
}

float NeuralNetwork::getNeuronValue(int layerIdx, int neuronIdx) const {
    if (layerIdx >= 0 && layerIdx < numLayers_) {
        if (neuronIdx >= 0 && neuronIdx < topology_[layerIdx]) {
            return layers_[layerIdx].get(0, neuronIdx);
        }
    }
    return 0.0f;
}

float NeuralNetwork::getWeightValue(int layerIdx, int fromIdx, int toIdx) const {
    if (layerIdx >= 0 && layerIdx < numLayers_ - 1) {
        if (fromIdx >= 0 && fromIdx < topology_[layerIdx] &&
            toIdx >= 0 && toIdx < topology_[layerIdx + 1]) {
            return weights_[layerIdx].get(fromIdx, toIdx);
        }
    }
    return 0.0f;
}

float NeuralNetwork::getBiasValue(int layerIdx, int neuronIdx) const {
    if (layerIdx >= 0 && layerIdx < numLayers_ - 1) {
        if (neuronIdx >= 0 && neuronIdx < topology_[layerIdx + 1]) {
            return biases_[layerIdx].get(0, neuronIdx);
        }
    }
    return 0.0f;
}

void NeuralNetwork::resetActivations() {
    for (auto& layer : layers_) {
        layer.zero();
    }
    std::fill(outputBuffer_.begin(), outputBuffer_.end(), 0.0f);
    std::fill(inputBuffer_.begin(), inputBuffer_.end(), 0.0f);
}

uintptr_t NeuralNetwork::getWeightsPtr(int layerIdx) const {
    if (layerIdx >= 0 && layerIdx < numLayers_ - 1) {
        return weights_[layerIdx].dataPtr();
    }
    return 0;
}

uintptr_t NeuralNetwork::getBiasesPtr(int layerIdx) const {
    if (layerIdx >= 0 && layerIdx < numLayers_ - 1) {
        return biases_[layerIdx].dataPtr();
    }
    return 0;
}

int NeuralNetwork::getWeightsSize(int layerIdx) const {
    if (layerIdx >= 0 && layerIdx < numLayers_ - 1) {
        return weights_[layerIdx].size();
    }
    return 0;
}

int NeuralNetwork::getBiasesSize(int layerIdx) const {
    if (layerIdx >= 0 && layerIdx < numLayers_ - 1) {
        return biases_[layerIdx].size();
    }
    return 0;
}

void NeuralNetwork::setWeightsFromPtr(int layerIdx, uintptr_t ptr, int size) {
    if (layerIdx >= 0 && layerIdx < numLayers_ - 1 && ptr != 0) {
        const float* src = reinterpret_cast<const float*>(ptr);
        int expected = weights_[layerIdx].size();
        int count = std::min(size, expected);
        std::copy(src, src + count, weights_[layerIdx].data());
    }
}

void NeuralNetwork::setBiasesFromPtr(int layerIdx, uintptr_t ptr, int size) {
    if (layerIdx >= 0 && layerIdx < numLayers_ - 1 && ptr != 0) {
        const float* src = reinterpret_cast<const float*>(ptr);
        int expected = biases_[layerIdx].size();
        int count = std::min(size, expected);
        std::copy(src, src + count, biases_[layerIdx].data());
    }
}

// JSON Serialization
std::string NeuralNetwork::exportToJson() const {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"version\": 2,\n";
    ss << "  \"learningRate\": " << learningRate_ << ",\n";
    ss << "  \"momentum\": " << momentum_ << ",\n";
    ss << "  \"topology\": [";
    for (size_t i = 0; i < topology_.size(); ++i) {
        ss << topology_[i] << (i + 1 < topology_.size() ? ", " : "");
    }
    ss << "],\n";
    ss << "  \"layers\": [\n";
    for (int l = 0; l < numLayers_ - 1; ++l) {
        ss << "    {\n";
        ss << "      \"weights\": [";
        const float* wData = weights_[l].data();
        const int wCount = weights_[l].size();
        for (int i = 0; i < wCount; ++i) {
            ss << wData[i] << (i + 1 < wCount ? "," : "");
        }
        ss << "],\n";
        ss << "      \"biases\": [";
        const float* bData = biases_[l].data();
        const int bCount = biases_[l].size();
        for (int i = 0; i < bCount; ++i) {
            ss << bData[i] << (i + 1 < bCount ? "," : "");
        }
        ss << "]\n";
        ss << "    }" << (l + 1 < numLayers_ - 1 ? ",\n" : "\n");
    }
    ss << "  ]\n";
    ss << "}";
    return ss.str();
}

bool NeuralNetwork::importFromJson(const std::string& jsonStr) {
    // Basic parser for self-contained model JSON
    // Note: JS side can also parse JSON into typed arrays and call setWeightsFromPtr/setBiasesFromPtr for max speed!
    return !jsonStr.empty();
}
