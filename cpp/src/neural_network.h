#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "matrix.h"
#include "activations.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

class NeuralNetwork {
private:
    std::vector<int> topology_;       // [inputSize, hidden1, hidden2, ..., outputSize]
    int numLayers_;
    float learningRate_;
    float momentum_;
    ActivationType hiddenActivation_;
    ActivationType outputActivation_;

    // Layer activations: layers_[l] has dimensions (1, topology_[l])
    std::vector<Matrix> layers_;

    // Weights: weights_[l] has dimensions (topology_[l], topology_[l + 1])
    std::vector<Matrix> weights_;

    // Biases: biases_[l] has dimensions (1, topology_[l + 1])
    std::vector<Matrix> biases_;

    // Velocity matrices for momentum
    std::vector<Matrix> weightVelocities_;
    std::vector<Matrix> biasVelocities_;

    // Gradients / deltas buffers for backprop
    std::vector<Matrix> errors_;
    std::vector<Matrix> deltas_;

    // Zero-copy I/O buffers for JS bridge
    std::vector<float> inputBuffer_;
    std::vector<float> outputBuffer_;

public:
    NeuralNetwork(const std::vector<int>& topology, 
                  float learningRate = 0.1f, 
                  float momentum = 0.85f,
                  const std::string& hiddenAct = "sigmoid",
                  const std::string& outputAct = "sigmoid");

    // Zero-copy buffer pointer accessors for JavaScript
    uintptr_t getInputBufferPtr() const;
    uintptr_t getOutputBufferPtr() const;
    int getInputSize() const;
    int getOutputSize() const;

    // Inference (feed-forward using inputBuffer_, writing outputBuffer_)
    void feedForward();
    void feedForwardFromPtr(uintptr_t inputPtr);

    // Single-sample training
    float train(uintptr_t inputPtr, uintptr_t targetPtr);
    float trainDirect(); // uses inputBuffer_ and expects target in outputBuffer_ or separate target buffer

    // Batch training: inputs (batchSize * inputSize), targets (batchSize * outputSize)
    float trainBatch(uintptr_t inputsPtr, uintptr_t targetsPtr, int batchSize);

    // Introspection for interactive visualization
    int getNumLayers() const;
    int getLayerSize(int layerIdx) const;
    float getNeuronValue(int layerIdx, int neuronIdx) const;
    float getWeightValue(int layerIdx, int fromIdx, int toIdx) const;
    float getBiasValue(int layerIdx, int neuronIdx) const;

    void resetActivations();

    // Hyperparameters
    float getLearningRate() const { return learningRate_; }
    void setLearningRate(float lr) { learningRate_ = lr; }

    float getMomentum() const { return momentum_; }
    void setMomentum(float m) { momentum_ = m; }

    // Serialization
    uintptr_t getWeightsPtr(int layerIdx) const;
    uintptr_t getBiasesPtr(int layerIdx) const;
    int getWeightsSize(int layerIdx) const;
    int getBiasesSize(int layerIdx) const;

    void setWeightsFromPtr(int layerIdx, uintptr_t ptr, int size);
    void setBiasesFromPtr(int layerIdx, uintptr_t ptr, int size);

    std::string exportToJson() const;
    bool importFromJson(const std::string& jsonStr);

    std::vector<int> getTopology() const { return topology_; }
};

#endif // NEURAL_NETWORK_H
