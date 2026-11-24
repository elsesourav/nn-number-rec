#include "matrix.h"
#include "neural_network.h"
#include "activations.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(nn_engine_module) {
    register_vector<int>("VectorInt");
    register_vector<float>("VectorFloat");

    class_<NeuralNetwork>("NeuralNetwork")
        .constructor<std::vector<int>, float, float, std::string, std::string>()
        
        // Zero-copy buffer pointer accessors
        .function("getInputBufferPtr", &NeuralNetwork::getInputBufferPtr)
        .function("getOutputBufferPtr", &NeuralNetwork::getOutputBufferPtr)
        .function("getInputSize", &NeuralNetwork::getInputSize)
        .function("getOutputSize", &NeuralNetwork::getOutputSize)

        // Direct typed memory views (Float32Array)
        .function("getInputView", optional_override([](NeuralNetwork& self) {
            return val(typed_memory_view(self.getInputSize(), reinterpret_cast<const float*>(self.getInputBufferPtr())));
        }))
        .function("getOutputView", optional_override([](NeuralNetwork& self) {
            return val(typed_memory_view(self.getOutputSize(), reinterpret_cast<const float*>(self.getOutputBufferPtr())));
        }))

        // Core inference and training
        .function("feedForward", &NeuralNetwork::feedForward)
        .function("feedForwardFromPtr", &NeuralNetwork::feedForwardFromPtr)
        .function("train", &NeuralNetwork::train)
        .function("trainBatch", &NeuralNetwork::trainBatch)

        // High-level train function accepting Float32Array from JavaScript directly
        .function("trainWithArrays", optional_override([](NeuralNetwork& self, const val& inArr, const val& tgtArr) {
            const int inDim = self.getInputSize();
            const int outDim = self.getOutputSize();

            std::vector<float> inVec = convertJSArrayToNumberVector<float>(inArr);
            std::vector<float> tgtVec = convertJSArrayToNumberVector<float>(tgtArr);

            return self.train(reinterpret_cast<uintptr_t>(inVec.data()), reinterpret_cast<uintptr_t>(tgtVec.data()));
        }))

        // Introspection for visualization
        .function("getNumLayers", &NeuralNetwork::getNumLayers)
        .function("getLayerSize", &NeuralNetwork::getLayerSize)
        .function("getNeuronValue", &NeuralNetwork::getNeuronValue)
        .function("getWeightValue", &NeuralNetwork::getWeightValue)
        .function("getBiasValue", &NeuralNetwork::getBiasValue)
        .function("resetActivations", &NeuralNetwork::resetActivations)

        // Hyperparameters
        .property("learningRate", &NeuralNetwork::getLearningRate, &NeuralNetwork::setLearningRate)
        .property("momentum", &NeuralNetwork::getMomentum, &NeuralNetwork::setMomentum)

        // Serialization
        .function("getWeightsPtr", &NeuralNetwork::getWeightsPtr)
        .function("getBiasesPtr", &NeuralNetwork::getBiasesPtr)
        .function("getWeightsSize", &NeuralNetwork::getWeightsSize)
        .function("getBiasesSize", &NeuralNetwork::getBiasesSize)
        .function("setWeightsFromPtr", &NeuralNetwork::setWeightsFromPtr)
        .function("setBiasesFromPtr", &NeuralNetwork::setBiasesFromPtr)
        .function("exportToJson", &NeuralNetwork::exportToJson)
        .function("getTopology", &NeuralNetwork::getTopology);
}
