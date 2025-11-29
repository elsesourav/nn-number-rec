#include "matrix.h"
#include "nn.h"
#include <emscripten/bind.h>
#include <numeric>
#include <vector>

using namespace emscripten;

// Bindings
EMSCRIPTEN_BINDINGS(my_module) {
   // Matrix bindings
   register_vector<double>("vector1d");
   register_vector<int>("vectorInt");
   register_vector<std::vector<double>>("vector2d");

   class_<Matrix>("Matrix")
       .constructor<int, int>()
       .constructor<int, int, const std::vector<std::vector<double>> &>()
       .constructor<emscripten::val>()
       .function("getRows", &Matrix::getRows)
       .function("getCols", &Matrix::getCols)
       .function("getData", &Matrix::getData)
       .function("setData", &Matrix::setData)
       .function("randomWeights", select_overload<void()>(&Matrix::randomWeights))
       .function("randomWeightsRounded", select_overload<void(bool)>(&Matrix::randomWeights))
       .function("add", select_overload<void(const Matrix &)>(&Matrix::add))
       .function("subtract", select_overload<void(const Matrix &)>(&Matrix::subtract))
       .function("multiply", select_overload<void(const Matrix &)>(&Matrix::multiply))
       .function("multiplyScalar", select_overload<void(double)>(&Matrix::multiply))
       .function("transpose", select_overload<void()>(&Matrix::transpose))
       .function("mapSigmoid", optional_override([](Matrix &self) {
                    self.map([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
                 }))
       .function("mapDSigmoid", optional_override([](Matrix &self) {
                    self.map([](double x) { return x * (1.0 - x); });
                 }))
       .function("at", select_overload<const double &(int, int) const>(&Matrix::at))
       .function("show", &Matrix::show)
       .function("clone", &Matrix::clone)
       .class_function("dot", &Matrix::dot)
       .class_function("subtract", &Matrix::subtract)
       .class_function("multiply", select_overload<Matrix(const Matrix &, const Matrix &)>(&Matrix::multiply))
       .class_function("transpose", &Matrix::transpose)
       .class_function("convertFromArray", &Matrix::convertFromArray);


   class_<NeuralNetwork>("NeuralNetwork")
       .constructor<int, std::vector<int>, int, double>()
       .function("feedForward", &NeuralNetwork::feedForward)
       .function("feedForwardArray", &NeuralNetwork::feedForwardArray)
       .function("train", &NeuralNetwork::train)
       .function("trainArray", &NeuralNetwork::trainArray)
       .function("getNumLayers", &NeuralNetwork::getNumLayers)
       .function("getLayer", &NeuralNetwork::getLayer)
       .function("getWeights", &NeuralNetwork::getWeights)
       .function("getBiases", &NeuralNetwork::getBiases)
       .function("setWeights", &NeuralNetwork::setWeights)
       .function("setBiases", &NeuralNetwork::setBiases)
       .function("getNeuronVal", &NeuralNetwork::getNeuronVal)
       .function("getWeightVal", &NeuralNetwork::getWeightVal)
       .function("getLayerSize", &NeuralNetwork::getLayerSize)
       .function("resetActivations", &NeuralNetwork::resetActivations)
       .property("lrnRate", &NeuralNetwork::getLrnRate, &NeuralNetwork::setLrnRate)
       .property("lrStep", &NeuralNetwork::getLrStep, &NeuralNetwork::setLrStep);
}
