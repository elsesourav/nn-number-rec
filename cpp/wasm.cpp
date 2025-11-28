#include "matrix.h"
#include "tensor.h"
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
       .class_function("dot", &Matrix::dot)
       .class_function("subtract", &Matrix::subtract)
       .class_function("transpose", &Matrix::transpose)
       .class_function("convertFromArray", &Matrix::convertFromArray);

   class_<Tensor>("Tensor")
       .constructor<emscripten::val>()
       .function("getShape", &Tensor::getShape)
       .function("getData", &Tensor::getData)
       .function("get", select_overload<double(emscripten::val) const>(&Tensor::get))
       .function("set", select_overload<void(emscripten::val, double)>(&Tensor::set))
       .function("push", &Tensor::push)
       .function("insert", &Tensor::insert)
       .function("pop", &Tensor::pop)
       .function("slice", select_overload<Tensor(int, int) const>(&Tensor::slice))
       .function("slice", select_overload<Tensor(int) const>(&Tensor::slice))
       .function("show", &Tensor::show);
}
