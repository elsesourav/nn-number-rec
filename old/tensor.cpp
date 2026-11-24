#include "tensor.h"
#include <iostream>
#include <numeric>
#include <stdexcept>

// Helper to calculate size from shape
int sizeFromShape(const std::vector<int> &shape) {
   if (shape.empty())
      return 0;
   int size = 1;
   for (int dim : shape)
      size *= dim;
   return size;
}

// Helper to parse JS array recursively
void parseRecursive(emscripten::val v, std::vector<double> &data, std::vector<int> &shape, int dim) {
   unsigned int len = v["length"].as<unsigned int>();

   if (shape.size() <= dim) {
      shape.push_back(len);
   } else {
      if (shape[dim] != len)
         throw std::invalid_argument("Ragged arrays are not supported");
   }

   if (len == 0)
      return;

   emscripten::val first = v[0];
   // Check if the element is an array (has length and is object)
   bool isArray = (first.typeOf().as<std::string>() == "object") && !first["length"].isUndefined();

   if (isArray) {
      for (unsigned int i = 0; i < len; ++i) {
         parseRecursive(v[i], data, shape, dim + 1);
      }
   } else {
      // Base case: numbers
      for (unsigned int i = 0; i < len; ++i) {
         data.push_back(v[i].as<double>());
      }
   }
}

// Helper to convert JS array to std::vector<int>
std::vector<int> valToVecInt(emscripten::val v) {
   std::vector<int> vec;
   if (v.typeOf().as<std::string>() != "object" || v["length"].isUndefined()) {
      // Try to treat as single integer if not array? No, indices must be array/vector.
      // Or maybe it's a vectorInt object?
      // For now assume it's a JS array.
      throw std::invalid_argument("Indices must be an array");
   }
   unsigned int length = v["length"].as<unsigned int>();
   vec.reserve(length);
   for (unsigned int i = 0; i < length; ++i) {
      vec.push_back(v[i].as<int>());
   }
   return vec;
}

Tensor::Tensor(emscripten::val jsArray) {
   if (jsArray.typeOf().as<std::string>() != "object" || jsArray["length"].isUndefined()) {
      throw std::invalid_argument("Input must be an array");
   }
   parseRecursive(jsArray, data, shape, 0);
}

Tensor::Tensor(const std::vector<int> &shape) : shape(shape) {
   int size = sizeFromShape(shape);
   data.resize(size, 0.0);
}

std::vector<int> Tensor::getShape() const {
   return shape;
}

std::vector<double> Tensor::getData() const {
   return data;
}

double Tensor::get(const std::vector<int> &indices) const {
   int flatIndex = 0;
   int multiplier = 1;
   for (int i = shape.size() - 1; i >= 0; --i) {
      if (i >= indices.size())
         throw std::out_of_range("Index dimension mismatch");
      flatIndex += indices[i] * multiplier;
      multiplier *= shape[i];
   }
   if (flatIndex < 0 || flatIndex >= data.size())
      throw std::out_of_range("Index out of bounds");
   return data[flatIndex];
}

double Tensor::get(emscripten::val indices) const {
   return get(valToVecInt(indices));
}

void Tensor::set(const std::vector<int> &indices, double value) {
   int flatIndex = 0;
   int multiplier = 1;
   for (int i = shape.size() - 1; i >= 0; --i) {
      if (i >= indices.size())
         throw std::out_of_range("Index dimension mismatch");
      flatIndex += indices[i] * multiplier;
      multiplier *= shape[i];
   }
   if (flatIndex < 0 || flatIndex >= data.size())
      throw std::out_of_range("Index out of bounds");
   data[flatIndex] = value;
}

void Tensor::set(emscripten::val indices, double value) {
   set(valToVecInt(indices), value);
}

// Helper to flatten a JS item (sub-tensor) into a vector
void flattenItem(emscripten::val item, std::vector<double> &flatData, const std::vector<int> &expectedShape, int dim) {
   // If we are at the last dimension of expected shape (which is shape[1:] of tensor), item should be number
   // Wait, expectedShape is the shape of the ITEM being pushed.

   if (dim == expectedShape.size()) {
      // Should be a number
      if (item.typeOf().as<std::string>() == "number") {
         flatData.push_back(item.as<double>());
      } else {
         throw std::invalid_argument("Structure mismatch");
      }
      return;
   }

   unsigned int len = item["length"].as<unsigned int>();
   if (len != expectedShape[dim]) {
      throw std::invalid_argument("Dimension mismatch in pushed item");
   }

   for (unsigned int i = 0; i < len; ++i) {
      flattenItem(item[i], flatData, expectedShape, dim + 1);
   }
}

void Tensor::push(emscripten::val item) {
   // If tensor is empty (shape empty), initialize it
   if (shape.empty()) {
      // Parse as new tensor
      std::vector<double> newData;
      std::vector<int> newShape;
      // Wrap item in array to make it [item]
      emscripten::val wrapper = emscripten::val::array();
      wrapper.call<void>("push", item);
      parseRecursive(wrapper, data, shape, 0);
      return;
   }

   // Expected shape of item is shape[1:]
   std::vector<int> itemShape(shape.begin() + 1, shape.end());

   // Flatten item
   std::vector<double> itemData;
   // We need a robust way to flatten 'item' checking against 'itemShape'
   // Re-using parseRecursive logic is tricky because we need to enforce shape.
   // Let's assume the user passes correct structure for now or use a simplified parser.

   // Actually, let's just use a temporary Tensor to parse the item, then check shape
   // But item might be a number if Tensor is 1D.
   if (shape.size() == 1) {
      if (item.typeOf().as<std::string>() != "number")
         throw std::invalid_argument("Expected number for 1D tensor push");
      data.push_back(item.as<double>());
      shape[0]++;
      return;
   }

   // For N-D, item must be an array
   // We can cheat: create a Tensor from item, check if its shape matches itemShape
   Tensor temp(item);
   if (temp.shape != itemShape) {
      throw std::invalid_argument("Item shape does not match tensor dimensions");
   }

   data.insert(data.end(), temp.data.begin(), temp.data.end());
   shape[0]++;
}

void Tensor::insert(int index, emscripten::val item) {
   if (shape.empty()) {
      push(item); // Index doesn't matter if empty, it becomes index 0
      return;
   }

   if (index < 0 || index > shape[0])
      throw std::out_of_range("Index out of bounds");

   if (shape.size() == 1) {
      if (item.typeOf().as<std::string>() != "number")
         throw std::invalid_argument("Expected number");
      data.insert(data.begin() + index, item.as<double>());
      shape[0]++;
      return;
   }

   Tensor temp(item);
   std::vector<int> itemShape(shape.begin() + 1, shape.end());
   if (temp.shape != itemShape) {
      throw std::invalid_argument("Item shape does not match tensor dimensions");
   }

   // Calculate offset in data
   int stride = sizeFromShape(itemShape);
   int dataIndex = index * stride;

   data.insert(data.begin() + dataIndex, temp.data.begin(), temp.data.end());
   shape[0]++;
}

void Tensor::pop() {
   if (shape.empty() || shape[0] == 0)
      return;

   int stride = 1;
   if (shape.size() > 1) {
      std::vector<int> itemShape(shape.begin() + 1, shape.end());
      stride = sizeFromShape(itemShape);
   }

   // Remove last 'stride' elements
   data.resize(data.size() - stride);
   shape[0]--;

   if (shape[0] == 0 && shape.size() == 1) {
      // Keep shape as [0] for 1D? Or clear?
      // Usually 1D array of size 0 is [0]
   }
}

Tensor Tensor::slice(int start, int end) const {
   if (shape.empty())
      return *this;

   int dim0 = shape[0];
   if (start < 0)
      start += dim0;
   if (end < 0)
      end += dim0;

   if (start < 0)
      start = 0;
   if (end > dim0)
      end = dim0;
   if (start > end)
      start = end;

   int newDim0 = end - start;
   std::vector<int> newShape = shape;
   newShape[0] = newDim0;

   if (newDim0 == 0) {
      Tensor t(newShape); // Empty tensor with correct dimensionality
      return t;
   }

   int stride = 1;
   if (shape.size() > 1) {
      std::vector<int> itemShape(shape.begin() + 1, shape.end());
      stride = sizeFromShape(itemShape);
   }

   int startData = start * stride;
   int endData = end * stride;

   Tensor t(newShape);
   t.data.assign(data.begin() + startData, data.begin() + endData);
   return t;
}

Tensor Tensor::slice(int start) const {
   if (shape.empty())
      return *this;
   return slice(start, shape[0]);
}

void Tensor::show() const {
   std::cout << "Tensor shape: [";
   for (size_t i = 0; i < shape.size(); ++i) {
      std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
   }
   std::cout << "], size: " << data.size() << std::endl;
}
