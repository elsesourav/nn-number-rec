#ifndef TENSOR_H
#define TENSOR_H

#include <emscripten/val.h>
#include <string>
#include <vector>

class Tensor {
public:
   std::vector<double> data;
   std::vector<int> shape;

   Tensor(emscripten::val jsArray);
   Tensor(const std::vector<int> &shape);

   std::vector<int> getShape() const;
   std::vector<double> getData() const;

   // Access/Modify
   double get(const std::vector<int> &indices) const;
   double get(emscripten::val indices) const; // Overload for JS array
   void set(const std::vector<int> &indices, double value);
   void set(emscripten::val indices, double value); // Overload for JS array

   // Array-like operations on axis 0
   void push(emscripten::val item);
   void insert(int index, emscripten::val item);
   void pop();
   Tensor slice(int start, int end) const;
   Tensor slice(int start) const;

   void show() const;
};

#endif
