#ifndef MATRIX_H
#define MATRIX_H

#include <cmath>
#include <emscripten/val.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

class Matrix {
private:
   int rows;
   int cols;
   std::vector<std::vector<double>> data;

public:
   Matrix(int rows, int cols);
   Matrix(int rows, int cols, const std::vector<std::vector<double>> &data);
   Matrix(emscripten::val data); // Constructor from JS array

   int getRows() const;
   int getCols() const;
   std::vector<std::vector<double>> getData() const;
   void setData(const std::vector<std::vector<double>> &newData);

   Matrix clone() const;

   // Access element directly (helper for C++ usage)
   double &at(int i, int j);
   const double &at(int i, int j) const;

   void randomWeights();
   void randomWeights(bool round);

   static Matrix add(const Matrix &m1, const Matrix &m2);
   void add(const Matrix &m2);
   void add(double scalar); // Helper often useful

   static Matrix subtract(const Matrix &m1, const Matrix &m2);
   void subtract(const Matrix &m2);

   static Matrix multiply(const Matrix &m1, const Matrix &m2); // Element-wise
   void multiply(const Matrix &m2);                            // Element-wise
   void multiply(double scalar);                               // Helper often useful

   static Matrix dot(const Matrix &m1, const Matrix &m2); // Matrix product

   static Matrix convertFromArray(const std::vector<double> &arr);

   static Matrix map(const Matrix &m1, std::function<double(double)> func);
   void map(std::function<double(double)> func);

   static Matrix transpose(const Matrix &m1);
   void transpose();

   void show() const;

   static void checkDimensions(const Matrix &m1, const Matrix &m2);
};

#endif
