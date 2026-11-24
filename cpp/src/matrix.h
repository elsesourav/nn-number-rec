#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cstdint>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

class Matrix {
private:
    int rows_;
    int cols_;
    std::vector<float> data_;

public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, float fillValue);
    Matrix(int rows, int cols, const std::vector<float>& data);
    Matrix(int rows, int cols, const float* data);

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return rows_ * cols_; }
    
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    uintptr_t dataPtr() const { return reinterpret_cast<uintptr_t>(data_.data()); }

    float get(int r, int c) const {
        return data_[r * cols_ + c];
    }

    void set(int r, int c, float val) {
        data_[r * cols_ + c] = val;
    }

    float& operator()(int r, int c) {
        return data_[r * cols_ + c];
    }

    const float& operator()(int r, int c) const {
        return data_[r * cols_ + c];
    }

    void fill(float value);
    void zero();

    // Weight initializations
    void randomizeUniform(float low = -1.0f, float high = 1.0f);
    void randomizeXavier(int fanIn, int fanOut);
    void randomizeHe(int fanIn);

    // Operations (SIMD accelerated where possible)
    static void add(const Matrix& a, const Matrix& b, Matrix& out);
    static void subtract(const Matrix& a, const Matrix& b, Matrix& out);
    static void hadamard(const Matrix& a, const Matrix& b, Matrix& out); // Element-wise product
    static void dot(const Matrix& a, const Matrix& b, Matrix& out);      // Matrix multiplication
    static void vectorMatrixMultiply(const float* vec, const Matrix& mat, float* out);
    static void transpose(const Matrix& in, Matrix& out);

    // In-place operations
    void add(const Matrix& other);
    void subtract(const Matrix& other);
    void multiplyScalar(float scalar);
    void addScalar(float scalar);

    // Static constructors
    static Matrix fromVector(const std::vector<float>& vec);
    static Matrix fromArray(const float* arr, int rows, int cols);
};

#endif // MATRIX_H
