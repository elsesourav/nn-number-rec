#include "matrix.h"

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define HAS_WASM_SIMD 1
#else
#define HAS_WASM_SIMD 0
#endif

Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

Matrix::Matrix(int rows, int cols, float fillValue) : rows_(rows), cols_(cols), data_(rows * cols, fillValue) {}

Matrix::Matrix(int rows, int cols, const std::vector<float>& data) : rows_(rows), cols_(cols), data_(data) {
    if (static_cast<int>(data_.size()) != rows * cols) {
        throw std::invalid_argument("Data size does not match matrix dimensions.");
    }
}

Matrix::Matrix(int rows, int cols, const float* data) : rows_(rows), cols_(cols), data_(data, data + rows * cols) {}

void Matrix::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Matrix::zero() {
    fill(0.0f);
}

void Matrix::randomizeUniform(float low, float high) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(low, high);
    for (float& v : data_) {
        v = dis(gen);
    }
}

// Xavier / Glorot initialization: Normal(0, std = sqrt(2 / (fanIn + fanOut)))
void Matrix::randomizeXavier(int fanIn, int fanOut) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    float stdDev = std::sqrt(2.0f / static_cast<float>(fanIn + fanOut));
    std::normal_distribution<float> dis(0.0f, stdDev);
    for (float& v : data_) {
        v = dis(gen);
    }
}

// He initialization: Normal(0, std = sqrt(2 / fanIn))
void Matrix::randomizeHe(int fanIn) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    float stdDev = std::sqrt(2.0f / static_cast<float>(fanIn));
    std::normal_distribution<float> dis(0.0f, stdDev);
    for (float& v : data_) {
        v = dis(gen);
    }
}

// Element-wise addition
void Matrix::add(const Matrix& a, const Matrix& b, Matrix& out) {
    const int total = a.size();
    const float* aPtr = a.data();
    const float* bPtr = b.data();
    float* outPtr = out.data();

    int i = 0;
#if HAS_WASM_SIMD
    for (; i + 3 < total; i += 4) {
        v128_t va = wasm_v128_load(&aPtr[i]);
        v128_t vb = wasm_v128_load(&bPtr[i]);
        wasm_v128_store(&outPtr[i], wasm_f32x4_add(va, vb));
    }
#endif
    for (; i < total; ++i) {
        outPtr[i] = aPtr[i] + bPtr[i];
    }
}

// Element-wise subtraction
void Matrix::subtract(const Matrix& a, const Matrix& b, Matrix& out) {
    const int total = a.size();
    const float* aPtr = a.data();
    const float* bPtr = b.data();
    float* outPtr = out.data();

    int i = 0;
#if HAS_WASM_SIMD
    for (; i + 3 < total; i += 4) {
        v128_t va = wasm_v128_load(&aPtr[i]);
        v128_t vb = wasm_v128_load(&bPtr[i]);
        wasm_v128_store(&outPtr[i], wasm_f32x4_sub(va, vb));
    }
#endif
    for (; i < total; ++i) {
        outPtr[i] = aPtr[i] - bPtr[i];
    }
}

// Element-wise multiplication (Hadamard product)
void Matrix::hadamard(const Matrix& a, const Matrix& b, Matrix& out) {
    const int total = a.size();
    const float* aPtr = a.data();
    const float* bPtr = b.data();
    float* outPtr = out.data();

    int i = 0;
#if HAS_WASM_SIMD
    for (; i + 3 < total; i += 4) {
        v128_t va = wasm_v128_load(&aPtr[i]);
        v128_t vb = wasm_v128_load(&bPtr[i]);
        wasm_v128_store(&outPtr[i], wasm_f32x4_mul(va, vb));
    }
#endif
    for (; i < total; ++i) {
        outPtr[i] = aPtr[i] * bPtr[i];
    }
}

// Highly optimized Vector-Matrix multiplication:
// vec (1 x K) * mat (K x N) -> out (1 x N)
// This is the core operation of feedforward neural networks!
// Because mat is stored row-major, row k is contiguous: mat(k, 0 .. N-1).
// We accumulate out += vec[k] * mat[k, :] using contiguous SIMD streaming!
void Matrix::vectorMatrixMultiply(const float* vec, const Matrix& mat, float* out) {
    const int K = mat.rows_;
    const int N = mat.cols_;
    const float* matPtr = mat.data();

    // Initialize output to 0
    std::fill(out, out + N, 0.0f);

    for (int k = 0; k < K; ++k) {
        const float a_k = vec[k];
        if (std::abs(a_k) < 1e-8f) continue; // Skip near-zero inputs (sparse activations)

        const float* rowPtr = matPtr + (k * N);
        int j = 0;
#if HAS_WASM_SIMD
        v128_t va = wasm_f32x4_splat(a_k);
        for (; j + 3 < N; j += 4) {
            v128_t vout = wasm_v128_load(&out[j]);
            v128_t vmat = wasm_v128_load(&rowPtr[j]);
            v128_t vprod = wasm_f32x4_mul(va, vmat);
            wasm_v128_store(&out[j], wasm_f32x4_add(vout, vprod));
        }
#endif
        for (; j < N; ++j) {
            out[j] += a_k * rowPtr[j];
        }
    }
}

// General Matrix Multiplication: out = a * b
// a: (M x K), b: (K x N), out: (M x N)
void Matrix::dot(const Matrix& a, const Matrix& b, Matrix& out) {
    const int M = a.rows_;
    const int K = a.cols_;
    const int N = b.cols_;

    if (a.cols_ != b.rows_) {
        throw std::invalid_argument("Incompatible matrix dimensions for dot product.");
    }

    out.zero();
    const float* aData = a.data();
    const float* bData = b.data();
    float* outData = out.data();

    for (int i = 0; i < M; ++i) {
        const float* aRow = aData + (i * K);
        float* outRow = outData + (i * N);

        for (int k = 0; k < K; ++k) {
            const float a_ik = aRow[k];
            if (std::abs(a_ik) < 1e-8f) continue;

            const float* bRow = bData + (k * N);
            int j = 0;
#if HAS_WASM_SIMD
            v128_t va = wasm_f32x4_splat(a_ik);
            for (; j + 3 < N; j += 4) {
                v128_t vout = wasm_v128_load(&outRow[j]);
                v128_t vb = wasm_v128_load(&bRow[j]);
                v128_t vprod = wasm_f32x4_mul(va, vb);
                wasm_v128_store(&outRow[j], wasm_f32x4_add(vout, vprod));
            }
#endif
            for (; j < N; ++j) {
                outRow[j] += a_ik * bRow[j];
            }
        }
    }
}

// Transpose: in (rows x cols) -> out (cols x rows)
void Matrix::transpose(const Matrix& in, Matrix& out) {
    const int R = in.rows_;
    const int C = in.cols_;
    const float* inData = in.data();
    float* outData = out.data();

    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            outData[c * R + r] = inData[r * C + c];
        }
    }
}

void Matrix::add(const Matrix& other) {
    const int total = size();
    float* thisPtr = data();
    const float* otherPtr = other.data();

    int i = 0;
#if HAS_WASM_SIMD
    for (; i + 3 < total; i += 4) {
        v128_t va = wasm_v128_load(&thisPtr[i]);
        v128_t vb = wasm_v128_load(&otherPtr[i]);
        wasm_v128_store(&thisPtr[i], wasm_f32x4_add(va, vb));
    }
#endif
    for (; i < total; ++i) {
        thisPtr[i] += otherPtr[i];
    }
}

void Matrix::subtract(const Matrix& other) {
    const int total = size();
    float* thisPtr = data();
    const float* otherPtr = other.data();

    int i = 0;
#if HAS_WASM_SIMD
    for (; i + 3 < total; i += 4) {
        v128_t va = wasm_v128_load(&thisPtr[i]);
        v128_t vb = wasm_v128_load(&otherPtr[i]);
        wasm_v128_store(&thisPtr[i], wasm_f32x4_sub(va, vb));
    }
#endif
    for (; i < total; ++i) {
        thisPtr[i] -= otherPtr[i];
    }
}

void Matrix::multiplyScalar(float scalar) {
    const int total = size();
    float* thisPtr = data();

    int i = 0;
#if HAS_WASM_SIMD
    v128_t vs = wasm_f32x4_splat(scalar);
    for (; i + 3 < total; i += 4) {
        v128_t va = wasm_v128_load(&thisPtr[i]);
        wasm_v128_store(&thisPtr[i], wasm_f32x4_mul(va, vs));
    }
#endif
    for (; i < total; ++i) {
        thisPtr[i] *= scalar;
    }
}

void Matrix::addScalar(float scalar) {
    const int total = size();
    float* thisPtr = data();

    int i = 0;
#if HAS_WASM_SIMD
    v128_t vs = wasm_f32x4_splat(scalar);
    for (; i + 3 < total; i += 4) {
        v128_t va = wasm_v128_load(&thisPtr[i]);
        wasm_v128_store(&thisPtr[i], wasm_f32x4_add(va, vs));
    }
#endif
    for (; i < total; ++i) {
        thisPtr[i] += scalar;
    }
}

Matrix Matrix::fromVector(const std::vector<float>& vec) {
    return Matrix(1, static_cast<int>(vec.size()), vec);
}

Matrix Matrix::fromArray(const float* arr, int rows, int cols) {
    return Matrix(rows, cols, arr);
}
