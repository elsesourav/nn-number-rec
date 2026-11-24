# C++ SIMD Matrix Engine Documentation

The `Matrix` class is a high-performance 2D linear algebra engine written in modern C++20 and compiled to WebAssembly with 128-bit SIMD acceleration (`<wasm_simd128.h>`). It powers the neural network inference and backpropagation hot paths in this project.

Unlike legacy implementations that relied on fragmented nested vectors (`std::vector<std::vector<double>>`), this implementation uses a **flat 1D contiguous buffer** (`std::vector<float> data_`) in row-major order. This guarantees sequential memory alignment, CPU L1/L2 cache efficiency, vectorization registers, and zero-copy pointer access across the WebAssembly/JavaScript boundary.

---

## 🚀 Key Architectural Innovations

1. **Contiguous Row-Major Memory Layout**:
   - For an $R \times C$ matrix, the element at row $r$ and column $c$ is indexed linearly as:
     $$\text{index}(r, c) = r \cdot C + c$$
   - Eliminates pointer chasing, dynamic memory fragmentation, and cache line invalidation.
2. **128-bit SIMD Vectorization (`wasm_simd128`)**:
   - Math operations (matrix multiplication, Hadamard product, vector-matrix product, and element-wise arithmetic) process four 32-bit floating-point numbers simultaneously per clock cycle using `wasm_f32x4` intrinsics.
3. **Deep Learning Initialization Schemes**:
   - Built-in **He (Kaiming)** initialization for ReLU and Leaky ReLU activations.
   - Built-in **Xavier (Glorot)** initialization for Sigmoid and Tanh activations.
4. **Zero-Copy WebAssembly Bridge**:
   - Exposes raw heap memory pointers via `uintptr_t dataPtr()`.
   - JavaScript code can instantly construct a `Float32Array` view over `Module.HEAPF32.buffer` without copying or serializing memory.

---

## 🛠️ Header Definition (`cpp/src/matrix.h`)

```cpp
#include "matrix.h"

// Instantiate a 64 x 784 synaptic weight matrix
Matrix weights(64, 784);

// Initialize weights using He (Kaiming) standard normal distribution
weights.randomizeHe(784);
```

---

## 📖 Constructors & Factory Methods

### `Matrix()`
Default constructor. Creates an empty $0 \times 0$ matrix.

### `Matrix(int rows, int cols)`
Creates a matrix of dimensions $rows \times cols$, pre-allocated and initialized to zero.

- **Parameters:**
  - `rows` (`int`): Number of rows ($R$).
  - `cols` (`int`): Number of columns ($C$).
- **Example:**
  ```cpp
  Matrix layerBiases(1, 64); // 1x64 row vector initialized to 0.0f
  ```

### `Matrix(int rows, int cols, float fillValue)`
Creates a matrix filled uniformly with `fillValue`.

- **Parameters:**
  - `rows` (`int`): Number of rows.
  - `cols` (`int`): Number of columns.
  - `fillValue` (`float`): Initial scalar value.

### `Matrix(int rows, int cols, const std::vector<float>& data)`
Constructs a matrix from an existing contiguous `std::vector<float>`.

### `Matrix(int rows, int cols, const float* data)`
Constructs a matrix by copying $rows \times cols$ elements from a raw float buffer pointer.

### `Matrix::fromVector(const std::vector<float>& vec)`
Static factory method. Converts a 1D vector into a $1 \times N$ row-vector matrix.

### `Matrix::fromArray(const float* arr, int rows, int cols)`
Static factory method. Wraps raw memory into a managed `Matrix` instance.

---

## 🔍 Dimensions & Element Access

| Method | Return Type | Description |
| :--- | :--- | :--- |
| `rows()` | `int` | Returns number of rows. |
| `cols()` | `int` | Returns number of columns. |
| `size()` | `int` | Returns total element count ($rows \times cols$). |
| `data()` | `float*` | Returns raw mutable pointer to the underlying buffer. |
| `dataPtr()` | `uintptr_t` | Returns memory address as integer for WebAssembly JS bridge. |
| `get(r, c)` | `float` | Retrieves element at $(r, c)$ using zero-based indexing. |
| `set(r, c, val)` | `void` | Sets element at $(r, c)$ to `val`. |
| `operator()(r, c)`| `float&` | Subscript operator for direct element read/write. |

```cpp
Matrix m(3, 3);
m(0, 0) = 1.0f;
m.set(1, 1, 2.5f);
float val = m.get(1, 1); // 2.5f
```

---

## 🎲 Weight Initialization Strategies

### `randomizeUniform(float low = -1.0f, float high = 1.0f)`
Populates the matrix using uniform distribution $\mathcal{U}(\text{low}, \text{high})$.

### `randomizeXavier(int fanIn, int fanOut)`
Initializes weights according to **Glorot / Xavier Uniform Initialization**:
$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{\text{fanIn} + \text{fanOut}}}, +\sqrt{\frac{6}{\text{fanIn} + \text{fanOut}}}\right)$$
Ideal for layers utilizing **Sigmoid** or **Tanh** activations, preventing vanishing/exploding gradients.

### `randomizeHe(int fanIn)`
Initializes weights according to **He / Kaiming Normal Initialization**:
$$W \sim \mathcal{N}\left(0, \sigma = \sqrt{\frac{2}{\text{fanIn}}}\right)$$
Essential for networks utilizing **ReLU** or **Leaky ReLU** activations to sustain gradient variance across deep layers.

---

## ⚡ Mathematical Operations (SIMD Accelerated)

All binary operations check dimension compatibility and use WebAssembly 128-bit SIMD intrinsics for maximum hardware throughput.

### 1. Vector-Matrix Dot Product (`vectorMatrixMultiply`)
The computational core of neural network forward inference ($\mathbf{z} = \mathbf{x} \cdot W$).
```cpp
static void vectorMatrixMultiply(const float* vec, const Matrix& mat, float* out);
```
- Multiplies a $1 \times K$ input vector by a $K \times N$ weight matrix, writing into a $1 \times N$ output vector.
- Vectorized using 4-wide float registers (`v128_t` / `wasm_f32x4_add`, `wasm_f32x4_mul`).

### 2. General Matrix Multiplication (`Matrix::dot`)
Calculates the matrix product $C = A \cdot B$:
```cpp
Matrix A(64, 128);
Matrix B(128, 32);
Matrix C(64, 32);

Matrix::dot(A, B, C); // C is 64x32
```

### 3. Hadamard Product (`Matrix::hadamard`)
Calculates element-wise multiplication ($C_{ij} = A_{ij} \cdot B_{ij}$):
```cpp
Matrix::hadamard(gradients, activationDerivatives, deltas);
```
Crucial for backpropagation chain-rule evaluation.

### 4. Element-Wise Addition & Subtraction (`Matrix::add`, `Matrix::subtract`)
Performs $C = A + B$ or $C = A - B$ with SIMD vectorization:
```cpp
Matrix::add(currentWeights, weightUpdates, newWeights);
```

### 5. Matrix Transposition (`Matrix::transpose`)
Transposes an $R \times C$ matrix into a $C \times R$ matrix:
```cpp
Matrix W(784, 64);
Matrix W_T(64, 784);
Matrix::transpose(W, W_T);
```
Used in backpropagation to pass output error deltas backwards to preceding layers.

### 6. In-Place Operations
- `void add(const Matrix& other)`: $M \leftarrow M + \text{other}$
- `void subtract(const Matrix& other)`: $M \leftarrow M - \text{other}$
- `void multiplyScalar(float scalar)`: $M \leftarrow M \cdot \text{scalar}$
- `void addScalar(float scalar)`: $M \leftarrow M + \text{scalar}$
- `void fill(float value)`: Fills all elements with `value`
- `void zero()`: Sets all elements to zero

---

## 🌐 WebAssembly & JavaScript Zero-Copy Interop

Through Emscripten's Embind and raw heap pointers, the frontend interacts directly with C++ matrices without JSON or array copy overhead:

```typescript
// Access raw WebAssembly Heap pointer
const weightsPtr: number = nn.getWeightsPtr(layerIndex);
const weightsSize: number = nn.getWeightsSize(layerIndex);

// Zero-copy Float32Array view over WebAssembly linear memory (HEAPF32)
const weightsView = new Float32Array(
  wasmModule.HEAPF32.buffer,
  weightsPtr,
  weightsSize
);

// Read or mutate values directly in WebAssembly heap
console.log('Weight at index 0:', weightsView[0]);
```
