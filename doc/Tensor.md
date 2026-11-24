# Tensor & Multidimensional Architecture Documentation

This document explains the mathematical foundations, memory layout, and operational design of **Tensors** in the Neural Network Digit Recognition engine.

---

## 📐 1. Tensor Hierarchy in Neural Networks

In modern deep learning, a **Tensor** is a geometric object that maps geometric vectors, scalars, and other tensors to an $N$-dimensional coordinate array. Within this engine, tensors of varying ranks represent model parameters, activations, and dataset samples:

| Rank | Mathematical Entity | Project Implementation | Dimensionality / Shape | Examples |
| :---: | :---: | :---: | :---: | :---: |
| **0** | **Scalar** | `float` | $()$ | Learning rate $\alpha$, Momentum $\beta$, Loss $\mathcal{L}$ |
| **1** | **Vector** | `std::vector<float>` / `Float32Array` | $(N)$ | Input vector $\mathbf{x} \in \mathbb{R}^{784}$, Bias $\mathbf{b} \in \mathbb{R}^{64}$, Target $\mathbf{y} \in \mathbb{R}^{10}$ |
| **2** | **Matrix** | `Matrix` (`cpp/src/matrix.h`) | $(R \times C)$ | Synaptic weights $W^{[l]} \in \mathbb{R}^{N_{l-1} \times N_l}$, Batch inputs $\mathbf{X} \in \mathbb{R}^{B \times 784}$ |
| **3** | **Volume** | Contiguous Batch Tensor | $(B \times H \times W)$ | Batch of 2D raw handwritten digits |

---

## ⚡ 2. Memory Architecture: Contiguous Strides vs. Ragged Arrays

### The Problem with Nested Arrays
Legacy neural network implementations often represented multidimensional data using nested structures such as `std::vector<std::vector<double>>` or JavaScript arrays of arrays `number[][]`.

This introduces critical performance bottlenecks:
1. **Pointer Chasing**: Every sub-array is an isolated heap allocation; reading consecutive elements requires traversing multiple pointer addresses.
2. **Cache Misses**: Memory is scattered non-contiguously throughout RAM, evicting CPU L1/L2 cache lines (64 bytes).
3. **No SIMD Vectorization**: Vector registers require contiguous 16-byte memory alignment to load 4 floats in a single instruction (`v128_load`).

### The Modern Contiguous 1D Row-Major Solution
The modern C++ engine flattens all $N$-dimensional tensors into a **single, contiguous 1D block** of 32-bit floats (`std::vector<float>`).

#### Strided Flat Indexing Formula
For an arbitrary $D$-dimensional tensor with shape $(S_0, S_1, \dots, S_{D-1})$, the flat 1D buffer offset for coordinate $(i_0, i_1, \dots, i_{D-1})$ is computed using **strides**:

$$\text{Offset}(i_0, i_1, \dots, i_{D-1}) = \sum_{k=0}^{D-1} i_k \cdot \text{stride}_k$$

Where the stride for dimension $k$ represents the number of elements skipped to advance by one unit along that axis:

$$\text{stride}_k = \prod_{m=k+1}^{D-1} S_m, \quad \text{with } \text{stride}_{D-1} = 1$$

#### 2D Example (Matrix of Shape $R \times C$):
$$\text{stride}_0 = C, \quad \text{stride}_1 = 1 \implies \text{Offset}(r, c) = r \cdot C + c$$

#### 3D Example (Batch of $B$ Images of Size $H \times W$):
$$\text{stride}_0 = H \cdot W, \quad \text{stride}_1 = W, \quad \text{stride}_2 = 1$$
$$\text{Offset}(b, h, w) = b \cdot (H \cdot W) + h \cdot W + w$$

---

## 🧠 3. Neural Network Tensor Flow & Dimensionality

During feedforward inference and backpropagation, tensors undergo linear and non-linear transformations:

```
[Input Canvas: 28×28] 
         │ (Flatten & Center-of-Mass)
         ▼
[Input Tensor x: 784 × 1]
         │
         ├─── Weight Matrix W[1]: [64 × 784]
         ├─── Bias Vector b[1]:   [64 × 1]
         ▼
[Linear Dot Product z[1]: 64 × 1]
         │ (Non-linear Activation: Leaky ReLU / ReLU / Sigmoid / Tanh)
         ▼
[Activation Tensor a[1]: 64 × 1]
         │
         ├─── Weight Matrix W[2]: [64 × 64]
         ├─── Bias Vector b[2]:   [64 × 1]
         ▼
[Activation Tensor a[2]: 64 × 1]
         │
         ├─── Weight Matrix W[3]: [10 × 64]
         ├─── Bias Vector b[3]:   [10 × 1]
         ▼
[Output Logits z[3]: 10 × 1]
         │ (Softmax Normalization)
         ▼
[Probability Tensor ŷ: 10 × 1] ─── Target Vector y [10 × 1]
                                        │
                                        ▼
                         [Cross-Entropy Loss & Residual Gradient δ: 10 × 1]
                                        │
                                        ▼ (Backpropagation Chain Rule)
                         [Weight Velocity Gradients vt: 64 × 784, etc.]
```

---

## 📦 4. High-Throughput Batch Tensor Training

For training efficiency, the C++ engine provides a vectorized batch training interface:

```cpp
// cpp/src/neural_network.h
float trainBatch(uintptr_t inputsPtr, uintptr_t targetsPtr, int batchSize);
```

### Parameters:
- `inputsPtr`: Pointer to a contiguous rank-2 input tensor $\mathbf{X} \in \mathbb{R}^{B \times N_{\text{in}}}$, containing $B$ samples of flattened pixel floats.
- `targetsPtr`: Pointer to a contiguous rank-2 one-hot target tensor $\mathbf{Y} \in \mathbb{R}^{B \times 10}$, containing $B$ target distributions.
- `batchSize`: Batch count ($B$).

### Execution:
1. Iterates through sample slices using row stride $N_{\text{in}}$.
2. Executes forward inference with SIMD dot products.
3. Computes output error residuals $\boldsymbol{\delta}^{[L]} = \hat{\mathbf{y}} - \mathbf{y}$.
4. Propagates deltas backwards through transposed weight matrices.
5. Accumulates momentum velocities:
   $$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)\alpha \nabla_W \mathcal{L}$$
6. Updates weights in-place ($W \leftarrow W - \mathbf{v}_t$).

---

## 🌐 5. Zero-Copy WebAssembly Bridge Interface

In `src/wasm/wasmBridge.ts`, the frontend maps JavaScript typed arrays directly over the WebAssembly linear memory heap (`HEAPF32`) using raw pointers:

```typescript
// Allocate flat target tensor in WebAssembly memory once
const targetPtr = wasmModule._malloc(outputSize * Float32Array.BYTES_PER_ELEMENT);
const targetView = new Float32Array(wasmModule.HEAPF32.buffer, targetPtr, outputSize);

// One-hot encode target digit into WebAssembly tensor buffer without copying
targetView.fill(0);
targetView[targetDigit] = 1.0;

// Execute C++ training directly on memory addresses
const loss = nn.train(inputBufferPtr, targetPtr);
```

### Benefits:
- **Zero Overhead**: No serialization, JSON parsing, or garbage collection spikes.
- **Maximum Frame Rate**: Visualizer and training loops run at a stable 60 FPS.
- **Memory Safety**: Buffer boundaries are verified; memory growth events automatically refresh heap array views.
