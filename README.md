<h1 align="center">🧠 Neural Network Number Recognition (WASM + React)</h1>

<p align="center">
  A high-performance handwritten digit recognition application powered by a <strong>C++ SIMD WebAssembly (Wasm)</strong> neural network engine, rebuilt from the ground up with <strong>React 19, Vite, TypeScript, and Tailwind CSS</strong>.
</p>

---

## 🚀 Key Highlights & Architectural Upgrades

### 1. High-Performance C++ WebAssembly Engine (`cpp/`)
* **Flat 1D Row-Major Matrices (`std::vector<float>`):** Replaces jagged nested vectors (`vector<vector<double>>`) with contiguous cache-friendly arrays.
* **128-bit SIMD Vectorization (`<wasm_simd128.h>`):** Core dot product, matrix-vector multiplication, Hadamard product, and activation derivatives run on 4-wide 32-bit float vector registers (`wasm_f32x4`).
* **Zero-Copy Data Transfer:** Embind exposes direct `Float32Array` views over the WASM linear memory heap (`HEAPF32`). Pixel buffers and prediction probabilities are passed across the JS/WASM boundary without serialization or memory allocation.
* **SGD with Momentum ($\beta = 0.85$):** Drastically accelerates convergence on handwritten digit patterns.

### 2. Configurable Dynamic Architecture
* **Configurable Input Layer Resolution:** Control the input layer dimension based on image pixel resolution ($14\times14 = 196$, $16\times16 = 256$, $20\times20 = 400$, or standard MNIST $28\times28 = 784$ inputs).
* **Dynamic 2-to-N Hidden Layers:** Add or remove any number of hidden layers dynamically.
* **Per-Layer Node Count:** Configure neuron counts individually per layer (8 to 512 nodes) with instant parameter count calculation.
* **Configurable Activation Functions:** Choose between **Sigmoid**, **ReLU**, **LeakyReLU**, and **Tanh**.
* **Dynamic Hyperparameters:** Real-time adjustments to Learning Rate and Momentum without rebuilding the network.

### 3. Anti-Aliased Pressure-Sensitive Drawing Input
* **Pointer Events API:** Unified handling for mouse, Apple Pencil / stylus, and touchscreen devices.
* **Quadratic Bézier Smoothing:** Eliminates jagged polylines through midpoint interpolation.
* **Pressure Sensitivity:** Modulates stroke thickness according to pen pressure and brush sizing.
* **MNIST-Standard Centering:** Computes center-of-mass $(\bar{x}, \bar{y})$ and shifts the drawn digit to the center of the grid.
* **Live Centered Preview:** An offscreen downscaled thumbnail shows the exact $N\times N$ matrix input the neural network sees.

### 4. Interactive Visualization & Model Persistence
* **Live Feedforward Flow:** Animated visualizer rendering network connections, color-coded weights (green for positive, purple for negative), and animated feedforward activation pulses.
* **Model Storage:** Save and load models to browser `localStorage`, export/import full JSON models, and load bundled pre-trained weights.

---

## 📁 Code Structure

```
number-recognition/
├── cpp/                               # C++ WebAssembly Engine
│   ├── src/
│   │   ├── matrix.h                   # Flat-array Matrix with SIMD operations
│   │   ├── matrix.cpp
│   │   ├── activations.h              # Sigmoid, ReLU, LeakyReLU, Tanh, Softmax
│   │   ├── activations.cpp
│   │   ├── neural_network.h           # Configurable N-layer Neural Network
│   │   ├── neural_network.cpp         # Forward pass, backpropagation, batch training
│   │   └── bindings.cpp               # Embind bindings & typed memory views
│   └── build.sh                       # Emscripten compilation script
│
├── src/                               # React + TypeScript Frontend
│   ├── components/
│   │   ├── DrawingCanvas/
│   │   │   └── DrawingCanvas.tsx      # Smooth drawing canvas with Pointer Events
│   │   ├── NetworkConfig/
│   │   │   └── NetworkConfigurator.tsx# Dynamic N-layer architecture editor
│   │   ├── Visualization/
│   │   │   └── NetworkVisualizer.tsx  # Dynamic SVG/Canvas live network graph
│   │   ├── Training/
│   │   │   └── TrainingPanel.tsx      # Probability bars, numpad training, loss
│   │   └── ModelManager/
│   │       └── ModelManager.tsx       # LocalStorage, JSON import/export
│   │
│   ├── hooks/
│   │   ├── useWasmModule.ts           # Asynchronous WebAssembly loader singleton
│   │   └── useNeuralNetwork.ts        # Network lifecycle, training, and inference hook
│   │
│   ├── utils/
│   │   └── imageProcessing.ts         # Downscaling, thresholding, and center-of-mass shift
│   │
│   ├── wasm/
│   │   ├── types.ts                   # TypeScript interfaces for WASM API
│   │   ├── wasmBridge.ts              # Zero-copy bridge wrapper
│   │   ├── nn_engine.js               # Generated Emscripten ES6 glue
│   │   └── nn_engine.wasm             # Compiled WebAssembly binary
│   │
│   ├── types/
│   │   └── index.ts                   # App-wide TypeScript definitions
│   │
│   ├── App.tsx                        # Master layout & tabs
│   ├── main.tsx                       # React application entry point
│   └── index.css                      # Tailwind CSS v4 styling
│
├── public/
│   ├── wasm/
│   │   └── nn_engine.wasm             # Static asset copy for Vite
│   ├── model.json                     # Bundled pre-trained model weights
│   └── favicon.png
│
├── vite.config.ts                     # Vite + Tailwind configuration
├── tsconfig.json                      # TypeScript configuration
├── package.json
└── README.md
```

---

## 🛠️ Getting Started

### Prerequisites
* **Node.js:** v18 or higher (v24 recommended)
* **npm:** v9 or higher
* *(Optional)* **Emscripten SDK (`em++`):** Only needed if modifying the C++ engine in `cpp/src/`. The repository comes with pre-compiled WebAssembly binaries in `public/wasm/`.

### 1. Installation
```bash
npm install
```

### 2. Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 3. Production Build
```bash
npm run build
```
Generates an optimized static bundle in `dist/`.

### 4. (Optional) Recompiling C++ to WebAssembly
If you make changes to `cpp/src/`:
```bash
npm run build:wasm
```

---

## 🧠 How to Use

1. **Draw a Digit:** Use the drawing pad on the left to draw any number from 0 to 9.
2. **Instant Prediction:** The network immediately computes feedforward probabilities and highlights the winning digit with its confidence score.
3. **Train the Network:** Click any number button on the numpad to train the network on your drawing. Notice the loss drop and the visualizer animate with live backpropagated weights!
4. **Customize Topology:** Open the **N-Layer Architecture** tab to add or remove hidden layers, adjust neuron counts, change the pixel grid resolution (e.g., $20\times20$ vs $28\times28$), and click **Apply & Initialize**.
5. **Save & Export:** Use the **Model Storage** tab to persist your trained weights in browser storage or export them as a reusable JSON file.

---

## 📄 License
MIT License.
