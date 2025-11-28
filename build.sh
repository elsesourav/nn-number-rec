#!/bin/bash
echo "Compiling C++ to WebAssembly..."
# Added optimization flags:
# -O3: Aggressive optimization for speed
# -flto: Link Time Optimization
# -msimd128: Enable SIMD instructions (great for matrix ops)
emcc cpp/wasm.cpp cpp/matrix.cpp cpp/tensor.cpp cpp/nn.cpp -lembind -o wasmJs/wasm.js -s MODULARIZE=1 -s EXPORT_NAME='createMathModule' -O3 -flto -msimd128
echo "Done! Output saved to wasmJs/wasm.js"
