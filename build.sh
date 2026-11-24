#!/bin/bash
set -euo pipefail

echo "=============================================="
echo " Compiling C++ SIMD Neural Network to WebAssembly"
echo "=============================================="

mkdir -p src/wasm public/wasm

em++ -std=c++20 \
  -O3 \
  -flto \
  -msimd128 \
  -sEXPORT_ES6=1 \
  -sMODULARIZE=1 \
  -sEXPORT_NAME='createNNEngine' \
  -sALLOW_MEMORY_GROWTH=1 \
  -lembind \
  cpp/src/matrix.cpp \
  cpp/src/activations.cpp \
  cpp/src/neural_network.cpp \
  cpp/src/bindings.cpp \
  -o src/wasm/nn_engine.js

cp src/wasm/nn_engine.wasm public/wasm/nn_engine.wasm

echo "Done! Generated:"
echo " - src/wasm/nn_engine.js"
echo " - src/wasm/nn_engine.wasm"
echo " - public/wasm/nn_engine.wasm"
