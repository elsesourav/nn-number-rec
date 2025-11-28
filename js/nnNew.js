"use strict";

class NeuralNetwork {
   /**
    * @param {number} numInp           - number of input neurons
    * @param {number[]} hiddenSizes    - array of hidden layer sizes, e.g. [16, 16, 32]
    * @param {number} numOut           - number of output neurons
    * @param {number} lrnRate          - learning rate
    * @param {number} lrStep           - (optional) your existing param, kept for compatibility
    * @param {number} viewW            - for NNView (optional)
    * @param {number} viewH            - for NNView (optional)
    */
   constructor(
      numInp,
      hiddenSizes = [],
      numOut,
      lrnRate = 1,
      lrStep,
      viewW,
      viewH,
      wasmModule
   ) {
      if (!wasmModule) {
         throw new Error("Wasm module is required for NeuralNetwork");
      }
      this.wasm = wasmModule;
      this.Matrix = this.wasm.Matrix;

      // Architecture: [input, ...hidden, output]
      this.layerSizes = [numInp, ...hiddenSizes, numOut];
      this.numLayers = this.layerSizes.length; // includes input + all hidden + output

      this._lrnRate = lrnRate;
      this.lrStep = lrStep;

      // Activations of each layer (Matrix objects)
      // layers[0] = inputs, layers[last] = outputs
      this.layers = new Array(this.numLayers).fill(null);

      // Biases[i] is bias for layer (i+1)
      this.biases = [];
      // Weights[i] connects layer i -> i+1
      this.weights = [];

      // Initialize weights & biases
      for (let i = 0; i < this.numLayers - 1; i++) {
         const inSize = this.layerSizes[i];
         const outSize = this.layerSizes[i + 1];

         // weights: (inSize x outSize)
         const w = new this.Matrix(inSize, outSize);
         w.randomWeights();
         this.weights.push(w);

         // bias: (1 x outSize)
         const b = new this.Matrix(1, outSize);
         b.randomWeights();
         this.biases.push(b);
      }

      // For debugging / inspection if you want (not mandatory)
      this.$errors = [];
      this.$deltas = [];

      // View (if you still want to visualize)
      if (typeof NNView !== "undefined") {
         this.view = new NNView(this, viewW, viewH);
         if (this.view && this.view.draw) this.view.draw();
      }
   }

   // ---------- FEED FORWARD ----------
   feedForward(inputArray) {
      // Clean up previous input layer if it exists
      if (this.layers[0]) this.layers[0].delete();

      // Convert input to Matrix and store as first layer
      // inputArray is expected to be a flat array or array of arrays?
      // Matrix constructor from JS array handles array of arrays or flat array if we implement it.
      // My C++ constructor takes emscripten::val which expects array of arrays (rows).
      // If inputArray is flat [x, y, z], we need to wrap it [[x, y, z]].
      let inputMatrixData = inputArray;
      if (inputArray.length > 0 && !Array.isArray(inputArray[0])) {
         inputMatrixData = [inputArray];
      }
      this.layers[0] = new this.Matrix(inputMatrixData);

      // Go through each layer
      for (let i = 0; i < this.numLayers - 1; i++) {
         // z = a[i] · W[i] + b[i]
         let z = this.Matrix.dot(this.layers[i], this.weights[i]);
         z.add(this.biases[i]);

         // a[i+1] = sigmoid(z)
         z.mapSigmoid();

         // Clean up old layer before overwriting
         if (this.layers[i + 1]) this.layers[i + 1].delete();
         this.layers[i + 1] = z;
      }

      if (this.view && this.view.aniStart) this.view.aniStart();

      // Output is last layer
      return this.layers[this.numLayers - 1];
   }

   // ---------- BACKPROP / TRAIN ----------
   train(inputArray, targetArray) {
      // 1. Forward pass
      const outputs = this.feedForward(inputArray);

      // 2. Build targets Matrix
      let targetMatrixData = targetArray;
      if (targetArray.length > 0 && !Array.isArray(targetArray[0])) {
         targetMatrixData = [targetArray];
      }
      const targets = new this.Matrix(targetMatrixData);

      const L = this.numLayers - 1; // index of last layer (output)

      // Clean up old errors/deltas
      if (this.$errors) this.$errors.forEach((m) => m && m.delete());
      if (this.$deltas) this.$deltas.forEach((m) => m && m.delete());

      this.$errors = new Array(this.numLayers).fill(null);
      this.$deltas = new Array(this.numLayers).fill(null);

      // 3. Output error = target - output
      this.$errors[L] = this.Matrix.subtract(targets, outputs);
      targets.delete();

      // 4. Output delta = error * derivative(sigmoid(output))
      // Create copy of outputs for derivative calculation
      const outputDerivs = outputs.clone();
      outputDerivs.mapDSigmoid();

      this.$deltas[L] = this.Matrix.multiply(this.$errors[L], outputDerivs);
      outputDerivs.delete();

      // 5. Backprop through hidden layers: from last hidden → first hidden
      for (let i = L - 1; i > 0; i--) {
         // i is hidden layer index
         // error[i] = delta[i+1] · W[i]^T
         const wT = this.Matrix.transpose(this.weights[i]);
         this.$errors[i] = this.Matrix.dot(this.$deltas[i + 1], wT);
         wT.delete();

         // delta[i] = error[i] * derivative(sigmoid(a[i]))
         const layerI = this.layers[i];
         const derivs = layerI.clone();
         derivs.mapDSigmoid();

         this.$deltas[i] = this.Matrix.multiply(this.$errors[i], derivs);
         derivs.delete();
      }

      // 6. Gradient descent: update weights & biases
      for (let i = 0; i < this.numLayers - 1; i++) {
         // a[i]^T · delta[i+1]
         const aT = this.Matrix.transpose(this.layers[i]);
         let weightDeltas = this.Matrix.dot(aT, this.$deltas[i + 1]);
         aT.delete();

         // scale by learning rate
         weightDeltas.multiplyScalar(this.lrnRate);

         // update weights
         this.weights[i].add(weightDeltas);
         weightDeltas.delete();

         // update biases
         const biasDeltas = this.$deltas[i + 1].clone();
         biasDeltas.multiplyScalar(this.lrnRate);
         this.biases[i].add(biasDeltas);
         biasDeltas.delete();
      }
   }

   // ---------- Getters / Setters for compatibility ----------
   get lrnRate() {
      return this._lrnRate;
   }
   set lrnRate(v) {
      this._lrnRate = v;
   }

   // For convenience, like your old code:
   get inputs() {
      return this.layers[0];
   }
   set inputs(m) {
      this.layers[0] = m;
   }

   get outputs() {
      return this.layers[this.numLayers - 1];
   }
   set outputs(m) {
      this.layers[this.numLayers - 1] = m;
   }

   // If you really want "hidden0", "hidden1" style access,
   // you can emulate like this:
   get hiddenLayers() {
      // layers[1]..layers[numLayers-2]
      return this.layers.slice(1, this.numLayers - 1);
   }
}

// Same sigmoid as you have
function sigmoid(x, derivative = false) {
   if (derivative) return x * (1 - x);
   return 1 / (1 + Math.exp(-x));
}
