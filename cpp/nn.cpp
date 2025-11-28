#include "nn.h"

NeuralNetwork::NeuralNetwork(int numInp, std::vector<int> hiddenSizes, int numOut, double lrnRate)
    : lrnRate(lrnRate), lrStep(0) {

   layerSizes.push_back(numInp);
   for (int size : hiddenSizes) {
      layerSizes.push_back(size);
   }
   layerSizes.push_back(numOut);

   numLayers = layerSizes.size();

   // Initialize layers (empty placeholders)
   // We'll resize them properly during feedForward or init
   // Actually, let's just reserve space.
   layers.reserve(numLayers);
   // Fill with empty matrices for now so we can index them
   for (int i = 0; i < numLayers; i++) {
      layers.push_back(Matrix(0, 0));
   }

   // Initialize weights and biases
   for (int i = 0; i < numLayers - 1; i++) {
      int inSize = layerSizes[i];
      int outSize = layerSizes[i + 1];

      Matrix w(inSize, outSize);
      w.randomWeights();
      weights.push_back(w);

      Matrix b(1, outSize);
      b.randomWeights();
      biases.push_back(b);
   }

   // Resize errors and deltas
   errors.resize(numLayers, Matrix(0, 0));
   deltas.resize(numLayers, Matrix(0, 0));
}

Matrix NeuralNetwork::feedForward(const Matrix &input) {
   // Input layer
   // If input is 1D (1 row), good.
   layers[0] = input;

   for (int i = 0; i < numLayers - 1; i++) {
      // z = a[i] * W[i] + b[i]
      Matrix z = Matrix::dot(layers[i], weights[i]);
      z.add(biases[i]);

      // a[i+1] = sigmoid(z)
      z.map([](double x) { return 1.0 / (1.0 + std::exp(-x)); });

      layers[i + 1] = z;
   }

   return layers[numLayers - 1];
}

Matrix NeuralNetwork::feedForwardArray(const std::vector<double> &input) {
   Matrix m = Matrix::convertFromArray(input);
   return feedForward(m);
}

void NeuralNetwork::train(const Matrix &input, const Matrix &target) {
   // 1. Forward pass
   Matrix outputs = feedForward(input);

   int L = numLayers - 1;

   // 2. Output error = target - output
   errors[L] = Matrix::subtract(target, outputs);

   // 3. Output delta
   // derivative of sigmoid(output) = output * (1 - output)
   Matrix outputDerivs = outputs; // Copy
   outputDerivs.map([](double x) { return x * (1.0 - x); });

   deltas[L] = Matrix::multiply(errors[L], outputDerivs);

   // 4. Backprop
   for (int i = L - 1; i > 0; i--) {
      // error[i] = delta[i+1] * W[i]^T
      Matrix wT = Matrix::transpose(weights[i]);
      errors[i] = Matrix::dot(deltas[i + 1], wT);

      // delta[i]
      Matrix layerI = layers[i];
      Matrix derivs = layerI; // Copy
      derivs.map([](double x) { return x * (1.0 - x); });

      deltas[i] = Matrix::multiply(errors[i], derivs);
   }

   // 5. Update weights & biases
   for (int i = 0; i < numLayers - 1; i++) {
      Matrix aT = Matrix::transpose(layers[i]);
      Matrix weightDeltas = Matrix::dot(aT, deltas[i + 1]);

      weightDeltas.multiply(lrnRate);
      weights[i].add(weightDeltas);

      // Update biases
      Matrix biasDeltas = deltas[i + 1]; // Copy
      biasDeltas.multiply(lrnRate);
      biases[i].add(biasDeltas);
   }
}

void NeuralNetwork::trainArray(const std::vector<double> &input, const std::vector<double> &target) {
   Matrix mIn = Matrix::convertFromArray(input);
   Matrix mTgt = Matrix::convertFromArray(target);
   train(mIn, mTgt);
}

int NeuralNetwork::getNumLayers() const { return numLayers; }

Matrix NeuralNetwork::getLayer(int index) const {
   if (index < 0 || index >= numLayers)
      return Matrix(0, 0);
   return layers[index];
}

Matrix NeuralNetwork::getWeights(int index) const {
   if (index < 0 || index >= weights.size())
      return Matrix(0, 0);
   return weights[index];
}

Matrix NeuralNetwork::getBiases(int index) const {
   if (index >= 0 && index < numLayers - 1) {
      return biases[index];
   }
   return Matrix(0, 0);
}

double NeuralNetwork::getNeuronVal(int layerIdx, int neuronIdx) const {
   if (layerIdx >= 0 && layerIdx < numLayers) {
      return layers[layerIdx].at(0, neuronIdx);
   }
   return 0.0;
}

void NeuralNetwork::resetActivations() {
   for (auto &layer : layers) {
      layer.multiply(0.0);
   }
}

double NeuralNetwork::getWeightVal(int layerIdx, int fromIdx, int toIdx) const {
   if (layerIdx >= 0 && layerIdx < numLayers - 1) {
      return weights[layerIdx].at(fromIdx, toIdx);
   }
   return 0.0;
}

int NeuralNetwork::getLayerSize(int layerIdx) const {
   if (layerIdx >= 0 && layerIdx < numLayers) {
      return layerSizes[layerIdx];
   }
   return 0;
}

double NeuralNetwork::getLrnRate() const {
   return lrnRate;
}
void NeuralNetwork::setLrnRate(double rate) { lrnRate = rate; }

double NeuralNetwork::getLrStep() const { return lrStep; }
void NeuralNetwork::setLrStep(double step) { lrStep = step; }

void NeuralNetwork::setWeights(int index, const Matrix &w) {
   if (index >= 0 && index < weights.size()) {
      weights[index] = w;
   }
}

void NeuralNetwork::setBiases(int index, const Matrix &b) {
   if (index >= 0 && index < biases.size()) {
      biases[index] = b;
   }
}
