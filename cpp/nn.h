#ifndef NN_H
#define NN_H

#include "matrix.h"
#include <cmath>
#include <iostream>
#include <vector>

class NeuralNetwork {
private:
   std::vector<int> layerSizes;
   int numLayers;
   double lrnRate;
   double lrStep; // Kept for compatibility

   std::vector<Matrix> layers;
   std::vector<Matrix> weights;
   std::vector<Matrix> biases;

   std::vector<Matrix> errors;
   std::vector<Matrix> deltas;

public:
   NeuralNetwork(int numInp, std::vector<int> hiddenSizes, int numOut, double lrnRate = 0.1);

   Matrix feedForward(const Matrix &input);
   Matrix feedForwardArray(const std::vector<double> &input); // Helper for JS array input

   void train(const Matrix &input, const Matrix &target);
   void trainArray(const std::vector<double> &input, const std::vector<double> &target);

   // Getters
   int getNumLayers() const;
   Matrix getLayer(int index) const;
   Matrix getWeights(int index) const;
   Matrix getBiases(int index) const;

   // Optimized getters for visualization
   double getNeuronVal(int layerIdx, int neuronIdx) const;
   double getWeightVal(int layerIdx, int fromIdx, int toIdx) const;
   int getLayerSize(int layerIdx) const;

   void resetActivations();

   double getLrnRate() const;
   void setLrnRate(double rate);

   double getLrStep() const;
   void setLrStep(double step);

   // For saving/loading
   void setWeights(int index, const Matrix &w);
   void setBiases(int index, const Matrix &b);
};

#endif
