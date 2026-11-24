export interface VectorInt {
  push_back(val: number): void;
  size(): number;
  get(idx: number): number;
  delete(): void;
}

export interface VectorFloat {
  push_back(val: number): void;
  size(): number;
  get(idx: number): number;
  delete(): void;
}

export interface WasmNeuralNetwork {
  // Zero-copy buffer pointers
  getInputBufferPtr(): number;
  getOutputBufferPtr(): number;
  getInputSize(): number;
  getOutputSize(): number;

  // Inference and training
  feedForward(): void;
  feedForwardFromPtr(inputPtr: number): void;
  train(inputPtr: number, targetPtr: number): number;
  trainBatch(inputsPtr: number, targetsPtr: number, batchSize: number): number;

  // Introspection for visualizer
  getNumLayers(): number;
  getLayerSize(layerIdx: number): number;
  getNeuronValue(layerIdx: number, neuronIdx: number): number;
  getWeightValue(layerIdx: number, fromIdx: number, toIdx: number): number;
  getBiasValue(layerIdx: number, neuronIdx: number): number;
  resetActivations(): void;

  // Hyperparameters
  learningRate: number;
  momentum: number;

  // Serialization
  getWeightsPtr(layerIdx: number): number;
  getBiasesPtr(layerIdx: number): number;
  getWeightsSize(layerIdx: number): number;
  getBiasesSize(layerIdx: number): number;
  setWeightsFromPtr(layerIdx: number, ptr: number, size: number): void;
  setBiasesFromPtr(layerIdx: number, ptr: number, size: number): void;
  exportToJson(): string;
  getTopology(): VectorInt;

  delete(): void;
}

export interface NNEngineModule {
  NeuralNetwork: new (
    topology: VectorInt,
    learningRate: number,
    momentum: number,
    hiddenAct: string,
    outputAct: string
  ) => WasmNeuralNetwork;
  VectorInt: new () => VectorInt;
  VectorFloat: new () => VectorFloat;
  HEAPF32: Float32Array;
  _malloc: (bytes: number) => number;
  _free: (ptr: number) => void;
}

export interface SerializedLayer {
  weights: number[];
  biases: number[];
}

export interface SerializedModel {
  version: number;
  topology: number[];
  learningRate: number;
  momentum?: number;
  pixelResolution: number; // e.g. 28 -> 28x28 = 784
  layers?: SerializedLayer[];
  // Legacy model compatibility index signature
  [key: string]: unknown;
}
