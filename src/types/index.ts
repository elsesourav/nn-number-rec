export interface LayerSpec {
  id: string;
  neurons: number;
}

export interface NetworkConfig {
  pixelResolution: number; // e.g. 28 -> inputSize = 784
  inputSize: number;
  outputSize: number; // 10 for digits 0-9
  hiddenLayers: LayerSpec[];
  learningRate: number;
  momentum: number;
  hiddenActivation: 'sigmoid' | 'relu' | 'leaky_relu' | 'tanh';
  outputActivation: 'sigmoid' | 'softmax';
  useBias: boolean; // Option to enable or disable layer biases
}

export interface PredictionResult {
  digit: number;
  confidence: number;
  probabilities: number[];
}

export interface TrainingStats {
  samplesTrained: number;
  currentLoss: number;
  lastLossChange: number;
  accuracy: number;
  history: { loss: number; accuracy?: number }[];
}

export interface SavedPreset {
  id: string;
  name: string;
  timestamp: number;
  config: NetworkConfig;
  model?: any; // SerializedModel
  samplesTrained: number;
  currentLoss: number;
}

