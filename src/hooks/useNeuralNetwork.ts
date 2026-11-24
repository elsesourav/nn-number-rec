import { useState, useEffect, useCallback, useRef } from 'react';
import type { NNEngineModule, SerializedModel } from '../wasm/types';
import { NeuralNetworkBridge } from '../wasm/wasmBridge';
import type { NetworkConfig, PredictionResult, TrainingStats, SavedPreset } from '../types';

const STORAGE_KEY = 'nn_app_v2_model';
const NAMED_PRESETS_KEY = 'nn_named_presets';

const DEFAULT_CONFIG: NetworkConfig = {
  pixelResolution: 28,
  inputSize: 784,
  outputSize: 10,
  hiddenLayers: [
    { id: '1', neurons: 64 },
    { id: '2', neurons: 64 },
  ],
  learningRate: 0.04,
  momentum: 0.5,
  hiddenActivation: 'leaky_relu',
  outputActivation: 'softmax',
  useBias: true,
};

export function useNeuralNetwork(wasmModule: NNEngineModule | null) {
  const [bridge, setBridge] = useState<NeuralNetworkBridge | null>(null);
  const bridgeRef = useRef<NeuralNetworkBridge | null>(null);
  const [config, setConfig] = useState<NetworkConfig>(DEFAULT_CONFIG);
  const configRef = useRef<NetworkConfig>(DEFAULT_CONFIG);
  configRef.current = config;

  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isAnimating, setIsAnimating] = useState<boolean>(false);
  const [animationDuration, setAnimationDuration] = useState<number>(700);
  const [trainingStats, setTrainingStats] = useState<TrainingStats>({
    samplesTrained: 0,
    currentLoss: 0,
    lastLossChange: 0,
    accuracy: 0,
    history: [],
  });

  const lastPixelsRef = useRef<Float32Array | null>(null);

  // Initialize or Rebuild Network with current or custom configuration
  const rebuildNetwork = useCallback(
    (customConfig?: NetworkConfig) => {
      if (!wasmModule) return;

      const targetConfig = customConfig || configRef.current;

      // Safely dispose the old instance before creating a new one
      if (bridgeRef.current) {
        bridgeRef.current.dispose();
        bridgeRef.current = null;
      }

      const topology = [
        targetConfig.inputSize,
        ...targetConfig.hiddenLayers.map((l) => l.neurons),
        targetConfig.outputSize,
      ];

      const newBridge = new NeuralNetworkBridge(
        wasmModule,
        topology,
        targetConfig.learningRate,
        targetConfig.momentum,
        targetConfig.hiddenActivation,
        targetConfig.outputActivation,
        targetConfig.useBias ?? true
      );

      bridgeRef.current = newBridge;
      setBridge(newBridge);

      if (customConfig) {
        setConfig(customConfig);
      }
      setPrediction(null);
      setTrainingStats({
        samplesTrained: 0,
        currentLoss: 0,
        lastLossChange: 0,
        accuracy: 0,
        history: [],
      });
    },
    [wasmModule]
  );

  // Initialize network once WASM module is ready
  useEffect(() => {
    if (wasmModule && !bridgeRef.current) {
      rebuildNetwork();
    }
  }, [wasmModule, rebuildNetwork]);

  // Clean up on component unmount ONLY
  useEffect(() => {
    return () => {
      if (bridgeRef.current) {
        bridgeRef.current.dispose();
        bridgeRef.current = null;
      }
    };
  }, []);

  // Inference (Predict)
  const predict = useCallback(
    (pixels: Float32Array, triggerAnimation = true, customDuration = 700) => {
      const activeBridge = bridgeRef.current;
      if (!activeBridge || activeBridge.disposed) return;
      lastPixelsRef.current = pixels;

      // Check if blank
      let sum = 0;
      for (let i = 0; i < pixels.length; i++) sum += pixels[i];
      if (sum < 0.05) {
        activeBridge.resetActivations();
        setPrediction(null);
        return;
      }

      const output = activeBridge.predict(pixels);
      const probabilities = Array.from(output);
      let maxVal = -1;
      let maxIdx = -1;

      for (let i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxVal) {
          maxVal = probabilities[i];
          maxIdx = i;
        }
      }

      setPrediction({
        digit: maxIdx,
        confidence: maxVal,
        probabilities,
      });

      if (triggerAnimation) {
        setAnimationDuration(customDuration);
        setIsAnimating(true);
        if (customDuration > 0) {
          setTimeout(() => setIsAnimating(false), customDuration + 40);
        } else {
          setIsAnimating(false);
        }
      }
    },
    []
  );

  // Train on a single sample
  const train = useCallback(
    (pixels: Float32Array, label: number, customDuration = 700) => {
      const activeBridge = bridgeRef.current;
      if (!activeBridge || activeBridge.disposed) return;

      const outputDim = configRef.current.outputSize;
      const target = new Float32Array(outputDim).fill(0);
      if (label >= 0 && label < outputDim) {
        target[label] = 1.0;
      }

      const loss = activeBridge.trainSingle(pixels, target);

      setTrainingStats((prev) => ({
        samplesTrained: prev.samplesTrained + 1,
        currentLoss: loss,
        lastLossChange: loss - prev.currentLoss,
        accuracy: prev.accuracy,
        history: [...prev.history.slice(-30), { loss }],
      }));

      // Re-run inference with updated weights to show progression
      predict(pixels, true, customDuration);
    },
    [predict]
  );

  // Live hyperparameter updates
  const updateLearningRate = useCallback(
    (lr: number) => {
      if (bridgeRef.current && !bridgeRef.current.disposed) {
        bridgeRef.current.learningRate = lr;
      }
      setConfig((prev) => ({ ...prev, learningRate: lr }));
    },
    []
  );

  const updateMomentum = useCallback(
    (m: number) => {
      if (bridgeRef.current && !bridgeRef.current.disposed) {
        bridgeRef.current.momentum = m;
      }
      setConfig((prev) => ({ ...prev, momentum: m }));
    },
    []
  );

  // Persistence: Save to LocalStorage
  const saveToLocalStorage = useCallback(() => {
    const activeBridge = bridgeRef.current;
    if (!activeBridge || activeBridge.disposed) return;
    const model = activeBridge.exportModel(configRef.current.pixelResolution);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(model));
  }, []);

  // Named Presets Collection
  const [namedPresets, setNamedPresets] = useState<SavedPreset[]>(() => {
    try {
      const raw = localStorage.getItem(NAMED_PRESETS_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  });

  const saveNamedPreset = useCallback(
    (name: string): SavedPreset | null => {
      const activeBridge = bridgeRef.current;
      if (!activeBridge || activeBridge.disposed) return null;

      const currentCfg = configRef.current;
      const exported = activeBridge.exportModel(currentCfg.pixelResolution);

      const newPreset: SavedPreset = {
        id: `preset_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
        name: name.trim() || `Net ${currentCfg.pixelResolution}x${currentCfg.pixelResolution}`,
        timestamp: Date.now(),
        config: { ...currentCfg },
        model: exported,
        samplesTrained: trainingStats.samplesTrained,
        currentLoss: trainingStats.currentLoss,
      };

      setNamedPresets((prev) => {
        const updated = [newPreset, ...prev.filter((p) => p.name !== newPreset.name)];
        localStorage.setItem(NAMED_PRESETS_KEY, JSON.stringify(updated));
        return updated;
      });

      return newPreset;
    },
    [trainingStats]
  );

  const loadNamedPreset = useCallback(
    (id: string): boolean => {
      const preset = namedPresets.find((p) => p.id === id);
      if (!preset) return false;

      rebuildNetwork(preset.config);

      if (preset.model) {
        setTimeout(() => {
          if (bridgeRef.current && !bridgeRef.current.disposed) {
            bridgeRef.current.importModel(preset.model);
            if (lastPixelsRef.current) predict(lastPixelsRef.current, false);
          }
        }, 50);
      }

      setTrainingStats({
        samplesTrained: preset.samplesTrained || 0,
        currentLoss: preset.currentLoss || 0,
        lastLossChange: 0,
        accuracy: 0,
        history: [],
      });

      return true;
    },
    [namedPresets, rebuildNetwork, predict]
  );

  const deleteNamedPreset = useCallback((id: string) => {
    setNamedPresets((prev) => {
      const updated = prev.filter((p) => p.id !== id);
      localStorage.setItem(NAMED_PRESETS_KEY, JSON.stringify(updated));
      return updated;
    });
  }, []);

  const setBiasEnabled = useCallback((enabled: boolean) => {
    if (bridgeRef.current && !bridgeRef.current.disposed) {
      bridgeRef.current.useBias = enabled;
    }
    setConfig((prev) => ({ ...prev, useBias: enabled }));
  }, []);

  // Persistence: Load from LocalStorage
  const loadFromLocalStorage = useCallback((): boolean => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;
    try {
      const parsed: SerializedModel = JSON.parse(raw);
      if (parsed.topology && parsed.topology.length >= 2) {
        const newHidden = parsed.topology.slice(1, -1).map((n, i) => ({
          id: String(i + 1),
          neurons: n,
        }));
        const newPixelRes = parsed.pixelResolution || Math.round(Math.sqrt(parsed.topology[0]));

        const newConfig: NetworkConfig = {
          ...configRef.current,
          pixelResolution: newPixelRes,
          inputSize: parsed.topology[0],
          outputSize: parsed.topology[parsed.topology.length - 1],
          hiddenLayers: newHidden,
          learningRate: parsed.learningRate || configRef.current.learningRate,
          momentum: parsed.momentum ?? configRef.current.momentum,
        };

        rebuildNetwork(newConfig);

        setTimeout(() => {
          if (bridgeRef.current && !bridgeRef.current.disposed) {
            bridgeRef.current.importModel(parsed);
            if (lastPixelsRef.current) predict(lastPixelsRef.current, false);
          }
        }, 50);
        return true;
      }
      if (bridgeRef.current && !bridgeRef.current.disposed) {
        return bridgeRef.current.importModel(parsed);
      }
      return false;
    } catch (e) {
      console.error(e);
      return false;
    }
  }, [rebuildNetwork, predict]);

  // Persistence: Export as JSON file download
  const exportModel = useCallback(() => {
    const activeBridge = bridgeRef.current;
    if (!activeBridge || activeBridge.disposed) return;
    const model = activeBridge.exportModel(configRef.current.pixelResolution);
    const jsonStr = JSON.stringify(model, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neural_network_${configRef.current.pixelResolution}x${configRef.current.pixelResolution}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  // Persistence: Import JSON object
  const importModel = useCallback(
    (model: SerializedModel): boolean => {
      if (model.topology && model.topology.length >= 2) {
        const newHidden = model.topology.slice(1, -1).map((n, i) => ({
          id: String(i + 1),
          neurons: n,
        }));
        const newPixelRes = model.pixelResolution || Math.round(Math.sqrt(model.topology[0]));

        const newConfig: NetworkConfig = {
          ...configRef.current,
          pixelResolution: newPixelRes,
          inputSize: model.topology[0],
          outputSize: model.topology[model.topology.length - 1],
          hiddenLayers: newHidden,
          learningRate: model.learningRate || configRef.current.learningRate,
          momentum: model.momentum ?? configRef.current.momentum,
        };

        rebuildNetwork(newConfig);

        setTimeout(() => {
          if (bridgeRef.current && !bridgeRef.current.disposed) {
            bridgeRef.current.importModel(model);
            if (lastPixelsRef.current) predict(lastPixelsRef.current, false);
          }
        }, 50);
        return true;
      }
      if (bridgeRef.current && !bridgeRef.current.disposed) {
        return bridgeRef.current.importModel(model);
      }
      return false;
    },
    [rebuildNetwork, predict]
  );

  // Persistence: Load Pretrained model (96% accuracy on MNIST)
  const loadPretrained = useCallback(async (): Promise<boolean> => {
    try {
      const res = await fetch('/model.json');
      if (!res.ok) throw new Error('Could not fetch /model.json');
      const data = await res.json();

      const pretrainedConfig: NetworkConfig = {
        pixelResolution: 28,
        inputSize: 784,
        outputSize: 10,
        hiddenLayers: [
          { id: '1', neurons: 64 },
          { id: '2', neurons: 64 },
        ],
        hiddenActivation: 'sigmoid',
        outputActivation: 'sigmoid',
        learningRate: 0.1,
        momentum: 0.5,
        useBias: true,
      };

      rebuildNetwork(pretrainedConfig);

      setTimeout(() => {
        if (bridgeRef.current && !bridgeRef.current.disposed) {
          bridgeRef.current.importModel(data);
          setTrainingStats({
            samplesTrained: 60000,
            currentLoss: 0.042,
            lastLossChange: -0.002,
            accuracy: 0.96,
            history: [{ loss: 0.042, accuracy: 0.96 }],
          });
          if (lastPixelsRef.current) predict(lastPixelsRef.current, false);
        }
      }, 50);

      return true;
    } catch (err) {
      console.warn('Pre-trained load warning:', err);
      return false;
    }
  }, [rebuildNetwork, predict]);

  // Reset Weights to Random
  const resetWeights = useCallback(() => {
    rebuildNetwork();
  }, [rebuildNetwork]);

  return {
    bridge,
    config,
    prediction,
    isAnimating,
    animationDuration,
    trainingStats,
    setConfig,
    predict,
    train,
    rebuildNetwork,
    updateLearningRate,
    updateMomentum,
    saveToLocalStorage,
    loadFromLocalStorage,
    exportModel,
    importModel,
    loadPretrained,
    resetWeights,
    namedPresets,
    saveNamedPreset,
    loadNamedPreset,
    deleteNamedPreset,
    setBiasEnabled,
    resetStats: () =>
      setTrainingStats({
        samplesTrained: 0,
        currentLoss: 0,
        lastLossChange: 0,
        accuracy: 0,
        history: [],
      }),
  };
}
