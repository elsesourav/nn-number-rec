import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Sliders, Database, X, Check, PenTool, Zap, Sparkles } from 'lucide-react';
import { Button, Badge } from '@elsesourav/react-ui';
import { useWasmModule } from './hooks/useWasmModule';
import { useNeuralNetwork } from './hooks/useNeuralNetwork';
import {
  DrawingCanvas,
  type DrawingCanvasHandle,
} from './components/DrawingCanvas/DrawingCanvas';
import { NetworkConfigurator } from './components/NetworkConfig/NetworkConfigurator';
import { NetworkVisualizer } from './components/Visualization/NetworkVisualizer';
import { TrainingPanel } from './components/Training/TrainingPanel';
import { ModelManager } from './components/ModelManager/ModelManager';
import { loadOnlineMnistDataset, type MnistSample } from './utils/onlineDataset';

export default function App() {
  const { module: wasmModule, loading: wasmLoading, error: wasmError } = useWasmModule();
  const {
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
  } = useNeuralNetwork(wasmModule);

  // Drawing Canvas Ref for programmatic digit drawing (Live Training)
  const drawingRef = useRef<DrawingCanvasHandle | null>(null);

  // Mode Switcher: 'recognize' (pure test/predict) vs 'train' (training studio)
  const [viewMode, setViewMode] = useState<'recognize' | 'train'>('recognize');

  // Floating Side Drawer State ('architecture' | 'storage' | null)
  const [drawerTab, setDrawerTab] = useState<'architecture' | 'storage' | null>(null);
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  const [currentPixels, setCurrentPixels] = useState<Float32Array>(
    new Float32Array(config.inputSize)
  );

  // Online MNIST Dataset State (Real handwritten digits loaded from online)
  const [onlineSamples, setOnlineSamples] = useState<MnistSample[]>([]);
  const [datasetSource, setDatasetSource] = useState<'online' | 'synthetic'>('online');
  const [currentSampleInfo, setCurrentSampleInfo] = useState<{ id: number; label: number } | null>(null);
  const mnistIndexRef = useRef<number>(0);

  // Training Dataset Progress Tracker (Target samples + items remaining)
  const [targetSamples, setTargetSamples] = useState<number>(100);
  const [batchTrained, setBatchTrained] = useState<number>(0);

  // Live Auto-Training Engine State
  const [isLiveTraining, setIsLiveTraining] = useState<boolean>(false);
  const [trainSpeed, setTrainSpeed] = useState<number>(1);
  const currentTrainDigitRef = useRef<number>(0);
  const liveTrainTimerRef = useRef<number | null>(null);

  const showToast = (msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(null), 2500);
  };

  // Fetch online MNIST dataset on initialization
  useEffect(() => {
    loadOnlineMnistDataset().then((samples) => {
      if (samples.length > 0) {
        setOnlineSamples(samples);
        setCurrentSampleInfo({ id: 0, label: samples[0].label });
      }
    });
  }, []);

  const handlePixelDataChange = useCallback(
    (pixels: Float32Array) => {
      setCurrentPixels(pixels);
      predict(pixels, true);
    },
    [predict]
  );

  const handleTrainDigit = useCallback(
    (digit: number) => {
      train(currentPixels, digit);
      setBatchTrained((prev) => {
        const next = prev + 1;
        if (targetSamples !== Infinity && next >= targetSamples) {
          setIsLiveTraining(false);
          showToast(`Target reached: ${targetSamples}/${targetSamples} trained!`);
          return targetSamples;
        }
        return next;
      });
      showToast(`Trained on digit ${digit}`);
    },
    [train, currentPixels, targetSamples]
  );

  // Execute 1 complete step of dataset training
  const runTrainingStep = useCallback(
    (animDuration = 0) => {
      if (!bridge || !drawingRef.current) return;

      let label: number;
      let pixels: Float32Array;

      if (datasetSource === 'online' && onlineSamples.length > 0) {
        const sampleIdx = mnistIndexRef.current % onlineSamples.length;
        mnistIndexRef.current = (mnistIndexRef.current + 1) % onlineSamples.length;
        const sample = onlineSamples[sampleIdx];
        label = sample.label;
        setCurrentSampleInfo({ id: sampleIdx, label });

        // Draw authentic online digit on canvas & downsample to active architecture
        pixels = drawingRef.current.drawMnistSample(sample.pixels);
      } else {
        // Cycle synthetic digit 0 -> 9
        label = currentTrainDigitRef.current;
        currentTrainDigitRef.current = (label + 1) % 10;
        pixels = drawingRef.current.drawSample(label);
      }

      setCurrentPixels(pixels);

      // Train WASM neural network on sample
      train(pixels, label, animDuration);

      setBatchTrained((prev) => {
        const next = prev + 1;
        if (targetSamples !== Infinity && next >= targetSamples) {
          setIsLiveTraining(false);
          showToast(`Dataset complete: ${targetSamples}/${targetSamples} samples trained!`);
          return targetSamples;
        }
        return next;
      });
    },
    [bridge, train, targetSamples, datasetSource, onlineSamples]
  );

  // Preview an online sample on canvas without training
  const showSampleOnCanvas = useCallback(
    (idx: number) => {
      if (!drawingRef.current || onlineSamples.length === 0) return;
      const sample = onlineSamples[idx];
      if (!sample) return;
      setCurrentSampleInfo({ id: idx, label: sample.label });
      const pixels = drawingRef.current.drawMnistSample(sample.pixels);
      setCurrentPixels(pixels);
    },
    [onlineSamples]
  );

  const handleNextSample = useCallback(() => {
    if (onlineSamples.length === 0) return;
    const nextIdx = (mnistIndexRef.current + 1) % onlineSamples.length;
    mnistIndexRef.current = nextIdx;
    showSampleOnCanvas(nextIdx);
  }, [onlineSamples, showSampleOnCanvas]);

  const handlePrevSample = useCallback(() => {
    if (onlineSamples.length === 0) return;
    const prevIdx = (mnistIndexRef.current - 1 + onlineSamples.length) % onlineSamples.length;
    mnistIndexRef.current = prevIdx;
    showSampleOnCanvas(prevIdx);
  }, [onlineSamples, showSampleOnCanvas]);

  const handleToggleDatasetSource = useCallback(() => {
    setDatasetSource((prev) => {
      const next = prev === 'online' ? 'synthetic' : 'online';
      showToast(`Dataset source: ${next === 'online' ? 'Online MNIST (Real Digits)' : 'Synthetic Digits'}`);
      return next;
    });
  }, []);

  const handleToggleLiveTraining = () => {
    setIsLiveTraining((prev) => {
      if (!prev && targetSamples !== Infinity && batchTrained >= targetSamples) {
        setBatchTrained(0);
      }
      return !prev;
    });
  };

  const handleApplyTemplate = (template: Partial<typeof config>) => {
    const res = template.pixelResolution || config.pixelResolution;
    const newCfg = {
      ...config,
      ...template,
      pixelResolution: res,
      inputSize: res * res,
    };
    rebuildNetwork(newCfg);
    showToast('Architecture applied!');
  };

  // Live Auto-Training Animation Loop
  useEffect(() => {
    if (!isLiveTraining) {
      if (liveTrainTimerRef.current) {
        clearTimeout(liveTrainTimerRef.current);
        liveTrainTimerRef.current = null;
      }
      return;
    }

    let delayMs: number;
    let animDuration: number;

    if (trainSpeed === 1) {
      animDuration = 700;
      delayMs = 1000;
    } else if (trainSpeed === 2) {
      animDuration = 400;
      delayMs = 550;
    } else if (trainSpeed === 3) {
      animDuration = 260;
      delayMs = 380;
    } else if (trainSpeed <= 6) {
      animDuration = 140;
      delayMs = Math.round(700 / trainSpeed);
    } else {
      animDuration = 0;
      delayMs = Math.max(12, Math.round(240 / trainSpeed));
    }

    const scheduleNext = () => {
      liveTrainTimerRef.current = window.setTimeout(() => {
        runTrainingStep(animDuration);
        scheduleNext();
      }, delayMs);
    };

    scheduleNext();

    return () => {
      if (liveTrainTimerRef.current) {
        clearTimeout(liveTrainTimerRef.current);
      }
    };
  }, [isLiveTraining, trainSpeed, runTrainingStep]);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement) {
        const tagName = e.target.tagName.toLowerCase();
        const isTextInput =
          (tagName === 'input' && (e.target as HTMLInputElement).type !== 'range') ||
          tagName === 'textarea' ||
          e.target.isContentEditable;
        if (isTextInput) return;
      }

      if (e.ctrlKey || e.metaKey || e.altKey) return;

      if (e.code === 'Space') {
        e.preventDefault();
        if (viewMode === 'train') {
          setIsLiveTraining((prev) => !prev);
        } else {
          drawingRef.current?.clear();
        }
        return;
      }

      const keyLower = e.key.toLowerCase();
      if (e.code === 'KeyB' || keyLower === 'b') {
        e.preventDefault();
        drawingRef.current?.setTool('brush');
        return;
      }
      if (e.code === 'KeyE' || keyLower === 'e') {
        e.preventDefault();
        drawingRef.current?.setTool('eraser');
        return;
      }
      if (e.code === 'KeyC' || keyLower === 'c') {
        e.preventDefault();
        drawingRef.current?.clear();
        return;
      }
      if (e.code === 'KeyF' || keyLower === 'f') {
        e.preventDefault();
        drawingRef.current?.autoFit();
        return;
      }

      let digit: number | null = null;
      if (e.key >= '0' && e.key <= '9') {
        digit = parseInt(e.key, 10);
      } else if (e.code.startsWith('Digit') && e.code.length === 6) {
        const d = parseInt(e.code.replace('Digit', ''), 10);
        if (!isNaN(d) && d >= 0 && d <= 9) digit = d;
      } else if (e.code.startsWith('Numpad') && e.code.length === 7) {
        const d = parseInt(e.code.replace('Numpad', ''), 10);
        if (!isNaN(d) && d >= 0 && d <= 9) digit = d;
      }

      if (digit !== null) {
        e.preventDefault();
        handleTrainDigit(digit);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleTrainDigit, viewMode]);

  const calculateTotalParams = () => {
    let total = 0;
    const fullTopology = [
      config.inputSize,
      ...config.hiddenLayers.map((l) => l.neurons),
      config.outputSize,
    ];
    for (let i = 0; i < fullTopology.length - 1; i++) {
      total += fullTopology[i] * fullTopology[i + 1] + fullTopology[i + 1];
    }
    return total;
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-cyan-500/30 selection:text-cyan-300">
      {/* Toast Notification */}
      {toastMsg && (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 px-3.5 py-1.5 rounded-full bg-slate-900/90 backdrop-blur-md border border-emerald-500/50 shadow-2xl flex items-center gap-1.5 text-xs text-emerald-300 font-medium animate-in fade-in slide-in-from-top-2 duration-150">
          <Check size={12} className="text-emerald-400" />
          <span>{toastMsg}</span>
        </div>
      )}

      {/* Modern Responsive App Header (Clean & not locked to full screen) */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-30 px-3 sm:px-6 py-2.5">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-3">
          {/* Title & Tech Badges */}
          <div className="flex items-center gap-2.5 min-w-0">
            <h1 className="text-sm sm:text-base font-bold text-slate-100 tracking-tight whitespace-nowrap">
              Neural Digit Recognizer
            </h1>
            <Badge variant="primary" size="sm" className="hidden sm:inline-flex text-[10px] font-mono">
              SIMD128 C++ WASM
            </Badge>
          </div>

          {/* Mode Switcher: Recognize vs Train */}
          <div className="flex items-center bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-xs shadow-inner">
            <button
              type="button"
              data-testid="tab-recognize"
              onClick={() => setViewMode('recognize')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md font-semibold transition-all cursor-pointer ${
                viewMode === 'recognize'
                  ? 'bg-cyan-500/20 text-cyan-300 shadow-xs'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Recognize drawn digits in real-time"
            >
              <PenTool size={12} />
              <span>Recognize</span>
            </button>
            <button
              type="button"
              data-testid="tab-train"
              onClick={() => setViewMode('train')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md font-semibold transition-all cursor-pointer ${
                viewMode === 'train'
                  ? 'bg-cyan-500/20 text-cyan-300 shadow-xs'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Train neural network with dataset"
            >
              <Zap size={12} />
              <span>Train</span>
            </button>
          </div>

          {/* Right Drawer Triggers */}
          <div className="flex items-center gap-1.5">
            <Button
              variant="outline"
              size="sm"
              data-testid="btn-architecture"
              onClick={() => setDrawerTab((prev) => (prev === 'architecture' ? null : 'architecture'))}
              className="text-xs h-7.5 px-2.5 text-slate-300 hover:text-white border-slate-800 bg-slate-900 cursor-pointer"
              title="Configure Network Architecture"
            >
              <Sliders size={12} className="mr-1 text-cyan-400" />
              <span className="hidden sm:inline">Architecture</span>
              <span className="text-[10px] text-cyan-300 font-mono ml-1 px-1 py-0.2 rounded bg-cyan-950 border border-cyan-800/40">
                {calculateTotalParams().toLocaleString()}
              </span>
            </Button>

            <Button
              variant="outline"
              size="sm"
              data-testid="btn-storage"
              onClick={() => setDrawerTab((prev) => (prev === 'storage' ? null : 'storage'))}
              className="text-xs h-7.5 px-2.5 text-slate-300 hover:text-white border-slate-800 bg-slate-900 cursor-pointer"
              title="Model Storage & Presets"
            >
              <Database size={12} className="mr-1 text-emerald-400" />
              <span className="hidden sm:inline">Storage</span>
            </Button>
          </div>
        </div>
      </header>

      {/* Main Responsive Grid Layout (Natural Mobile Scrolling, Network on TOP on mobile) */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-3 sm:p-5 flex flex-col gap-4">
        <div className="flex flex-col md:flex-row gap-4 flex-1">
          {/* LEFT/BOTTOM COLUMN: Controls & Drawing Section (Order-2 on mobile, Order-1 on desktop) */}
          <div className="order-2 md:order-1 w-full md:w-[320px] lg:w-[340px] flex-shrink-0 flex flex-col gap-3">
            {viewMode === 'recognize' ? (
              <>
                {/* 1. Pure Distraction-Free Drawing Pad */}
                <DrawingCanvas
                  ref={drawingRef}
                  pixelResolution={config.pixelResolution}
                  onPixelDataChange={handlePixelDataChange}
                  onClear={() => {
                    setCurrentPixels(new Float32Array(config.inputSize));
                    predict(new Float32Array(config.inputSize), false);
                  }}
                  disabled={!bridge}
                />

                {/* 2. Interactive Numpad with Live Probabilities */}
                <TrainingPanel
                  prediction={prediction}
                  onTrainDigit={handleTrainDigit}
                  isLiveTraining={false}
                  onToggleLiveTraining={() => {}}
                  onStepTraining={() => {}}
                  trainSpeed={trainSpeed}
                  onChangeSpeed={setTrainSpeed}
                  showTrainingControls={false}
                  showNumpad={true}
                  disabled={!bridge}
                />

                {/* 3. Clean Prediction Result Card */}
                {prediction && prediction.digit !== null && (
                  <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800/80 flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <Sparkles size={14} className="text-emerald-400" />
                      <span className="text-slate-300 font-medium">Predicted:</span>
                      <span className="text-base font-bold text-emerald-400 font-mono">
                        Digit {prediction.digit}
                      </span>
                    </div>
                    <span className="text-cyan-300 font-mono text-[11px] font-semibold bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-800/50">
                      {Math.round((prediction.probabilities[prediction.digit] ?? 0) * 100)}% Confidence
                    </span>
                  </div>
                )}
              </>
            ) : (
              <>
                {/* 1. Training Studio Controls */}
                <TrainingPanel
                  prediction={prediction}
                  onTrainDigit={handleTrainDigit}
                  isLiveTraining={isLiveTraining}
                  onToggleLiveTraining={handleToggleLiveTraining}
                  onStepTraining={runTrainingStep}
                  trainSpeed={trainSpeed}
                  onChangeSpeed={setTrainSpeed}
                  batchTrained={batchTrained}
                  targetSamples={targetSamples}
                  onChangeTargetSamples={setTargetSamples}
                  datasetSource={datasetSource}
                  onToggleDatasetSource={handleToggleDatasetSource}
                  totalOnlineSamples={onlineSamples.length}
                  currentSampleInfo={currentSampleInfo}
                  onPrevSample={handlePrevSample}
                  onNextSample={handleNextSample}
                  showTrainingControls={true}
                  showNumpad={false}
                  disabled={!bridge}
                />

                {/* 2. Active Training Sample Preview on Canvas */}
                <DrawingCanvas
                  ref={drawingRef}
                  pixelResolution={config.pixelResolution}
                  onPixelDataChange={handlePixelDataChange}
                  onClear={() => {
                    setCurrentPixels(new Float32Array(config.inputSize));
                    predict(new Float32Array(config.inputSize), false);
                  }}
                  disabled={!bridge}
                />

                {/* 3. Numpad Probabilities during Training */}
                <TrainingPanel
                  prediction={prediction}
                  onTrainDigit={handleTrainDigit}
                  isLiveTraining={isLiveTraining}
                  onToggleLiveTraining={() => {}}
                  onStepTraining={() => {}}
                  trainSpeed={trainSpeed}
                  onChangeSpeed={setTrainSpeed}
                  showTrainingControls={false}
                  showNumpad={true}
                  disabled={!bridge}
                />
              </>
            )}
          </div>

          {/* RIGHT/TOP COLUMN: Neural Network Visualizer (Order-1 on mobile, Order-2 on desktop) */}
          <div className="order-1 md:order-2 flex-1 min-w-0 h-[320px] sm:h-[380px] md:h-auto md:min-h-[500px] relative flex flex-col rounded-2xl border border-slate-800/80 bg-slate-950/80 overflow-hidden shadow-xl">
            {/* Legend & Mode Status */}
            <div className="absolute top-2.5 left-3 z-20 pointer-events-none flex items-center gap-3 text-[10px] font-mono text-slate-400 bg-slate-900/70 backdrop-blur-xs px-2.5 py-1 rounded-lg border border-slate-800/60">
              <span className="text-slate-300 font-semibold font-sans">Active Flow</span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-0.5 bg-emerald-400 rounded"></span> +Weight
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-0.5 bg-purple-400 rounded"></span> -Weight
              </span>
            </div>

            {/* Neural Network Canvas */}
            <div className="flex-1 w-full h-full relative">
              <NetworkVisualizer
                bridge={bridge}
                animating={isAnimating}
                animationDuration={animationDuration}
                activeDigit={prediction?.digit ?? null}
              />
            </div>

            {/* Bottom Status Bar */}
            <div className="absolute bottom-1.5 left-3 right-3 z-20 pointer-events-none flex items-center justify-between text-[10px] font-mono text-slate-500 bg-slate-950/80 backdrop-blur-xs px-2.5 py-0.5 rounded border border-slate-900 shadow-sm">
              <span className="truncate mr-2">
                Topology: [{config.inputSize}, {config.hiddenLayers.map((l) => l.neurons).join(', ')}, {config.outputSize}]
              </span>
              <span className="flex-shrink-0">
                {config.hiddenActivation} • LR: {config.learningRate.toFixed(2)} • Mom: {config.momentum.toFixed(2)} • Loss:{' '}
                <span className="text-emerald-400 font-semibold">{trainingStats.currentLoss.toFixed(4)}</span>
                <span className="text-slate-500"> ({trainingStats.samplesTrained})</span>
              </span>
            </div>
          </div>
        </div>
      </main>

      {/* SLIDE-OVER SIDE DRAWER: Opens from the LEFT with reduced blur */}
      {drawerTab !== null && (
        <div className="fixed inset-0 z-50 overflow-hidden flex justify-start">
          {/* Subtle Backdrop with less blur */}
          <div
            className="fixed inset-0 bg-black/40 backdrop-blur-xs transition-opacity animate-in fade-in duration-150"
            onClick={() => setDrawerTab(null)}
          />

          {/* Sliding Panel from the Left */}
          <div className="relative w-full sm:max-w-[400px] h-full bg-slate-900 border-r border-slate-700/80 shadow-2xl flex flex-col z-10 animate-in slide-in-from-left duration-200">
            {/* Drawer Top Navigation & Close Button */}
            <div className="flex items-center justify-between p-2.5 border-b border-slate-800 bg-slate-950/80">
              <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800 text-xs">
                <button
                  type="button"
                  data-testid="drawer-tab-architecture"
                  onClick={() => setDrawerTab('architecture')}
                  className={`flex items-center gap-1.5 py-1 px-2.5 rounded-md font-medium transition-all cursor-pointer ${
                    drawerTab === 'architecture'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Sliders size={12} />
                  Architecture
                </button>
                <button
                  type="button"
                  data-testid="drawer-tab-storage"
                  onClick={() => setDrawerTab('storage')}
                  className={`flex items-center gap-1.5 py-1 px-2.5 rounded-md font-medium transition-all cursor-pointer ${
                    drawerTab === 'storage'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Database size={12} />
                  Storage
                </button>
              </div>

              <button
                type="button"
                data-testid="btn-close-drawer"
                onClick={() => setDrawerTab(null)}
                className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-all cursor-pointer"
                title="Close Drawer"
              >
                <X size={16} />
              </button>
            </div>

            {/* Drawer Scrollable Content */}
            <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
              {drawerTab === 'architecture' && (
                <NetworkConfigurator
                  config={config}
                  onConfigChange={setConfig}
                  onRebuildNetwork={() => {
                    rebuildNetwork();
                    showToast('Architecture applied & network rebuilt!');
                  }}
                  onLearningRateChange={updateLearningRate}
                  onMomentumChange={updateMomentum}
                  onSaveToLocalStorage={() => {
                    saveToLocalStorage();
                    showToast('Model saved to storage!');
                  }}
                  onLoadFromLocalStorage={() => {
                    const ok = loadFromLocalStorage();
                    showToast(ok ? 'Model loaded!' : 'No saved model found');
                    return ok;
                  }}
                  onLoadPretrained={async () => {
                    const ok = await loadPretrained();
                    showToast(ok ? 'Pretrained model loaded!' : 'Failed to load pretrained');
                    return ok;
                  }}
                  onResetWeights={() => {
                    resetWeights();
                    showToast('Weights randomized!');
                  }}
                  disabled={!wasmModule}
                />
              )}

              {drawerTab === 'storage' && (
                <div className="flex flex-col gap-3">
                  <ModelManager
                    onSaveToLocalStorage={() => {
                      saveToLocalStorage();
                      showToast('Model saved to storage!');
                    }}
                    onLoadFromLocalStorage={() => {
                      const ok = loadFromLocalStorage();
                      showToast(ok ? 'Loaded from storage!' : 'No saved model');
                      return ok;
                    }}
                    onExportModel={exportModel}
                    onImportModel={(m) => {
                      const ok = importModel(m);
                      showToast(ok ? 'Model imported successfully!' : 'Import failed');
                      return ok;
                    }}
                    onLoadPretrained={async () => {
                      const ok = await loadPretrained();
                      showToast(ok ? 'Pretrained loaded!' : 'Load failed');
                      return ok;
                    }}
                    onResetWeights={() => {
                      resetWeights();
                      showToast('Weights randomized!');
                    }}
                    namedPresets={namedPresets}
                    onSaveNamedPreset={(name) => {
                      const p = saveNamedPreset(name);
                      if (p) showToast(`Preset "${p.name}" saved!`);
                      return p;
                    }}
                    onLoadNamedPreset={(id) => {
                      const ok = loadNamedPreset(id);
                      showToast('Preset loaded successfully!');
                      return ok;
                    }}
                    onDeleteNamedPreset={(id) => {
                      deleteNamedPreset(id);
                      showToast('Preset deleted.');
                    }}
                    onApplyTemplate={handleApplyTemplate}
                    disabled={!bridge}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
