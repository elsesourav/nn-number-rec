import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Sliders, Database, X, Check } from 'lucide-react';
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
    resetStats,
    namedPresets,
    saveNamedPreset,
    loadNamedPreset,
    deleteNamedPreset,
    setBiasEnabled,
  } = useNeuralNetwork(wasmModule);

  // Drawing Canvas Ref for programmatic digit drawing (Live Training)
  const drawingRef = useRef<DrawingCanvasHandle | null>(null);

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
  const [trainSpeed, setTrainSpeed] = useState<number>(1); // Defaults to 1x for complete animation demo
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
        if (next >= targetSamples) {
          setIsLiveTraining(false);
          showToast(`Dataset complete: ${next}/${targetSamples} samples trained!`);
        }
        return next;
      });
    },
    [train, currentPixels, targetSamples]
  );

  // Single Step Live Training (uses real online MNIST image one by one!)
  const runTrainingStep = useCallback(
    (animDuration = 700) => {
      if (!bridge || !drawingRef.current) return;
      if (targetSamples !== Infinity && batchTrained >= targetSamples) {
        setIsLiveTraining(false);
        return;
      }

      let pixels: Float32Array;
      let label: number;

      if (datasetSource === 'online' && onlineSamples.length > 0) {
        // Take real MNIST sample from online dataset one by one
        const idx = mnistIndexRef.current;
        mnistIndexRef.current = (idx + 1) % onlineSamples.length;

        const sample = onlineSamples[idx];
        label = sample.label;
        setCurrentSampleInfo({ id: idx, label });

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
    [bridge, train, targetSamples, batchTrained, datasetSource, onlineSamples]
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
        setBatchTrained(0); // Reset progress on new batch run
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

    // Determine interval and animation mode based on trainSpeed
    let delayMs: number;
    let animDuration: number;

    if (trainSpeed === 1) {
      // Speed 1: Full complete train animation from input to output, then pause before next sample
      animDuration = 750;
      delayMs = 1050; // 750ms full wave from input to output + 300ms inspection pause
    } else if (trainSpeed === 2) {
      animDuration = 450;
      delayMs = 600;
    } else if (trainSpeed === 3) {
      animDuration = 300;
      delayMs = 400;
    } else if (trainSpeed <= 6) {
      animDuration = 160;
      delayMs = Math.round(750 / trainSpeed);
    } else {
      // Fast mode (trainSpeed >= 7):
      // "in between not show any animation just versy fast move"
      animDuration = 0; // No multi-frame pulse wave! Direct rapid state update!
      delayMs = Math.max(12, Math.round(250 / trainSpeed));
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

  // Global Keyboard Shortcuts (B=Brush, E=Eraser, C=Clear, 0-9=Train, Space=Live Train)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing inside text fields
      if (e.target instanceof HTMLElement) {
        const tagName = e.target.tagName.toLowerCase();
        const isTextInput =
          (tagName === 'input' && (e.target as HTMLInputElement).type !== 'range') ||
          tagName === 'textarea' ||
          e.target.isContentEditable;
        if (isTextInput) return;
      }

      // Do not trigger on browser hotkeys (Cmd+C, Ctrl+C, etc.)
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      // Spacebar: toggle live training
      if (e.code === 'Space') {
        e.preventDefault();
        setIsLiveTraining((prev) => !prev);
        return;
      }

      // Brush (B), Eraser (E), Clear (C), Auto-Fit (F)
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

      // Digits 0-9 (top row or numpad)
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
  }, [handleTrainDigit]);

  // Calculate total parameters for floating badge
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
    <div className="h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 flex flex-col md:flex-row p-2 sm:p-2.5 gap-2 sm:gap-2.5 selection:bg-cyan-500/30 selection:text-cyan-300">
      {/* Toast Notification (Floating Top Center) */}
      {toastMsg && (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 px-3 py-1.5 rounded-full bg-slate-900/90 backdrop-blur-md border border-emerald-500/50 shadow-2xl flex items-center gap-1.5 text-xs text-emerald-300 font-medium animate-in fade-in slide-in-from-top-2 duration-150">
          <Check size={12} className="text-emerald-400" />
          <span>{toastMsg}</span>
        </div>
      )}

      {/* LEFT PANEL: Compact Drawing Pad + Big Numpad with Live Training */}
      <div className="w-full md:w-[300px] lg:w-[320px] flex-shrink-0 flex flex-col gap-2 h-auto md:h-full overflow-y-auto pr-0.5">
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
          disabled={!bridge}
        />
      </div>

      {/* RIGHT PANEL: Wide Full-Screen Live Neural Network Visualizer */}
      <div className="flex-1 h-full min-w-0 min-h-[320px] relative flex flex-col rounded-2xl border border-slate-800/80 bg-slate-950/80 overflow-hidden shadow-2xl">
        {/* Floating Top Control Dock (Anchored Top-Right over the Visualizer) */}
        <div className="absolute top-2.5 right-2.5 z-30 flex items-center gap-1.5 p-1 rounded-xl bg-slate-900/85 backdrop-blur-md border border-slate-700/60 shadow-xl text-xs">
          {/* Architecture Drawer Trigger */}
          <button
            type="button"
            data-testid="btn-architecture"
            onClick={() => setDrawerTab((prev) => (prev === 'architecture' ? null : 'architecture'))}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg font-medium transition-all cursor-pointer ${
              drawerTab === 'architecture'
                ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-500/50 shadow-sm'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
            }`}
            title="Configure Input Pixels and Hidden Layers"
          >
            <Sliders size={13} className="text-cyan-400 flex-shrink-0" />
            <span className="hidden sm:inline">Architecture</span>
            <span className="text-[10px] text-cyan-300 font-mono font-semibold px-1.5 py-0.2 rounded bg-cyan-950 border border-cyan-800/50">
              {calculateTotalParams().toLocaleString()}
            </span>
          </button>

          {/* Storage Drawer Trigger */}
          <button
            type="button"
            data-testid="btn-storage"
            onClick={() => setDrawerTab((prev) => (prev === 'storage' ? null : 'storage'))}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg font-medium transition-all cursor-pointer ${
              drawerTab === 'storage'
                ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-500/50 shadow-sm'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
            }`}
            title="Model Storage & JSON Export/Import"
          >
            <Database size={13} className="text-emerald-400 flex-shrink-0" />
            <span className="hidden sm:inline">Storage</span>
          </button>

          {/* Engine Status Indicator */}
          <div className="flex items-center gap-1.5 px-2 py-1 text-[10px] font-mono text-emerald-400 border-l border-slate-800 pl-2">
            {wasmLoading ? (
              <span className="text-amber-400 animate-pulse">Loading...</span>
            ) : wasmError ? (
              <span className="text-rose-400">Error</span>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                <span>SIMD128</span>
              </>
            )}
          </div>
        </div>

        {/* Floating Top-Left Graph Legend */}
        <div className="hidden md:flex absolute top-3 left-3 z-20 pointer-events-none items-center gap-3 text-[10px] font-mono text-slate-400 bg-slate-900/60 backdrop-blur-sm px-2.5 py-1 rounded-lg border border-slate-800/50">
          <span className="text-slate-300 font-semibold font-sans">Live Flow</span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-0.5 bg-emerald-400 rounded"></span> +Weight
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-0.5 bg-purple-400 rounded"></span> -Weight
          </span>
        </div>

        {/* The Wide Canvas Visualizer (Expands to Full Width & Height!) */}
        <div className="flex-1 w-full h-full relative">
          <NetworkVisualizer
            bridge={bridge}
            animating={isAnimating}
            animationDuration={animationDuration}
            activeDigit={prediction?.digit ?? null}
          />
        </div>

        {/* Floating Bottom Bar: Live Topology, Hyperparameters & Single Loss Source */}
        <div className="absolute bottom-1.5 left-3 right-3 z-20 pointer-events-none flex items-center justify-between text-[10px] font-mono text-slate-500 bg-slate-950/70 backdrop-blur-xs px-2.5 py-0.5 rounded border border-slate-900/60 shadow-sm">
          <span className="truncate mr-2">
            Topology: [{config.inputSize}, {config.hiddenLayers.map((l) => l.neurons).join(', ')}, {config.outputSize}]
          </span>
          <span className="flex-shrink-0">
            {config.hiddenActivation} • LR: {config.learningRate.toFixed(2)} • Mom: {config.momentum.toFixed(2)} • Bias: {config.useBias ? 'ON' : 'OFF'} • Loss:{' '}
            <span className="text-emerald-400 font-semibold">{trainingStats.currentLoss.toFixed(4)}</span>
            <span className="text-slate-500"> ({trainingStats.samplesTrained})</span>
          </span>
        </div>
      </div>

      {/* SLIDE-OVER SIDE DRAWER / FLOATING MODAL WINDOW */}
      {drawerTab !== null && (
        <div className="fixed inset-0 z-50 overflow-hidden flex justify-end">
          {/* Backdrop for closing when clicking anywhere outside */}
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-xs transition-opacity animate-in fade-in duration-150"
            onClick={() => setDrawerTab(null)}
          />

          {/* Sliding Panel */}
          <div className="relative w-full sm:max-w-[390px] h-full bg-slate-900/98 backdrop-blur-2xl border-l border-slate-700/80 shadow-2xl flex flex-col z-10 animate-in slide-in-from-right duration-200">
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
                      if (ok) showToast('Preset loaded successfully!');
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
