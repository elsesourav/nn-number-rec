import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Sliders, Database, X, PenTool, Zap, Sparkles, Network, Sigma } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button, Badge, Dialog, useToast } from '@elsesourav/react-ui';
import { useWasmModule } from './hooks/useWasmModule';
import { useNeuralNetwork } from './hooks/useNeuralNetwork';
import {
  DrawingCanvas,
  type DrawingCanvasHandle,
} from './components/DrawingCanvas/DrawingCanvas';
import { NetworkConfigurator } from './components/NetworkConfig/NetworkConfigurator';
import { NetworkVisualizer } from './components/Visualization/NetworkVisualizer';
import { MechanismVisualizer } from './components/Visualization/MechanismVisualizer';
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

  // Visualizer View Mode: 'graph' (interactive node & synapse canvas) vs 'mechanism' (step-by-step mathematical & pipeline walkthrough)
  const [visualizerMode, setVisualizerMode] = useState<'graph' | 'mechanism'>('graph');

  // Floating Side Drawer State ('architecture' | 'storage' | null)
  const [drawerTab, setDrawerTab] = useState<'architecture' | 'storage' | null>(null);
  const [activeDrawerTab, setActiveDrawerTab] = useState<'architecture' | 'storage'>('architecture');
  const [isResetConfirmOpen, setIsResetConfirmOpen] = useState<boolean>(false);
  const { addToast } = useToast();

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

  const showToast = useCallback(
    (msg: string, type: 'info' | 'success' | 'warning' | 'error' = 'success') => {
      addToast({
        title: msg,
        type,
        duration: 2500,
      });
    },
    [addToast]
  );

  // Fetch online MNIST dataset on initialization
  useEffect(() => {
    loadOnlineMnistDataset().then((samples) => {
      if (samples.length > 0) {
        setOnlineSamples(samples);
        setCurrentSampleInfo({ id: 0, label: samples[0].label });
      }
    });
  }, []);

  // Automatically load pre-trained MNIST model by default on initial mount
  const hasAutoLoadedPretrained = useRef(false);
  useEffect(() => {
    if (bridge && !hasAutoLoadedPretrained.current) {
      hasAutoLoadedPretrained.current = true;
      loadPretrained().then((success) => {
        if (success) {
          showToast('Pre-trained MNIST model loaded (96% accuracy)', 'info');
        }
      });
    }
  }, [bridge, loadPretrained, showToast]);

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

  // Train & display a specific online sample on canvas with synaptic animation
  const trainAndDisplaySample = useCallback(
    (idx: number, animDuration = 400) => {
      if (!drawingRef.current || onlineSamples.length === 0) return;
      const sample = onlineSamples[idx];
      if (!sample) return;

      setCurrentSampleInfo({ id: idx, label: sample.label });
      const pixels = drawingRef.current.drawMnistSample(sample.pixels);
      setCurrentPixels(pixels);

      // Actively train WASM neural network on this sample!
      train(pixels, sample.label, animDuration);
    },
    [onlineSamples, train]
  );

  const handleNextSample = useCallback(() => {
    if (onlineSamples.length === 0) return;
    const nextIdx = (mnistIndexRef.current + 1) % onlineSamples.length;
    mnistIndexRef.current = nextIdx;
    trainAndDisplaySample(nextIdx, 400);
    setBatchTrained((prev) => (targetSamples !== Infinity ? Math.min(targetSamples, prev + 1) : prev + 1));
    const sample = onlineSamples[nextIdx];
    showToast(`Trained sample #${nextIdx + 1} (Digit ${sample?.label})`, 'success');
  }, [onlineSamples, trainAndDisplaySample, targetSamples, showToast]);

  const handlePrevSample = useCallback(() => {
    if (onlineSamples.length === 0) return;
    if (mnistIndexRef.current <= 0) {
      showToast('Already at the first sample (#1)', 'info');
      return;
    }
    const prevIdx = mnistIndexRef.current - 1;
    mnistIndexRef.current = prevIdx;
    trainAndDisplaySample(prevIdx, 400);
    setBatchTrained((prev) => Math.max(0, prev - 1));
    const sample = onlineSamples[prevIdx];
    showToast(`Trained sample #${prevIdx + 1} (Digit ${sample?.label})`, 'info');
  }, [onlineSamples, trainAndDisplaySample, showToast]);

  const handleResetToFirst = useCallback(() => {
    if (onlineSamples.length === 0) return;
    mnistIndexRef.current = 0;
    trainAndDisplaySample(0, 400);
    setBatchTrained(0);
    showToast('Reset to first sample (#1)', 'info');
  }, [onlineSamples, trainAndDisplaySample, showToast]);

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
      {/* Modern Responsive App Header (Clean & not locked to full screen) */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-30 px-2 sm:px-6 py-1.5 sm:py-2.5">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-1.5 sm:gap-3">
          {/* Title, Logo & Tech Badges */}
          <div className="flex items-center gap-1.5 sm:gap-2.5 min-w-0 shrink-0">
            <img
              src="/logo.png"
              alt="Neural Digit Recognizer"
              className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg object-contain bg-slate-950/80 border border-cyan-500/40 p-0.5 shadow-sm shadow-cyan-500/20 shrink-0 select-none"
            />
            <div className="hidden sm:flex items-center gap-2 min-w-0">
              <h1 className="text-sm sm:text-base font-bold text-slate-100 tracking-tight whitespace-nowrap">
                Neural Digit Recognizer
              </h1>
              <Badge variant="primary" size="sm" className="hidden lg:inline-flex text-[10px] font-mono">
                SIMD128 C++ WASM
              </Badge>
            </div>
          </div>

          {/* Mode Switcher: Recognize vs Train with Smooth Sliding Pill */}
          <div className="relative flex items-center bg-slate-950/90 p-0.5 rounded-lg border border-slate-800 shadow-inner">
            <button
              type="button"
              data-testid="tab-recognize"
              onClick={() => setViewMode('recognize')}
              className={`relative z-10 flex items-center gap-1 px-2.5 sm:px-3 py-1 text-xs font-semibold rounded-md transition-colors cursor-pointer select-none ${
                viewMode === 'recognize' ? 'text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Recognize drawn digits in real-time"
            >
              <PenTool size={11} className="shrink-0" />
              <span>Recognize</span>
              {viewMode === 'recognize' && (
                <motion.div
                  layoutId="header-mode-pill"
                  className="absolute inset-0 bg-cyan-400 rounded-md shadow-xs -z-10"
                  transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                />
              )}
            </button>

            <button
              type="button"
              data-testid="tab-train"
              onClick={() => setViewMode('train')}
              className={`relative z-10 flex items-center gap-1 px-2.5 sm:px-3 py-1 text-xs font-semibold rounded-md transition-colors cursor-pointer select-none ${
                viewMode === 'train' ? 'text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Train neural network with dataset"
            >
              <Zap size={11} className="shrink-0" />
              <span>Train</span>
              {viewMode === 'train' && (
                <motion.div
                  layoutId="header-mode-pill"
                  className="absolute inset-0 bg-amber-400 rounded-md shadow-xs -z-10"
                  transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                />
              )}
            </button>
          </div>

          {/* Right Header Trigger: Single Unified Button for Architecture & Storage */}
          <div className="flex items-center gap-1.5 shrink-0">
            <Button
              variant="outline"
              size="sm"
              data-testid="btn-architecture"
              onClick={() => setDrawerTab((prev) => (prev ? null : activeDrawerTab))}
              leftIcon={<Sliders size={12} className="text-cyan-400 shrink-0" />}
              className="inline-flex flex-row items-center gap-1 sm:gap-1.5 whitespace-nowrap text-xs h-7 sm:h-8 px-2 sm:px-3 text-slate-200 hover:text-white border-slate-800 bg-slate-900/90 hover:bg-slate-800 cursor-pointer [&>span]:inline-flex [&>span]:flex-row [&>span]:items-center [&>span]:gap-1"
              title="Architecture & Model Storage"
            >
              <span className="hidden sm:inline">Model</span>
              <span className="text-[10px] text-cyan-300 font-mono px-1 sm:px-1.5 py-0.5 rounded bg-cyan-950/80 border border-cyan-800/50">
                {calculateTotalParams() >= 1000 ? `${Math.round(calculateTotalParams() / 1000)}k` : calculateTotalParams()}
              </span>
            </Button>
          </div>
        </div>
      </header>

      {/* Main Responsive Grid Layout (Natural Mobile Scrolling, Network on TOP on mobile) */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-3 sm:p-5 flex flex-col gap-4">
        <div className="flex flex-col md:flex-row gap-4 flex-1">
          {/* LEFT/BOTTOM COLUMN: Controls & Drawing Section (Order-2 on mobile, Order-1 on desktop) */}
          <div className="order-2 md:order-1 w-full md:w-[320px] lg:w-[340px] flex-shrink-0 flex flex-col gap-3">
            <AnimatePresence mode="wait">
              {viewMode === 'recognize' ? (
                <motion.div
                  key="panel-recognize"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.22, ease: 'easeOut' }}
                  className="flex flex-col gap-3 w-full"
                >
                  {/* 1. Pure Distraction-Free Drawing Pad */}
                  <DrawingCanvas
                    ref={drawingRef}
                    pixelResolution={config.pixelResolution}
                    autoFitOnDraw={config.autoFitOnDraw !== false}
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
                </motion.div>
              ) : (
                <motion.div
                  key="panel-train"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.22, ease: 'easeOut' }}
                  className="flex flex-col gap-3 w-full"
                >
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
                    onResetToFirst={handleResetToFirst}
                    showTrainingControls={true}
                    showNumpad={false}
                    disabled={!bridge}
                  />

                  {/* 2. Active Training Sample Preview on Canvas */}
                  <DrawingCanvas
                    ref={drawingRef}
                    pixelResolution={config.pixelResolution}
                    autoFitOnDraw={config.autoFitOnDraw !== false}
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
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* RIGHT/TOP COLUMN: Neural Network Visualizer (Order-1 on mobile, Order-2 on desktop) */}
          <div className="order-1 md:order-2 flex-none md:flex-1 min-w-0 h-[380px] min-h-[380px] sm:h-[440px] sm:min-h-[440px] md:h-auto md:min-h-[540px] relative flex flex-col rounded-2xl border border-slate-800/80 bg-slate-950/80 overflow-hidden shadow-xl">
            {/* Top Bar: Left Legend / Mode Badge, Right Mode Switcher (Graph vs Mechanism) */}
            <div className="absolute top-2 left-2.5 right-2.5 sm:top-2.5 sm:left-3 sm:right-3 z-30 flex items-center justify-between pointer-events-none gap-1.5">
              {/* Left Side */}
              {visualizerMode === 'graph' ? (
                <div className="pointer-events-auto flex items-center gap-1.5 sm:gap-3 text-[10px] font-mono text-slate-300 bg-slate-900/90 backdrop-blur-xs px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-lg border border-slate-800/80 shadow-sm shrink-0">
                  <span className="flex items-center gap-1 sm:gap-1.5 font-semibold text-slate-200 font-sans">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span className="hidden sm:inline">Active Flow</span>
                    <span className="sm:hidden text-[9.5px]">Flow</span>
                  </span>
                  <span className="hidden sm:flex items-center gap-1.5 text-slate-400">
                    <span className="w-3.5 h-1 bg-emerald-400 rounded-full shadow-[0_0_6px_rgba(52,211,153,0.8)]"></span>
                    <span className="text-emerald-300 font-medium">+Weight</span>
                  </span>
                  <span className="hidden sm:flex items-center gap-1.5 text-slate-400">
                    <span className="w-3.5 h-1 bg-fuchsia-400 rounded-full shadow-[0_0_6px_rgba(232,121,249,0.8)]"></span>
                    <span className="text-fuchsia-300 font-medium">-Weight</span>
                  </span>
                </div>
              ) : (
                <div />
              )}

              {/* Right Side: Tab Switcher (Graph vs Mechanism) with Sliding Pill */}
              <div className="pointer-events-auto relative flex items-center bg-slate-900/90 backdrop-blur-xs p-0.5 rounded-lg border border-slate-800/80 shadow-md shrink-0">
                <button
                  type="button"
                  data-testid="tab-visualizer-graph"
                  onClick={() => setVisualizerMode('graph')}
                  className={`relative z-10 flex items-center gap-1 py-0.5 sm:py-1 px-2 sm:px-2.5 rounded-md text-[11px] font-medium transition-colors cursor-pointer select-none ${
                    visualizerMode === 'graph' ? 'text-cyan-300 font-semibold' : 'text-slate-400 hover:text-slate-200'
                  }`}
                  title="Interactive Node & Synapse Graph"
                >
                  <Network size={11} className="shrink-0" />
                  <span className="font-mono">Graph</span>
                  {visualizerMode === 'graph' && (
                    <motion.div
                      layoutId="vis-mode-pill"
                      className="absolute inset-0 bg-cyan-500/20 border border-cyan-500/40 rounded-md shadow-xs -z-10"
                      transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                    />
                  )}
                </button>
                <button
                  type="button"
                  data-testid="tab-visualizer-mechanism"
                  onClick={() => setVisualizerMode('mechanism')}
                  className={`relative z-10 flex items-center gap-1 py-0.5 sm:py-1 px-2 sm:px-2.5 rounded-md text-[11px] font-medium transition-colors cursor-pointer select-none ${
                    visualizerMode === 'mechanism' ? 'text-emerald-300 font-semibold' : 'text-slate-400 hover:text-slate-200'
                  }`}
                  title="Step-by-Step Mathematical Mechanism & Functions Preview"
                >
                  <Sigma size={11} className="shrink-0" />
                  <span className="font-mono">Mechanism</span>
                  {visualizerMode === 'mechanism' && (
                    <motion.div
                      layoutId="vis-mode-pill"
                      className="absolute inset-0 bg-emerald-500/20 border border-emerald-500/40 rounded-md shadow-xs -z-10"
                      transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                    />
                  )}
                </button>
              </div>
            </div>

            {/* Visualizer Canvas or Mechanism Walkthrough with Smooth Cross-Fade Transition */}
            <div className="flex-1 w-full relative overflow-hidden">
              <AnimatePresence mode="wait">
                {visualizerMode === 'graph' ? (
                  <motion.div
                    key="view-graph"
                    initial={{ opacity: 0, scale: 0.985 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.985 }}
                    transition={{ duration: 0.25, ease: 'easeOut' }}
                    className="absolute inset-0 w-full h-full"
                  >
                    <NetworkVisualizer
                      bridge={bridge}
                      animating={isAnimating}
                      animationDuration={animationDuration}
                      activeDigit={prediction?.digit ?? null}
                    />

                    {/* Bottom Status Bar */}
                    <div className="absolute bottom-1.5 left-2 right-2 sm:left-3 sm:right-3 z-20 pointer-events-none flex items-center justify-between text-[8.5px] sm:text-[10px] font-mono text-slate-500 bg-slate-950/80 backdrop-blur-xs px-2 py-0.5 rounded border border-slate-900 shadow-sm">
                      <span className="truncate mr-1.5">
                        <span className="hidden sm:inline">Topology: </span>[{config.inputSize}, {config.hiddenLayers.map((l) => l.neurons).join(', ')}, {config.outputSize}]
                      </span>
                      <span className="flex-shrink-0 text-[8.5px] sm:text-[10px]">
                        <span className="hidden sm:inline">{config.hiddenActivation} • </span>
                        LR: {config.learningRate.toFixed(2)} • Loss:{' '}
                        <span className="text-emerald-400 font-semibold">{trainingStats.currentLoss.toFixed(4)}</span>
                      </span>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="view-mechanism"
                    initial={{ opacity: 0, scale: 0.985 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.985 }}
                    transition={{ duration: 0.25, ease: 'easeOut' }}
                    className="absolute inset-0 w-full h-full"
                  >
                    <MechanismVisualizer
                      config={config}
                      currentPixels={currentPixels}
                      prediction={prediction}
                      trainingStats={trainingStats}
                      currentSampleInfo={currentSampleInfo}
                      isTraining={isLiveTraining}
                      bridge={bridge}
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </main>

      {/* SLIDE-OVER SIDE DRAWER: Opens from the RIGHT side */}
      {drawerTab !== null && (
        <div className="fixed inset-0 z-50 overflow-hidden flex justify-end">
          {/* Subtle Backdrop with less blur */}
          <div
            className="fixed inset-0 bg-black/40 backdrop-blur-xs transition-opacity animate-in fade-in duration-150"
            onClick={() => setDrawerTab(null)}
          />

          {/* Sliding Panel from the Right */}
          <div className="relative w-full sm:max-w-[400px] h-full bg-slate-900 border-l border-slate-700/80 shadow-2xl flex flex-col z-10 animate-in slide-in-from-right duration-200">
            {/* Drawer Top Navigation & Close Button */}
            <div className="flex items-center justify-between p-2.5 border-b border-slate-800 bg-slate-950/80">
              <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800 text-xs">
                <button
                  type="button"
                  data-testid="drawer-tab-architecture"
                  onClick={() => {
                    setDrawerTab('architecture');
                    setActiveDrawerTab('architecture');
                  }}
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
                  onClick={() => {
                    setDrawerTab('storage');
                    setActiveDrawerTab('storage');
                  }}
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
                  onResetWeights={() => setIsResetConfirmOpen(true)}
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
                    onResetWeights={() => setIsResetConfirmOpen(true)}
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
      {/* Confirmation Dialog for Resetting / Randomizing Weights */}
      <Dialog
        isOpen={isResetConfirmOpen}
        onClose={() => setIsResetConfirmOpen(false)}
        title="Reset & Randomize Weights?"
        description="This will re-initialize all layer weights and biases to random values using Xavier/He normal distribution. Current training progress will be lost."
        maxWidth="sm"
        footer={
          <div className="flex items-center justify-end gap-2 w-full pt-2">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setIsResetConfirmOpen(false)}
              className="text-xs text-slate-400 hover:text-slate-200"
            >
              Cancel
            </Button>
            <Button
              size="sm"
              variant="primary"
              onClick={() => {
                resetWeights();
                setIsResetConfirmOpen(false);
                showToast('Weights & biases reset to random!', 'info');
              }}
              className="text-xs bg-rose-600 hover:bg-rose-500 text-white border-rose-500 shadow-xs"
            >
              Reset Weights
            </Button>
          </div>
        }
      />
    </div>
  );
}
