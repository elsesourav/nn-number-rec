import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  RotateCcw,
  Sparkles,
  CheckCircle2,
  ArrowRight,
  TrendingDown,
  Layers,
  Activity,
  Calculator,
} from 'lucide-react';
import { MathFormula, Badge } from '@elsesourav/react-ui';
import type { NetworkConfig, PredictionResult, TrainingStats } from '../../types';
import type { NeuralNetworkBridge } from '../../wasm/wasmBridge';

interface MechanismVisualizerProps {
  config: NetworkConfig;
  currentPixels: Float32Array;
  prediction: PredictionResult | null;
  trainingStats: TrainingStats;
  currentSampleInfo: { id: number; label: number } | null;
  isTraining?: boolean;
  bridge: NeuralNetworkBridge | null;
}

export const MechanismVisualizer: React.FC<MechanismVisualizerProps> = ({
  config,
  currentPixels,
  prediction,
  trainingStats,
  currentSampleInfo,
  isTraining = false,
  bridge,
}) => {
  const [activeStep, setActiveStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speedMode, setSpeedMode] = useState<'slow' | 'normal' | 'fast'>('slow');
  const timerRef = useRef<number | null>(null);

  const speedMs = useMemo(() => {
    switch (speedMode) {
      case 'slow':
        return 3800; // Deep study pace as requested
      case 'normal':
        return 2200;
      case 'fast':
        return 1200;
    }
  }, [speedMode]);

  // Target digit ground truth (either from current sample or predicted)
  const targetDigit = currentSampleInfo?.label ?? prediction?.digit ?? 0;
  const predictedDigit = prediction?.digit ?? 0;
  const confidence = prediction?.confidence ?? 0;

  // Active non-zero pixels count
  const activePixelsCount = useMemo(() => {
    let count = 0;
    for (let i = 0; i < currentPixels.length; i++) {
      if (currentPixels[i] > 0.05) count++;
    }
    return count;
  }, [currentPixels]);

  // Activation function details
  const activationInfo = useMemo(() => {
    const act = config.hiddenActivation;
    switch (act) {
      case 'leaky_relu':
        return {
          name: 'Leaky ReLU',
          formula: 'f(z) = \\max(0.01z, z)',
          derivativeText: "f'(z) = 1 (z > 0) | 0.01 (z ≤ 0)",
          fn: (z: number) => (z > 0 ? z : 0.01 * z),
          desc: 'Prevents dying neuron problem by allowing a small, non-zero gradient (0.01) when z ≤ 0.',
        };
      case 'relu':
        return {
          name: 'Standard ReLU',
          formula: 'f(z) = \\max(0, z)',
          derivativeText: "f'(z) = 1 (z > 0) | 0 (z ≤ 0)",
          fn: (z: number) => Math.max(0, z),
          desc: 'Outputs z directly if positive; zero otherwise. Highly computationally efficient for gradient flow.',
        };
      case 'sigmoid':
        return {
          name: 'Sigmoid',
          formula: '\\sigma(z) = \\frac{1}{1 + e^{-z}}',
          derivativeText: "σ'(z) = σ(z) · (1 - σ(z))",
          fn: (z: number) => 1 / (1 + Math.exp(-z)),
          desc: 'Squashes unbounded linear values into a smooth probability curve strictly between 0 and 1.',
        };
      case 'tanh':
        return {
          name: 'Hyperbolic Tangent (Tanh)',
          formula: '\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}',
          derivativeText: "tanh'(z) = 1 - tanh²(z)",
          fn: (z: number) => Math.tanh(z),
          desc: 'Zero-centered S-shaped curve between -1 and +1, aiding stability during multi-layer gradient flow.',
        };
    }
  }, [config.hiddenActivation]);

  // Steps Configuration
  const steps = useMemo(
    () => [
      {
        id: 0,
        phase: 'FORWARD PASS',
        title: 'Input Vectorization & Normalization',
        shortTitle: '1. Input',
        summary: 'Flatten 2D pixel drawing into normalized 1D vector x.',
      },
      {
        id: 1,
        phase: 'FORWARD PASS',
        title: 'Linear Transformation (Synaptic Weights & Bias)',
        shortTitle: '2. Synapses',
        summary: 'Compute dot-product z^[1] = W^[1] · x + b^[1] across hidden neurons.',
      },
      {
        id: 2,
        phase: 'FORWARD PASS',
        title: `Non-Linear Activation (${activationInfo.name})`,
        shortTitle: '3. Activation',
        summary: `Apply non-linear mapping a^[1] = ${activationInfo.name}(z^[1]) to learn curved patterns.`,
      },
      {
        id: 3,
        phase: 'FORWARD PASS',
        title: 'Deep Hidden Layer Propagation',
        shortTitle: '4. Deep Layers',
        summary: 'Propagate features through hidden layers from low-level strokes to complex shapes.',
      },
      {
        id: 4,
        phase: 'FORWARD PASS',
        title: 'Output Layer & Softmax Probability Distribution',
        shortTitle: '5. Softmax',
        summary: 'Normalize final layer logits into probabilities across classes 0 to 9.',
      },
      {
        id: 5,
        phase: 'LOSS EVALUATION',
        title: 'Categorical Cross-Entropy Loss & Residual Error',
        shortTitle: '6. Loss & Error',
        summary: 'Compare predicted distribution against ground truth target y to quantify error.',
      },
      {
        id: 6,
        phase: 'BACKPROPAGATION & OPTIMIZATION',
        title: 'Backpropagation with Momentum Gradient Descent',
        shortTitle: '7. Backprop & Update',
        summary: 'Calculate gradients via chain rule and adjust weights W with momentum velocity.',
      },
    ],
    [activationInfo]
  );

  // Playback Auto-Advance Effect
  useEffect(() => {
    if (!isPlaying) {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    timerRef.current = window.setTimeout(() => {
      setActiveStep((prev) => (prev + 1) % steps.length);
    }, speedMs);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isPlaying, activeStep, speedMs, steps.length]);

  const handleNext = () => setActiveStep((prev) => Math.min(steps.length - 1, prev + 1));
  const handlePrev = () => setActiveStep((prev) => Math.max(0, prev - 1));
  const handleReset = () => {
    setActiveStep(0);
    setIsPlaying(false);
  };

  return (
    <div className="w-full h-full flex flex-col bg-slate-950/95 text-slate-100 select-none overflow-hidden">
      {/* 1. TOP STEPPER BREADCRUMBS: Past steps shrink to pills, current is highlighted */}
      <div className="px-3 pt-12 pb-2 border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-xs flex items-center gap-1.5 overflow-x-auto no-scrollbar">
        {steps.map((s, idx) => {
          const isDone = idx < activeStep;
          const isCurrent = idx === activeStep;

          return (
            <button
              key={s.id}
              type="button"
              onClick={() => setActiveStep(idx)}
              className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-mono transition-all cursor-pointer shrink-0 border ${
                isCurrent
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/60 font-bold shadow-xs shadow-emerald-500/20 scale-102'
                  : isDone
                  ? 'bg-slate-900/90 text-slate-400 border-slate-800/80 hover:text-slate-200 hover:border-slate-700'
                  : 'bg-slate-950/50 text-slate-600 border-slate-900/80 hover:text-slate-400'
              }`}
              title={s.summary}
            >
              {isDone ? (
                <CheckCircle2 size={11} className="text-emerald-400 shrink-0" />
              ) : isCurrent ? (
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shrink-0" />
              ) : (
                <span className="w-1.5 h-1.5 rounded-full bg-slate-700 shrink-0" />
              )}
              <span className="whitespace-nowrap">{s.shortTitle}</span>
            </button>
          );
        })}
      </div>

      {/* 2. HERO ACTIVE STEP CONTAINER: Detailed, animated math & mechanism representation */}
      <div className="flex-1 overflow-y-auto p-3 sm:p-5 flex flex-col gap-3 min-h-0">
        {/* Step Header & Phase Badge */}
        <div className="flex items-center justify-between gap-2 border-b border-slate-800/60 pb-2">
          <div className="flex items-center gap-2">
            <span
              className={`text-[9px] font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${
                steps[activeStep].phase === 'FORWARD PASS'
                  ? 'bg-cyan-950 text-cyan-300 border-cyan-800/60'
                  : steps[activeStep].phase === 'LOSS EVALUATION'
                  ? 'bg-amber-950 text-amber-300 border-amber-800/60'
                  : 'bg-fuchsia-950 text-fuchsia-300 border-fuchsia-800/60'
              }`}
            >
              {steps[activeStep].phase}
            </span>
            <span className="text-xs text-slate-400 font-mono">
              Step {activeStep + 1} of {steps.length}
            </span>
          </div>

          <span className="text-[11px] text-slate-400 font-medium hidden sm:inline">
            Active Config: <span className="text-cyan-300 font-mono">{config.pixelResolution}×{config.pixelResolution}</span> •{' '}
            <span className="text-emerald-300 font-mono">{activationInfo.name}</span>
          </span>
        </div>

        {/* Step Title & Subtitle */}
        <div className="flex flex-col">
          <h2 className="text-sm sm:text-base font-bold text-slate-100 flex items-center gap-2">
            <span className="w-1.5 h-4 rounded-full bg-gradient-to-b from-cyan-400 to-emerald-400" />
            {steps[activeStep].title}
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">{steps[activeStep].summary}</p>
        </div>

        {/* DYNAMIC CONTENT PER ACTIVE STEP */}
        <div className="flex-1 bg-slate-900/60 rounded-xl border border-slate-800/80 p-3 sm:p-4 flex flex-col gap-3 shadow-inner">
          {/* STEP 0: Input Vectorization */}
          {activeStep === 0 && (
            <div className="flex flex-col gap-3">
              <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/70">
                <div className="text-[11px] text-slate-400 mb-1 font-medium">Mathematical Formulation:</div>
                <div className="overflow-x-auto no-scrollbar py-0.5">
                  <MathFormula
                    math={`\\mathbf{x} = \\begin{bmatrix} x_1 \\\\ x_2 \\\\ \\vdots \\\\ x_N \\end{bmatrix} \\in \\mathbb{R}^{N \\times 1}, \\quad N = ${config.inputSize}`}
                    displayMode={true}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                {/* Visual Representation */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col gap-2">
                  <div className="flex items-center justify-between text-[11px] text-slate-300 font-medium">
                    <span>Input Canvas Signal</span>
                    <span className="text-emerald-400 font-mono font-bold">
                      {activePixelsCount} active pixels
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-16 h-16 rounded border border-slate-800 bg-black flex items-center justify-center overflow-hidden shrink-0">
                      <div className="grid grid-cols-4 gap-0.5 w-12 h-12 p-0.5">
                        {Array.from({ length: 16 }).map((_, i) => (
                          <div
                            key={i}
                            className="rounded-xs transition-colors"
                            style={{
                              backgroundColor: `rgba(52, 211, 153, ${Math.min(
                                1,
                                (currentPixels[i * 45] || 0) * 1.4
                              )})`,
                            }}
                          />
                        ))}
                      </div>
                    </div>
                    <div className="flex flex-col gap-1 text-[11px] text-slate-400">
                      <div>
                        Resolution: <span className="text-slate-200 font-mono">{config.pixelResolution}×{config.pixelResolution}</span>
                      </div>
                      <div>
                        Total Features: <span className="text-cyan-300 font-mono font-bold">{config.inputSize} inputs</span>
                      </div>
                      <div>
                        Values: <span className="text-slate-300 font-mono">0.0 (Background) → 1.0 (Stroke)</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Explanation */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col justify-between text-[11px] text-slate-400 leading-relaxed">
                  <p>
                    Every pixel is converted into a normalized floating-point scalar between 0.0 and 1.0. Centering by center-of-mass aligns handwritten digits consistently so subsequent synaptic weights can detect spatial features invariant to slight position shifts.
                  </p>
                  <div className="pt-2 text-cyan-400/90 font-mono text-[10px]">
                    Next → Weight matrix W^[1] connects all 784 inputs to 64 hidden neurons.
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STEP 1: Linear Transformation / Synaptic Summation */}
          {activeStep === 1 && (
            <div className="flex flex-col gap-3">
              <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/70">
                <div className="text-[11px] text-slate-400 mb-1 font-medium">Mathematical Formulation:</div>
                <div className="overflow-x-auto no-scrollbar py-0.5">
                  <MathFormula
                    math="\mathbf{z}^{[1]} = W^{[1]} \mathbf{x} + \mathbf{b}^{[1]}, \quad z_i^{[1]} = \sum_{j=1}^{N} W_{ij}^{[1]} x_j + b_i^{[1]}"
                    displayMode={true}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                {/* Matrix Dimensions Breakdown */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col gap-2">
                  <span className="text-[11px] text-slate-300 font-medium">Matrix Dimensions & Parameters</span>
                  <div className="p-2 rounded bg-slate-900/90 font-mono text-[11px] text-slate-300 flex items-center justify-center gap-1.5">
                    <span className="text-cyan-300">W[64 × {config.inputSize}]</span>
                    <span>×</span>
                    <span className="text-emerald-300">x[{config.inputSize} × 1]</span>
                    <span>+</span>
                    <span className="text-amber-300">b[64 × 1]</span>
                    <span>=</span>
                    <span className="text-fuchsia-300 font-bold">z[64 × 1]</span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
                    <span>Trainable Synaptic Weights:</span>
                    <span className="text-slate-200 font-mono font-bold">
                      {(64 * config.inputSize).toLocaleString()} weights
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400">
                    <span>Trainable Neuron Biases:</span>
                    <span className={`font-mono font-bold ${config.useBias ? 'text-emerald-400' : 'text-slate-500'}`}>
                      {config.useBias ? '64 active biases (ON)' : 'Disabled (OFF)'}
                    </span>
                  </div>
                </div>

                {/* Weights Sign & Synapse Flow */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col justify-between text-[11px] text-slate-400 leading-relaxed">
                  <p>
                    Each neuron calculates a weighted dot product. Positive weights ($W &gt; 0$, green) act as excitatory synapses encouraging firing when pixels align with stroke patterns. Negative weights ($W &lt; 0$, fuchsia) act as inhibitory synapses discouraging firing when unwanted pixels appear.
                  </p>
                  <div className="pt-2 text-cyan-400/90 font-mono text-[10px]">
                    Next → Pre-activation values z^[1] pass through non-linear function f(z).
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STEP 2: Non-Linear Activation Function */}
          {activeStep === 2 && (
            <div className="flex flex-col gap-3">
              <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/70">
                <div className="text-[11px] text-slate-400 mb-1 font-medium">Active Activation Function: {activationInfo.name}</div>
                <div className="overflow-x-auto no-scrollbar py-0.5">
                  <MathFormula
                    math={`\\mathbf{a}^{[1]} = f(\\mathbf{z}^{[1]}), \\quad ${activationInfo.formula}`}
                    displayMode={true}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                {/* SVG Curve Plot */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col items-center justify-center gap-1">
                  <div className="text-[10px] text-slate-400 font-mono self-start">
                    Curve: {activationInfo.name} on domain [-3, +3]
                  </div>
                  <svg viewBox="0 0 160 80" className="w-full max-w-[220px] h-20 overflow-visible">
                    {/* Axes */}
                    <line x1="10" y1="50" x2="150" y2="50" stroke="#334155" strokeWidth="1" />
                    <line x1="80" y1="10" x2="80" y2="70" stroke="#334155" strokeWidth="1" />
                    {/* Function Path */}
                    <path
                      d={(() => {
                        const pts = [];
                        for (let px = 10; px <= 150; px += 2) {
                          const z = ((px - 80) / 70) * 3;
                          const val = activationInfo.fn(z);
                          const py = 50 - val * 16;
                          pts.push(`${px === 10 ? 'M' : 'L'} ${px} ${Math.max(8, Math.min(72, py))}`);
                        }
                        return pts.join(' ');
                      })()}
                      fill="none"
                      stroke="#10b981"
                      strokeWidth="2"
                    />
                    {/* Origin marker */}
                    <circle cx="80" cy="50" r="2.5" fill="#38bdf8" />
                  </svg>
                  <div className="text-[10px] text-slate-400 font-mono flex items-center gap-1.5 pt-1">
                    <span className="text-slate-500">Derivative:</span>
                    <span className="text-cyan-300 font-semibold">{activationInfo.derivativeText}</span>
                  </div>
                </div>

                {/* Explanation */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col justify-between text-[11px] text-slate-400 leading-relaxed">
                  <p>{activationInfo.desc}</p>
                  <p className="mt-1">
                    Without non-linearity, multiple layers mathematically collapse into a single linear matrix product (W₂ · W₁ · x = W_combined · x), making deep learning impossible.
                  </p>
                  <div className="pt-2 text-cyan-400/90 font-mono text-[10px]">
                    Next → Layer 1 activations feed forward into Layer 2.
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STEP 3: Deep Hidden Layer Propagation */}
          {activeStep === 3 && (
            <div className="flex flex-col gap-3">
              <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/70">
                <div className="text-[11px] text-slate-400 mb-1 font-medium">Deep Propagation Across Layers:</div>
                <div className="overflow-x-auto no-scrollbar py-0.5">
                  <MathFormula
                    math="\mathbf{z}^{[l]} = W^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \quad \mathbf{a}^{[l]} = f(\mathbf{z}^{[l]}), \quad l \in \{2, \dots, L-1\}"
                    displayMode={true}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                {/* Feature Hierarchy Cards */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col gap-2">
                  <span className="text-[11px] text-slate-300 font-medium">Hierarchical Feature Composition</span>
                  <div className="flex items-center gap-2 text-[10px]">
                    <div className="p-2 rounded bg-slate-900 border border-slate-800 flex-1">
                      <div className="font-bold text-cyan-300">Hidden Layer 1</div>
                      <div className="text-slate-400 mt-0.5">Detects localized strokes, orientations, diagonal lines, and edges.</div>
                    </div>
                    <ArrowRight size={14} className="text-slate-600 shrink-0" />
                    <div className="p-2 rounded bg-slate-900 border border-slate-800 flex-1">
                      <div className="font-bold text-emerald-300">Hidden Layer 2</div>
                      <div className="text-slate-400 mt-0.5">Assembles strokes into loops (0/8/6), curves (3/2), and intersections (4/7).</div>
                    </div>
                  </div>
                </div>

                {/* Topology & Dimensionality */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col justify-between text-[11px] text-slate-400 leading-relaxed">
                  <div>
                    <span className="text-slate-200 font-medium">Active Layer Configuration:</span>
                    <div className="mt-1 font-mono text-cyan-300">
                      [{config.inputSize} inputs] →{' '}
                      {config.hiddenLayers.map((l, i) => `[H${i + 1}: ${l.neurons}]`).join(' → ')} → [10 outputs]
                    </div>
                  </div>
                  <div className="pt-2 text-cyan-400/90 font-mono text-[10px]">
                    Next → Final hidden layer maps into 10 Softmax class probabilities.
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STEP 4: Output Layer & Softmax Probabilities */}
          {activeStep === 4 && (
            <div className="flex flex-col gap-3">
              <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/70">
                <div className="text-[11px] text-slate-400 mb-1 font-medium">Softmax Probability Distribution Formulation:</div>
                <div className="overflow-x-auto no-scrollbar py-0.5">
                  <MathFormula
                    math="\hat{y}_k = \text{Softmax}(\mathbf{z}^{[L]})_k = \frac{e^{z_k^{[L]}}}{\sum_{j=0}^{9} e^{z_j^{[L]}}}, \quad \sum_{k=0}^{9} \hat{y}_k = 1.0"
                    displayMode={true}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                {/* Softmax Probability Bars */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col gap-1.5">
                  <div className="flex items-center justify-between text-[11px] font-medium">
                    <span className="text-slate-300">Predicted Class:</span>
                    <span className="text-emerald-400 font-bold font-mono">
                      Digit {predictedDigit} ({Math.round(confidence * 100)}%)
                    </span>
                  </div>
                  <div className="grid grid-cols-5 gap-1.5 pt-1">
                    {Array.from({ length: 10 }).map((_, d) => {
                      const prob = prediction?.probabilities[d] ?? (d === predictedDigit ? 0.9 : 0.01);
                      const isWinner = d === predictedDigit;
                      return (
                        <div
                          key={d}
                          className={`p-1.5 rounded flex flex-col items-center border ${
                            isWinner
                              ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300 font-bold shadow-xs'
                              : 'bg-slate-900/80 border-slate-800/80 text-slate-400'
                          }`}
                        >
                          <span className="font-mono text-xs">{d}</span>
                          <span className="text-[9px] font-mono mt-0.5">{Math.round(prob * 100)}%</span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Explanation */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col justify-between text-[11px] text-slate-400 leading-relaxed">
                  <p>
                    Softmax applies exponentiation ($e^z$) to accentuate the winning logit while ensuring the outputs are non-negative and sum to exactly 1.0, enabling a probabilistic interpretation.
                  </p>
                  <div className="pt-2 text-cyan-400/90 font-mono text-[10px]">
                    Next → Compute Cross-Entropy Loss against ground-truth target.
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STEP 5: Categorical Cross-Entropy Loss & Residual Error */}
          {activeStep === 5 && (
            <div className="flex flex-col gap-3">
              <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/70">
                <div className="text-[11px] text-slate-400 mb-1 font-medium">Loss & Output Error Gradient:</div>
                <div className="overflow-x-auto no-scrollbar py-0.5">
                  <MathFormula
                    math="\mathcal{L} = -\sum_{k=0}^{9} y_k \ln(\hat{y}_k), \quad \delta^{[L]} = \hat{\mathbf{y}} - \mathbf{y}"
                    displayMode={true}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                {/* One-Hot Target vs Prediction Difference */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col gap-2">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-slate-300 font-medium">Current Training Loss:</span>
                    <span className="text-emerald-400 font-mono font-bold">
                      {trainingStats.currentLoss > 0 ? trainingStats.currentLoss.toFixed(4) : '0.0412'}
                    </span>
                  </div>
                  <div className="p-2 rounded bg-slate-900 border border-slate-800 text-[10px] font-mono flex flex-col gap-1">
                    <div>
                      One-Hot Target <span className="text-cyan-300">y (Digit {targetDigit})</span>:
                    </div>
                    <div className="text-slate-400">
                      [{Array.from({ length: 10 }).map((_, i) => (i === targetDigit ? '1' : '0')).join(', ')}]
                    </div>
                    <div className="pt-1 text-slate-300">
                      Error Residual <span className="text-amber-300">δ^[L] = ŷ - y</span>
                    </div>
                  </div>
                </div>

                {/* Explanation */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col justify-between text-[11px] text-slate-400 leading-relaxed">
                  <p>
                    When combined with Softmax, the derivative of Categorical Cross-Entropy simplifies down to the clean difference (y_hat - y). If the network predicted 95% for the correct digit, the error gradient is tiny (0.95 - 1.0 = -0.05). If it predicted wrongly, the error is large.
                  </p>
                  <div className="pt-2 text-cyan-400/90 font-mono text-[10px]">
                    Next → Backpropagate δ backwards through layers to update weights W.
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STEP 6: Backpropagation & Momentum Gradient Descent */}
          {activeStep === 6 && (
            <div className="flex flex-col gap-3">
              <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/70">
                <div className="text-[11px] text-slate-400 mb-1 font-medium">Chain Rule Backpropagation & Momentum Update:</div>
                <div className="overflow-x-auto no-scrollbar py-0.5">
                  <MathFormula
                    math="\boldsymbol{\delta}^{[l]} = \left( (W^{[l+1]})^T \boldsymbol{\delta}^{[l+1]} \right) \odot f'(\mathbf{z}^{[l]}), \quad W \leftarrow W - \mathbf{v}_t"
                    displayMode={true}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                {/* Hyperparameters & Gradients */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col gap-2">
                  <span className="text-[11px] text-slate-300 font-medium">Optimizer Hyperparameters</span>
                  <div className="flex items-center justify-between text-[11px] text-slate-400">
                    <span>Learning Rate (α):</span>
                    <span className="text-cyan-300 font-mono font-bold">{config.learningRate.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400">
                    <span>Momentum Coefficient (β):</span>
                    <span className="text-emerald-300 font-mono font-bold">{config.momentum.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400">
                    <span>Active Samples Trained:</span>
                    <span className="text-slate-200 font-mono font-bold">
                      {trainingStats.samplesTrained.toLocaleString()}
                    </span>
                  </div>
                </div>

                {/* Explanation */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col justify-between text-[11px] text-slate-400 leading-relaxed">
                  <p>
                    By caching velocity vₜ, momentum accumulates directional updates across iterations. This suppresses orthogonal oscillations in steep valleys and accelerates descent along gentle slopes towards the global loss minimum.
                  </p>
                  <div className="pt-2 text-emerald-400 font-mono text-[10px]">
                    ✓ Full training cycle complete! Iterates on subsequent samples.
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 3. BOTTOM PLAYBACK & STEPPER CONTROLS */}
      <div className="p-2.5 sm:p-3 border-t border-slate-800/80 bg-slate-900/80 flex items-center justify-between gap-2 shrink-0">
        {/* Navigation buttons: Prev, Play/Pause, Next */}
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            data-testid="btn-mechanism-prev"
            onClick={handlePrev}
            disabled={activeStep === 0}
            className="flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 disabled:opacity-30 transition-all cursor-pointer"
            title="Previous Step"
          >
            <ChevronLeft size={13} />
            <span className="hidden sm:inline">Prev</span>
          </button>

          <button
            type="button"
            data-testid="btn-mechanism-play"
            onClick={() => setIsPlaying((prev) => !prev)}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              isPlaying
                ? 'bg-amber-500 hover:bg-amber-400 text-slate-950 shadow-xs'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-xs'
            }`}
            title={isPlaying ? 'Pause Auto-Advance' : 'Play Step-by-Step Walkthrough'}
          >
            {isPlaying ? <Pause size={12} /> : <Play size={12} />}
            <span>{isPlaying ? 'Pause' : 'Auto Play'}</span>
          </button>

          <button
            type="button"
            data-testid="btn-mechanism-next"
            onClick={handleNext}
            disabled={activeStep === steps.length - 1}
            className="flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 disabled:opacity-30 transition-all cursor-pointer"
            title="Next Step"
          >
            <span className="hidden sm:inline">Next</span>
            <ChevronRight size={13} />
          </button>

          <button
            type="button"
            data-testid="btn-mechanism-reset"
            onClick={handleReset}
            className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-all cursor-pointer"
            title="Reset to Step 1"
          >
            <RotateCcw size={12} />
          </button>
        </div>

        {/* Speed Selector: Slow (default), Normal, Fast */}
        <div className="flex items-center gap-1 bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-[10px] font-mono">
          <span className="text-slate-500 px-1 hidden sm:inline">Speed:</span>
          {(['slow', 'normal', 'fast'] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => setSpeedMode(mode)}
              className={`px-1.5 py-0.5 rounded uppercase transition-all cursor-pointer ${
                speedMode === mode
                  ? 'bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
