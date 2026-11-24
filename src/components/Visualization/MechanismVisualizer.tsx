import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Play,
  RotateCcw,
  Sparkles,
  Layers,
  Activity,
  Calculator,
  ArrowRight,
  TrendingDown,
  Cpu,
  Zap,
  Sliders,
  CheckCircle2,
  RefreshCw,
  X,
  Info,
} from 'lucide-react';
import { MathFormula } from '@elsesourav/react-ui';
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

interface WorkflowNode {
  id: string;
  index: number;
  category: 'SOURCE' | 'TRANSFORM' | 'ACTIVATION' | 'DEEP' | 'CLASSIFIER' | 'LOSS' | 'OPTIMIZER';
  title: string;
  badge: string;
  color: {
    primary: string;
    border: string;
    glow: string;
    bg: string;
    text: string;
  };
  x: number;
  y: number;
  width: number;
  height: number;
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
  // -1 indicates idle (no active flow pulse). Only 0..6 when flowing based on data!
  const [activeFlowStage, setActiveFlowStage] = useState<number>(-1);
  const [lastActionType, setLastActionType] = useState<'recognize' | 'train' | 'idle'>('idle');
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Auto-fitting container scale state
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [scale, setScale] = useState<number>(1);

  // Animation timer ref
  const animTimeoutRef = useRef<number | null>(null);
  const prevPredictionRef = useRef<PredictionResult | null>(null);
  const prevTrainedCountRef = useRef<number>(trainingStats.samplesTrained);
  const isInitialMount = useRef<boolean>(true);

  // Derived Values
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

  // Responsive Auto-Fit: Scales the entire 7-node graph to ALWAYS fit 100% on screen
  useEffect(() => {
    const handleResize = () => {
      if (!containerRef.current) return;
      const { clientWidth, clientHeight } = containerRef.current;
      const baseWidth = 860;
      const baseHeight = 365;

      const scaleX = (clientWidth - 16) / baseWidth;
      const scaleY = (clientHeight - 16) / baseHeight;
      const optimalScale = Math.min(scaleX, scaleY);

      // Clamp between 0.35 on small mobile to 1.15 on ultra-wide
      setScale(Math.max(0.35, Math.min(1.15, optimalScale)));
    };

    handleResize();
    const observer = new ResizeObserver(handleResize);
    if (containerRef.current) observer.observe(containerRef.current);

    return () => observer.disconnect();
  }, []);

  // Activation details based on config
  const activationInfo = useMemo(() => {
    const act = config.hiddenActivation;
    switch (act) {
      case 'leaky_relu':
        return {
          name: 'Leaky ReLU',
          formula: 'f(z) = \\max(0.01z, z)',
          derivative: "f'(z) = 1 \\; (z > 0) \\mid 0.01 \\; (z \\le 0)",
          fn: (z: number) => (z > 0 ? z : 0.01 * z),
        };
      case 'relu':
        return {
          name: 'Standard ReLU',
          formula: 'f(z) = \\max(0, z)',
          derivative: "f'(z) = 1 \\; (z > 0) \\mid 0 \\; (z \\le 0)",
          fn: (z: number) => Math.max(0, z),
        };
      case 'sigmoid':
        return {
          name: 'Sigmoid',
          formula: '\\sigma(z) = \\frac{1}{1 + e^{-z}}',
          derivative: "\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))",
          fn: (z: number) => 1 / (1 + Math.exp(-z)),
        };
      case 'tanh':
        return {
          name: 'Tanh',
          formula: '\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}',
          derivative: "\\tanh'(z) = 1 - \\tanh^2(z)",
          fn: (z: number) => Math.tanh(z),
        };
    }
  }, [config.hiddenActivation]);

  // 2-Row Compact Graph Layout (Fits 860x365 container perfectly)
  const nodes: WorkflowNode[] = useMemo(() => {
    const nodeW = 175;
    const nodeH = 145;
    const row1Y = 20;
    const row2Y = 195;

    return [
      // ROW 1: Forward Pass (Left -> Right)
      {
        id: 'node-input',
        index: 0,
        category: 'SOURCE',
        title: 'Input Tensor x',
        badge: `${config.pixelResolution}×${config.pixelResolution}`,
        color: {
          primary: '#06b6d4',
          border: 'border-cyan-500/60',
          glow: 'rgba(6, 182, 212, 0.45)',
          bg: 'bg-cyan-950/40',
          text: 'text-cyan-400',
        },
        x: 15,
        y: row1Y,
        width: nodeW,
        height: nodeH,
      },
      {
        id: 'node-synapse',
        index: 1,
        category: 'TRANSFORM',
        title: 'Dense Synapses',
        badge: 'W·x + b',
        color: {
          primary: '#3b82f6',
          border: 'border-blue-500/60',
          glow: 'rgba(59, 130, 246, 0.45)',
          bg: 'bg-blue-950/40',
          text: 'text-blue-400',
        },
        x: 220,
        y: row1Y,
        width: nodeW,
        height: nodeH,
      },
      {
        id: 'node-activation',
        index: 2,
        category: 'ACTIVATION',
        title: activationInfo.name,
        badge: 'f(z)',
        color: {
          primary: '#10b981',
          border: 'border-emerald-500/60',
          glow: 'rgba(16, 185, 129, 0.45)',
          bg: 'bg-emerald-950/40',
          text: 'text-emerald-400',
        },
        x: 425,
        y: row1Y,
        width: nodeW,
        height: nodeH,
      },
      {
        id: 'node-deep',
        index: 3,
        category: 'DEEP',
        title: 'Hidden Propagation',
        badge: `${config.hiddenLayers.length} Layers`,
        color: {
          primary: '#8b5cf6',
          border: 'border-violet-500/60',
          glow: 'rgba(139, 92, 246, 0.45)',
          bg: 'bg-violet-950/40',
          text: 'text-violet-400',
        },
        x: 630,
        y: row1Y,
        width: nodeW,
        height: nodeH,
      },

      // ROW 2: Prediction, Loss & Backpropagation (Right -> Left)
      {
        id: 'node-softmax',
        index: 4,
        category: 'CLASSIFIER',
        title: 'Softmax Output',
        badge: '0–9 Probs',
        color: {
          primary: '#f59e0b',
          border: 'border-amber-500/60',
          glow: 'rgba(245, 158, 11, 0.45)',
          bg: 'bg-amber-950/40',
          text: 'text-amber-400',
        },
        x: 630,
        y: row2Y,
        width: nodeW,
        height: nodeH,
      },
      {
        id: 'node-loss',
        index: 5,
        category: 'LOSS',
        title: 'Cross-Entropy Loss',
        badge: 'Error δ',
        color: {
          primary: '#f43f5e',
          border: 'border-rose-500/60',
          glow: 'rgba(244, 63, 94, 0.45)',
          bg: 'bg-rose-950/40',
          text: 'text-rose-400',
        },
        x: 425,
        y: row2Y,
        width: nodeW,
        height: nodeH,
      },
      {
        id: 'node-optimizer',
        index: 6,
        category: 'OPTIMIZER',
        title: 'Momentum Optimizer',
        badge: 'v_t Update',
        color: {
          primary: '#d946ef',
          border: 'border-fuchsia-500/60',
          glow: 'rgba(217, 70, 239, 0.45)',
          bg: 'bg-fuchsia-950/40',
          text: 'text-fuchsia-400',
        },
        x: 220,
        y: row2Y,
        width: nodeW,
        height: nodeH,
      },
    ];
  }, [config.pixelResolution, config.hiddenLayers.length, activationInfo.name]);

  // Connectors / Cables between nodes
  const cables = useMemo(() => {
    return [
      // 1. Forward Pass Row 1: Node 1 -> Node 2
      {
        id: 'c-1-2',
        from: 'node-input',
        to: 'node-synapse',
        label: `x [${config.inputSize}×1]`,
        stage: 0,
        color: '#06b6d4',
      },
      // 2. Forward Pass Row 1: Node 2 -> Node 3
      {
        id: 'c-2-3',
        from: 'node-synapse',
        to: 'node-activation',
        label: 'z [64×1]',
        stage: 1,
        color: '#3b82f6',
      },
      // 3. Forward Pass Row 1: Node 3 -> Node 4
      {
        id: 'c-3-4',
        from: 'node-activation',
        to: 'node-deep',
        label: 'a [64×1]',
        stage: 2,
        color: '#10b981',
      },
      // 4. Smooth 180° Right Turn Loop: Node 4 (Row 1) -> Node 5 (Row 2)
      {
        id: 'c-4-5',
        from: 'node-deep',
        to: 'node-softmax',
        label: 'z^[L] [10×1]',
        stage: 3,
        color: '#8b5cf6',
        isLoopRight: true,
      },
      // 5. Row 2 (Right to Left): Node 5 -> Node 6
      {
        id: 'c-5-6',
        from: 'node-softmax',
        to: 'node-loss',
        label: 'probs ŷ',
        stage: 4,
        color: '#f59e0b',
        isReverse: true,
      },
      // 6. Row 2 (Right to Left): Node 6 -> Node 7
      {
        id: 'c-6-7',
        from: 'node-loss',
        to: 'node-optimizer',
        label: 'error δ',
        stage: 5,
        color: '#f43f5e',
        isReverse: true,
      },
      // 7. Upward Return Loop: Node 7 (Optimizer) -> Node 2 (Synaptic Weights)
      {
        id: 'c-7-2',
        from: 'node-optimizer',
        to: 'node-synapse',
        label: 'ΔW = -vt',
        stage: 6,
        color: '#d946ef',
        isUpwardFeedback: true,
      },
    ];
  }, [config.inputSize]);

  const nodeMap = useMemo(() => {
    const map = new Map<string, WorkflowNode>();
    nodes.forEach((n) => map.set(n.id, n));
    return map;
  }, [nodes]);

  // TRIGGER ANIMATION ONCE PER DATA EVENT (Train or Recognize)
  const triggerFlowSequence = useCallback((maxStage: number, actionType: 'recognize' | 'train') => {
    if (animTimeoutRef.current) {
      window.clearTimeout(animTimeoutRef.current);
    }

    setLastActionType(actionType);
    const stepDuration = actionType === 'train' ? 120 : 140;

    let current = 0;
    setActiveFlowStage(0);

    const step = () => {
      current++;
      if (current <= maxStage) {
        setActiveFlowStage(current);
        animTimeoutRef.current = window.setTimeout(step, stepDuration);
      } else {
        // Animation finished! Return to idle
        setActiveFlowStage(-1);
      }
    };

    animTimeoutRef.current = window.setTimeout(step, stepDuration);
  }, []);

  // Watch for TRAIN or RECOGNIZE data events
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      prevTrainedCountRef.current = trainingStats.samplesTrained;
      prevPredictionRef.current = prediction;
      return;
    }

    // 1. Check for TRAINING event (samplesTrained incremented)
    if (trainingStats.samplesTrained !== prevTrainedCountRef.current) {
      prevTrainedCountRef.current = trainingStats.samplesTrained;
      prevPredictionRef.current = prediction;
      triggerFlowSequence(6, 'train'); // Full loop 0..6
      return;
    }

    // 2. Check for RECOGNIZE event (prediction updated from new drawing)
    if (prediction && prediction !== prevPredictionRef.current) {
      prevPredictionRef.current = prediction;
      triggerFlowSequence(3, 'recognize'); // Forward pass 0..3 to Softmax
      return;
    }
  }, [trainingStats.samplesTrained, prediction, triggerFlowSequence]);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (animTimeoutRef.current) window.clearTimeout(animTimeoutRef.current);
    };
  }, []);

  // Manual Replay / Test trigger
  const handleManualReplay = () => {
    triggerFlowSequence(6, 'train');
  };

  // Compute Cable Paths
  const computeCablePath = (cable: (typeof cables)[0]) => {
    const src = nodeMap.get(cable.from);
    const dst = nodeMap.get(cable.to);
    if (!src || !dst) return { pathD: '', midX: 0, midY: 0 };

    let x1 = 0,
      y1 = 0,
      x2 = 0,
      y2 = 0;
    let pathD = '';
    let midX = 0,
      midY = 0;

    if (cable.isLoopRight) {
      // 180° loop on the right (Node 4 right port -> Node 5 right port)
      x1 = src.x + src.width;
      y1 = src.y + src.height / 2;
      x2 = dst.x + dst.width;
      y2 = dst.y + dst.height / 2;
      const loopExtension = 40;
      pathD = `M ${x1} ${y1} C ${x1 + loopExtension} ${y1}, ${x2 + loopExtension} ${y2}, ${x2} ${y2}`;
      midX = x1 + loopExtension * 0.7;
      midY = (y1 + y2) / 2;
    } else if (cable.isUpwardFeedback) {
      // Upward feedback (Node 7 top port -> Node 2 bottom port)
      x1 = src.x + src.width / 2;
      y1 = src.y;
      x2 = dst.x + dst.width / 2;
      y2 = dst.y + dst.height;
      pathD = `M ${x1} ${y1} L ${x2} ${y2}`;
      midX = (x1 + x2) / 2;
      midY = (y1 + y2) / 2;
    } else if (cable.isReverse) {
      // Row 2 Right to Left (Node 5 left port -> Node 6 right port)
      x1 = src.x;
      y1 = src.y + src.height / 2;
      x2 = dst.x + dst.width;
      y2 = dst.y + dst.height / 2;
      pathD = `M ${x1} ${y1} L ${x2} ${y2}`;
      midX = (x1 + x2) / 2;
      midY = (y1 + y2) / 2;
    } else {
      // Row 1 Left to Right (Node 1 right port -> Node 2 left port)
      x1 = src.x + src.width;
      y1 = src.y + src.height / 2;
      x2 = dst.x;
      y2 = dst.y + dst.height / 2;
      pathD = `M ${x1} ${y1} L ${x2} ${y2}`;
      midX = (x1 + x2) / 2;
      midY = (y1 + y2) / 2;
    }

    return { pathD, midX, midY };
  };

  return (
    <div
      ref={containerRef}
      className="w-full h-full flex flex-col items-center justify-center bg-slate-950 text-slate-100 select-none overflow-hidden relative"
      style={{
        backgroundImage:
          'radial-gradient(circle, rgba(100, 116, 139, 0.18) 1px, transparent 1px)',
        backgroundSize: '20px 20px',
        backgroundColor: '#030712',
      }}
    >
      {/* 1. TOP STATUS & FLOW TRIGGER BAR */}
      <div className="absolute top-2 left-3 z-30 pointer-events-none">
        {/* Status Pill: Idle vs Active Flow */}
        <div className="pointer-events-auto flex items-center gap-1.5 sm:gap-2 bg-slate-900/90 backdrop-blur-md px-2 sm:px-2.5 py-1 rounded-lg border border-slate-800 shadow-md">
          <div className="flex items-center gap-1.5 text-[10px] font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                activeFlowStage >= 0
                  ? 'bg-emerald-400 animate-ping'
                  : 'bg-slate-500'
              }`}
            />
            <span className="font-bold text-slate-200 hidden sm:inline">
              {activeFlowStage >= 0
                ? lastActionType === 'train'
                  ? 'Training Flow'
                  : 'Inference Flow'
                : 'Pipeline Idle'}
            </span>
            <span className="font-bold text-slate-200 sm:hidden">
              {activeFlowStage >= 0 ? 'Flowing' : 'Idle'}
            </span>
          </div>

          <div className="h-3 w-px bg-slate-800" />

          {/* Replay Button to test on demand */}
          <button
            type="button"
            data-testid="btn-workflow-replay"
            onClick={handleManualReplay}
            className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-all cursor-pointer"
            title="Replay Animation Flow"
          >
            <RotateCcw size={9} />
            <span className="hidden sm:inline">Replay</span>
          </button>
        </div>
      </div>

      {/* 2. AUTO-SCALED 2-ROW COMPUTATIONAL GRAPH (Guaranteed 100% Fit) */}
      <div
        className="relative origin-center transition-transform duration-100 ease-out flex-shrink-0"
        style={{
          width: '855px',
          height: '360px',
          transform: `scale(${scale})`,
        }}
      >
        {/* SVG LAYER: Wires and Flowing Data Packets */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none z-10 overflow-visible">
          <defs>
            <filter id="glow-cyan" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3.5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="glow-emerald" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3.5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {cables.map((cable) => {
            const { pathD, midX, midY } = computeCablePath(cable);
            if (!pathD) return null;

            const isCurrentActive = activeFlowStage === cable.stage;

            return (
              <g key={cable.id}>
                {/* Background Shadow Cable */}
                <path
                  d={pathD}
                  fill="none"
                  stroke="#1e293b"
                  strokeWidth={4.5}
                  strokeLinecap="round"
                />
                {/* Main Cable */}
                <path
                  d={pathD}
                  fill="none"
                  stroke={isCurrentActive ? cable.color : '#334155'}
                  strokeWidth={2}
                  strokeDasharray={cable.isUpwardFeedback ? '3 3' : 'none'}
                  strokeLinecap="round"
                  className="transition-colors duration-200"
                />

                {/* Animated Luminous Packet (Only when active based on data!) */}
                {isCurrentActive && (
                  <>
                    <path
                      d={pathD}
                      fill="none"
                      stroke={cable.color}
                      strokeWidth={4}
                      strokeLinecap="round"
                      strokeDasharray="12 80"
                      className="animate-flow-dash"
                      filter="url(#glow-cyan)"
                    />
                    <circle r={4.5} fill="#ffffff" filter="url(#glow-emerald)">
                      <animateMotion path={pathD} dur="0.25s" repeatCount="1" />
                    </circle>
                  </>
                )}

                {/* Wire Data Packet Tag */}
                <g transform={`translate(${midX}, ${midY})`}>
                  <rect
                    x={-34}
                    y={-8}
                    width={68}
                    height={16}
                    rx={8}
                    fill="#0f172a"
                    stroke={isCurrentActive ? cable.color : '#334155'}
                    strokeWidth={1}
                    className="shadow-xs"
                  />
                  <text
                    textAnchor="middle"
                    y={3}
                    fill={isCurrentActive ? '#ffffff' : '#94a3b8'}
                    fontSize="8"
                    fontFamily="monospace"
                    fontWeight="bold"
                  >
                    {cable.label}
                  </text>
                </g>
              </g>
            );
          })}
        </svg>

        {/* HTML NODES LAYER (7 Interactive Workflow Nodes) */}
        {nodes.map((n) => {
          const isNodeActive =
            (n.id === 'node-input' && activeFlowStage === 0) ||
            (n.id === 'node-synapse' && (activeFlowStage === 1 || activeFlowStage === 6)) ||
            (n.id === 'node-activation' && activeFlowStage === 2) ||
            (n.id === 'node-deep' && activeFlowStage === 3) ||
            (n.id === 'node-softmax' && activeFlowStage === 4) ||
            (n.id === 'node-loss' && activeFlowStage === 5) ||
            (n.id === 'node-optimizer' && activeFlowStage === 6);

          const isSelected = selectedNodeId === n.id;

          return (
            <div
              key={n.id}
              onClick={() => setSelectedNodeId(n.id)}
              className={`interactive-node-card absolute z-20 rounded-xl bg-slate-900/95 border transition-all duration-150 cursor-pointer flex flex-col justify-between shadow-lg backdrop-blur-md ${
                isSelected
                  ? 'ring-2 ring-cyan-400 border-cyan-400 shadow-cyan-500/30 scale-102'
                  : isNodeActive
                  ? `${n.color.border} ring-2 scale-102`
                  : 'border-slate-800/90 hover:border-slate-700'
              }`}
              style={{
                left: `${n.x}px`,
                top: `${n.y}px`,
                width: `${n.width}px`,
                height: `${n.height}px`,
                boxShadow: isNodeActive ? `0 0 20px ${n.color.glow}` : undefined,
              }}
            >
              {/* Connector Handles */}
              <div
                className={`absolute -left-1.5 top-1/2 -translate-y-1/2 w-3 h-3 rounded-full border-2 bg-slate-900 transition-colors ${
                  isNodeActive ? 'border-cyan-400 bg-cyan-950 scale-110' : 'border-slate-600'
                }`}
              />
              <div
                className={`absolute -right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 rounded-full border-2 bg-slate-900 transition-colors ${
                  isNodeActive ? 'border-emerald-400 bg-emerald-950 scale-110' : 'border-slate-600'
                }`}
              />

              {/* Node Header */}
              <div className="px-2 py-1.5 border-b border-slate-800/80 flex items-center justify-between bg-slate-950/70 rounded-t-xl">
                <div className="flex items-center gap-1.5 min-w-0">
                  <div
                    className="w-4 h-4 rounded flex items-center justify-center shrink-0 border border-slate-700/80"
                    style={{ backgroundColor: `${n.color.primary}20` }}
                  >
                    {n.category === 'SOURCE' && <Cpu size={9} className={n.color.text} />}
                    {n.category === 'TRANSFORM' && <Sliders size={9} className={n.color.text} />}
                    {n.category === 'ACTIVATION' && <Activity size={9} className={n.color.text} />}
                    {n.category === 'DEEP' && <Layers size={9} className={n.color.text} />}
                    {n.category === 'CLASSIFIER' && <Sparkles size={9} className={n.color.text} />}
                    {n.category === 'LOSS' && <TrendingDown size={9} className={n.color.text} />}
                    {n.category === 'OPTIMIZER' && <RefreshCw size={9} className={n.color.text} />}
                  </div>
                  <div className="flex flex-col min-w-0">
                    <span className="text-[7px] font-mono font-bold uppercase tracking-wider text-slate-400">
                      {n.category}
                    </span>
                    <span className="text-[10px] font-bold text-slate-100 truncate">
                      {n.title}
                    </span>
                  </div>
                </div>

                <span
                  className={`text-[8px] font-mono px-1 py-0.2 rounded border shrink-0 ${
                    isNodeActive
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50 font-bold'
                      : 'bg-slate-800 text-slate-400 border-slate-700'
                  }`}
                >
                  {isNodeActive ? 'ACTIVE' : n.badge}
                </span>
              </div>

              {/* Node Body Data */}
              <div className="p-1.5 flex-1 flex flex-col justify-between text-xs overflow-hidden">
                {/* Node 1: Input Tensor */}
                {n.id === 'node-input' && (
                  <div className="flex flex-col gap-1 h-full justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-9 h-9 rounded border border-slate-800 bg-black flex items-center justify-center overflow-hidden shrink-0">
                        <div className="grid grid-cols-4 gap-0.5 w-7 h-7 p-0.5">
                          {Array.from({ length: 16 }).map((_, i) => (
                            <div
                              key={i}
                              className="rounded-xs"
                              style={{
                                backgroundColor: `rgba(6, 182, 212, ${Math.min(
                                  1,
                                  (currentPixels[i * 45] || 0) * 1.5
                                )})`,
                              }}
                            />
                          ))}
                        </div>
                      </div>
                      <div className="flex flex-col text-[8.5px] text-slate-400 font-mono leading-tight">
                        <div>
                          Signal: <span className="text-cyan-300 font-bold">{activePixelsCount} px</span>
                        </div>
                        <div>Dim: {config.inputSize}</div>
                      </div>
                    </div>
                    <div className="p-0.5 rounded bg-slate-950 border border-slate-800 text-[8.5px] font-mono text-cyan-300 text-center truncate">
                      x ∈ ℝ^{config.inputSize}×1
                    </div>
                  </div>
                )}

                {/* Node 2: Dense Synapses */}
                {n.id === 'node-synapse' && (
                  <div className="flex flex-col gap-1 h-full justify-between">
                    <div className="text-[8.5px] text-slate-400 font-mono flex flex-col gap-0.5">
                      <div className="flex justify-between">
                        <span>Weights W:</span>
                        <span className="text-blue-300 font-bold">
                          {(config.inputSize * 64).toLocaleString()}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Biases b:</span>
                        <span className={config.useBias ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
                          {config.useBias ? '64 ON' : 'OFF'}
                        </span>
                      </div>
                    </div>
                    <div className="p-0.5 rounded bg-slate-950 border border-slate-800 text-[8.5px] font-mono text-blue-300 text-center truncate">
                      z = W·x + b
                    </div>
                  </div>
                )}

                {/* Node 3: Activation */}
                {n.id === 'node-activation' && (
                  <div className="flex flex-col gap-0.5 h-full justify-between">
                    <div className="flex items-center justify-center h-10 w-full bg-slate-950/60 rounded border border-slate-800/80 p-0.5">
                      <svg viewBox="0 0 100 36" className="w-full h-full overflow-visible">
                        <line x1="5" y1="22" x2="95" y2="22" stroke="#334155" strokeWidth="1" />
                        <line x1="50" y1="3" x2="50" y2="33" stroke="#334155" strokeWidth="1" />
                        <path
                          d={(() => {
                            const pts = [];
                            for (let px = 5; px <= 95; px += 2) {
                              const z = ((px - 50) / 40) * 3;
                              const val = activationInfo.fn(z);
                              const py = 22 - val * 9;
                              pts.push(`${px === 5 ? 'M' : 'L'} ${px} ${Math.max(2, Math.min(34, py))}`);
                            }
                            return pts.join(' ');
                          })()}
                          fill="none"
                          stroke="#10b981"
                          strokeWidth="1.8"
                        />
                      </svg>
                    </div>
                    <div className="p-0.5 rounded bg-slate-950 border border-slate-800 text-[8px] font-mono text-emerald-300 text-center truncate">
                      a = {activationInfo.name}(z)
                    </div>
                  </div>
                )}

                {/* Node 4: Deep Hidden Propagation */}
                {n.id === 'node-deep' && (
                  <div className="flex flex-col gap-1 h-full justify-between">
                    <div className="text-[8.5px] text-slate-400 font-mono flex flex-col gap-0.5">
                      <div>Decomposition:</div>
                      <div className="text-[8px] text-slate-300 bg-slate-950/70 p-0.5 rounded border border-slate-800 truncate">
                        Strokes → Loops → Shapes
                      </div>
                    </div>
                    <div className="p-0.5 rounded bg-slate-950 border border-slate-800 text-[8px] font-mono text-violet-300 text-center truncate">
                      z^[l] = W^[l]·a^[l-1] + b^[l]
                    </div>
                  </div>
                )}

                {/* Node 5: Softmax Classification */}
                {n.id === 'node-softmax' && (
                  <div className="flex flex-col gap-1 h-full justify-between">
                    <div className="flex items-center justify-between text-[9px] font-mono">
                      <span className="text-slate-400">Class:</span>
                      <span className="text-amber-400 font-bold">
                        Digit {predictedDigit} ({Math.round(confidence * 100)}%)
                      </span>
                    </div>
                    <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden border border-slate-800">
                      <div
                        className="bg-amber-400 h-full rounded-full transition-all duration-200"
                        style={{ width: `${Math.max(5, Math.round(confidence * 100))}%` }}
                      />
                    </div>
                    <div className="p-0.5 rounded bg-slate-950 border border-slate-800 text-[8px] font-mono text-amber-300 text-center truncate">
                      ŷ_k = exp(z_k) / ∑exp(z_j)
                    </div>
                  </div>
                )}

                {/* Node 6: Cross-Entropy Loss */}
                {n.id === 'node-loss' && (
                  <div className="flex flex-col gap-1 h-full justify-between">
                    <div className="flex justify-between text-[8.5px] font-mono">
                      <span className="text-slate-400">Loss L:</span>
                      <span className="text-rose-400 font-bold">
                        {trainingStats.currentLoss > 0 ? trainingStats.currentLoss.toFixed(4) : '0.0412'}
                      </span>
                    </div>
                    <div className="flex justify-between text-[8.5px] font-mono">
                      <span className="text-slate-400">Target:</span>
                      <span className="text-cyan-300 font-bold">Digit {targetDigit}</span>
                    </div>
                    <div className="p-0.5 rounded bg-slate-950 border border-slate-800 text-[8px] font-mono text-rose-300 text-center truncate">
                      δ^[L] = ŷ - y
                    </div>
                  </div>
                )}

                {/* Node 7: Momentum Optimizer */}
                {n.id === 'node-optimizer' && (
                  <div className="flex flex-col gap-1 h-full justify-between">
                    <div className="grid grid-cols-2 gap-1 text-[8.5px] font-mono">
                      <div className="p-0.5 rounded bg-slate-950 border border-slate-800 flex justify-between">
                        <span className="text-slate-500">LR α:</span>
                        <span className="text-cyan-300 font-bold">{config.learningRate.toFixed(2)}</span>
                      </div>
                      <div className="p-0.5 rounded bg-slate-950 border border-slate-800 flex justify-between">
                        <span className="text-slate-500">Mom β:</span>
                        <span className="text-emerald-300 font-bold">{config.momentum.toFixed(2)}</span>
                      </div>
                    </div>
                    <div className="p-0.5 rounded bg-slate-950 border border-slate-800 text-[8px] font-mono text-fuchsia-300 text-center truncate">
                      vt = β·v_{'{t-1}'} + (1-β)α·∇L
                    </div>
                  </div>
                )}
              </div>

              {/* Node Footer */}
              <div className="px-2 py-0.5 bg-slate-950/80 border-t border-slate-800/60 rounded-b-xl flex items-center justify-between text-[7.5px] font-mono text-slate-500">
                <span>Inspect</span>
                <span className="text-slate-400">#{n.index + 1}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* 3. COMPACT FLOATING INSPECTOR DRAWER (Only when user taps/clicks a node) */}
      {selectedNodeId && (
        <div className="absolute bottom-2 left-2 right-2 z-40 bg-slate-900/95 backdrop-blur-md border border-slate-800 rounded-xl p-2.5 shadow-2xl flex flex-col gap-1 animate-in slide-in-from-bottom duration-150">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-[10.5px] font-bold font-mono text-slate-200">
                Node Inspector: {nodeMap.get(selectedNodeId)?.title}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setSelectedNodeId(null)}
              className="text-[9.5px] text-slate-400 hover:text-white px-1.5 py-0.2 rounded bg-slate-800 hover:bg-slate-700 cursor-pointer"
            >
              Close ✕
            </button>
          </div>

          <div className="text-xs text-slate-300">
            {selectedNodeId === 'node-input' && (
              <div className="flex flex-col sm:flex-row items-center justify-between gap-2 p-1.5 rounded bg-slate-950 border border-slate-800 text-[10.5px]">
                <div className="overflow-x-auto no-scrollbar">
                  <MathFormula
                    math={`\\mathbf{x} \\in \\mathbb{R}^{N \\times 1}, \\quad N = ${config.inputSize}`}
                    displayMode={true}
                  />
                </div>
                <div className="text-[9.5px] text-slate-400">
                  Drawn pixels are normalized into a 1D vector between 0.0 and 1.0, centered by center-of-mass.
                </div>
              </div>
            )}

            {selectedNodeId === 'node-synapse' && (
              <div className="flex flex-col sm:flex-row items-center justify-between gap-2 p-1.5 rounded bg-slate-950 border border-slate-800 text-[10.5px]">
                <div className="overflow-x-auto no-scrollbar">
                  <MathFormula
                    math="\mathbf{z}^{[1]} = W^{[1]} \mathbf{x} + \mathbf{b}^{[1]}"
                    displayMode={true}
                  />
                </div>
                <div className="text-[9.5px] text-slate-400">
                  Matrix multiplication using 128-bit SIMD vectorization to compute synaptic linear sums.
                </div>
              </div>
            )}

            {selectedNodeId === 'node-activation' && (
              <div className="flex flex-col sm:flex-row items-center justify-between gap-2 p-1.5 rounded bg-slate-950 border border-slate-800 text-[10.5px]">
                <div className="overflow-x-auto no-scrollbar">
                  <MathFormula
                    math={`\\mathbf{a}^{[1]} = f(\\mathbf{z}^{[1]}), \\quad ${activationInfo.formula}`}
                    displayMode={true}
                  />
                </div>
                <div className="text-[9.5px] text-slate-400">
                  Applies {activationInfo.name} non-linearity. Derivative for backpropagation: {activationInfo.derivative}.
                </div>
              </div>
            )}

            {selectedNodeId === 'node-deep' && (
              <div className="flex flex-col sm:flex-row items-center justify-between gap-2 p-1.5 rounded bg-slate-950 border border-slate-800 text-[10.5px]">
                <div className="overflow-x-auto no-scrollbar">
                  <MathFormula
                    math="\mathbf{z}^{[l]} = W^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}"
                    displayMode={true}
                  />
                </div>
                <div className="text-[9.5px] text-slate-400">
                  Deep propagation across {config.hiddenLayers.length} hidden layers assembling features into digits.
                </div>
              </div>
            )}

            {selectedNodeId === 'node-softmax' && (
              <div className="flex flex-col sm:flex-row items-center justify-between gap-2 p-1.5 rounded bg-slate-950 border border-slate-800 text-[10.5px]">
                <div className="overflow-x-auto no-scrollbar">
                  <MathFormula
                    math="\hat{y}_k = \\frac{e^{z_k}}{\\sum_{j=0}^{9} e^{z_j}}, \\quad \\sum_{k=0}^{9} \\hat{y}_k = 1.0"
                    displayMode={true}
                  />
                </div>
                <div className="text-[9.5px] text-slate-400">
                  Normalizes final output logits into probabilities. Winning class: Digit {predictedDigit} ({Math.round(confidence * 100)}%).
                </div>
              </div>
            )}

            {selectedNodeId === 'node-loss' && (
              <div className="flex flex-col sm:flex-row items-center justify-between gap-2 p-1.5 rounded bg-slate-950 border border-slate-800 text-[10.5px]">
                <div className="overflow-x-auto no-scrollbar">
                  <MathFormula
                    math="\mathcal{L} = -\\sum_{k=0}^{9} y_k \\ln(\\hat{y}_k), \\quad \\delta^{[L]} = \\hat{\\mathbf{y}} - \\mathbf{y}"
                    displayMode={true}
                  />
                </div>
                <div className="text-[9.5px] text-slate-400">
                  Cross-Entropy loss against target digit {targetDigit}. Derivative simplifies to (ŷ - y). Current loss: {trainingStats.currentLoss.toFixed(4)}.
                </div>
              </div>
            )}

            {selectedNodeId === 'node-optimizer' && (
              <div className="flex flex-col sm:flex-row items-center justify-between gap-2 p-1.5 rounded bg-slate-950 border border-slate-800 text-[10.5px]">
                <div className="overflow-x-auto no-scrollbar">
                  <MathFormula
                    math="\mathbf{v}_t = \\beta \\mathbf{v}_{t-1} + (1-\\beta)\\alpha \\frac{\\partial \\mathcal{L}}{\\partial W}, \\quad W \\leftarrow W - \\mathbf{v}_t"
                    displayMode={true}
                  />
                </div>
                <div className="text-[9.5px] text-slate-400">
                  Caches directional velocity to accelerate gradient descent and update synaptic weights.
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
