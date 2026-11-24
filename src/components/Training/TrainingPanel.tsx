import React from 'react';
import { Play, Pause, StepForward, Zap, CheckCircle2, Globe, ChevronLeft, ChevronRight, Shuffle } from 'lucide-react';
import type { PredictionResult } from '../../types';

interface TrainingPanelProps {
  prediction: PredictionResult | null;
  onTrainDigit: (digit: number) => void;
  isLiveTraining: boolean;
  onToggleLiveTraining: () => void;
  onStepTraining: () => void;
  trainSpeed: number; // 1 to 20
  onChangeSpeed: (speed: number) => void;
  batchTrained?: number;
  targetSamples?: number;
  onChangeTargetSamples?: (target: number) => void;
  datasetSource?: 'online' | 'synthetic';
  onToggleDatasetSource?: () => void;
  totalOnlineSamples?: number;
  currentSampleInfo?: { id: number; label: number } | null;
  onPrevSample?: () => void;
  onNextSample?: () => void;
  showTrainingControls?: boolean;
  showNumpad?: boolean;
  disabled?: boolean;
}

export const TrainingPanel: React.FC<TrainingPanelProps> = ({
  prediction,
  onTrainDigit,
  isLiveTraining,
  onToggleLiveTraining,
  onStepTraining,
  trainSpeed,
  onChangeSpeed,
  batchTrained = 0,
  targetSamples = 100,
  onChangeTargetSamples,
  datasetSource = 'online',
  onToggleDatasetSource,
  totalOnlineSamples = 1000,
  currentSampleInfo = null,
  onPrevSample,
  onNextSample,
  showTrainingControls = true,
  showNumpad = true,
  disabled = false,
}) => {
  const predictedDigit = prediction?.digit ?? null;
  const isLoop = targetSamples === Infinity;
  const remaining = isLoop ? '∞' : Math.max(0, targetSamples - batchTrained);
  const progressPct = isLoop
    ? Math.min(100, ((batchTrained % Math.max(1, totalOnlineSamples)) / Math.max(1, totalOnlineSamples)) * 100)
    : Math.min(100, (batchTrained / targetSamples) * 100);

  return (
    <div className="flex flex-col gap-2 p-2.5 sm:p-3 bg-slate-900/90 rounded-xl border border-slate-800 shadow-md">
      {/* 1. Live Training Control Bar (Play/Pause, Step, Speed) */}
      {showTrainingControls && (
        <>
          <div className="flex items-center justify-between gap-2 pb-2 border-b border-slate-800/80">
            <div className="flex items-center gap-1.5">
              {/* Live Train Play / Pause Button */}
              <button
                type="button"
                data-testid="btn-toggle-live-train"
                onClick={onToggleLiveTraining}
                disabled={disabled}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer shadow-sm ${
                  isLiveTraining
                    ? 'bg-rose-500/25 text-rose-300 border border-rose-500/50 hover:bg-rose-500/35 animate-pulse'
                    : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30'
                }`}
                title="Auto Train Animation (Space)"
              >
                {isLiveTraining ? (
                  <>
                    <Pause size={12} className="fill-rose-300" />
                    <span>Pause</span>
                  </>
                ) : (
                  <>
                    <Play size={12} className="fill-emerald-300" />
                    <span>Live Train</span>
                  </>
                )}
              </button>

              {/* Single Step Button */}
              <button
                type="button"
                data-testid="btn-step-train"
                onClick={onStepTraining}
                disabled={disabled || isLiveTraining}
                className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium text-slate-300 bg-slate-950 border border-slate-800 hover:text-white hover:bg-slate-800 disabled:opacity-40 transition-all cursor-pointer"
                title="Train 1 Image Step"
              >
                <StepForward size={12} />
                <span>Step</span>
              </button>
            </div>

            {/* Speed Slider */}
            <div className="flex items-center gap-1.5 flex-1 max-w-[110px] justify-end">
              <Zap size={11} className="text-amber-400 flex-shrink-0" />
              <input
                type="range"
                min={1}
                max={20}
                step={1}
                value={trainSpeed}
                onChange={(e) => onChangeSpeed(Number(e.target.value))}
                className="w-14 h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-amber-400"
                title={`Training Speed: ${trainSpeed}x`}
              />
              <span className="text-[10px] font-mono text-amber-300 w-5 text-right font-medium">
                {trainSpeed}x
              </span>
            </div>
          </div>

          {/* 2. Dataset Tracker & Progress Section (Clean & Integrated) */}
          <div className="flex flex-col gap-1.5 px-2.5 py-2 bg-slate-950/60 rounded-lg border border-slate-800/50 text-xs">
            {/* Source Toggle & Sample ID */}
            <div className="flex items-center justify-between">
              <button
                type="button"
                onClick={onToggleDatasetSource}
                className="flex items-center gap-1 px-2 py-0.5 rounded-md bg-slate-900 border border-slate-800 hover:border-cyan-500/40 text-cyan-300 text-[10px] font-medium transition-all cursor-pointer"
                title="Toggle between Online MNIST and Synthetic digits"
              >
                <Globe size={10} className="text-cyan-400" />
                <span>{datasetSource === 'online' ? 'Online MNIST' : 'Synthetic'}</span>
                <span className="text-[9px] text-slate-400 font-mono">
                  ({totalOnlineSamples})
                </span>
              </button>

              {/* Sample Navigation Buttons */}
              {datasetSource === 'online' && (
                <div className="flex items-center gap-1 font-mono text-[10px] text-slate-400">
                  {currentSampleInfo && (
                    <span className="text-slate-300 mr-0.5">
                      #{currentSampleInfo.id + 1}
                      <span className="text-emerald-400 font-bold ml-1">
                        (Digit {currentSampleInfo.label})
                      </span>
                    </span>
                  )}
                  {onPrevSample && (
                    <button
                      type="button"
                      onClick={onPrevSample}
                      disabled={isLiveTraining}
                      className="w-4 h-4 rounded bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white flex items-center justify-center cursor-pointer disabled:opacity-30 border border-slate-800"
                      title="Previous Online Image"
                    >
                      <ChevronLeft size={10} />
                    </button>
                  )}
                  {onNextSample && (
                    <button
                      type="button"
                      onClick={onNextSample}
                      disabled={isLiveTraining}
                      className="w-4 h-4 rounded bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white flex items-center justify-center cursor-pointer disabled:opacity-30 border border-slate-800"
                      title="Next Online Image"
                    >
                      <ChevronRight size={10} />
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Progress Counts */}
            <div className="flex items-center justify-between pt-0.5 text-[11px] font-mono">
              <div className="flex items-center gap-1">
                <span className="text-emerald-400 font-bold">{batchTrained}</span>
                <span className="text-slate-600">/</span>
                <span className="text-slate-300">{isLoop ? '∞' : targetSamples}</span>
              </div>
              <span className="text-[10px] text-cyan-400 font-medium">
                {isLoop
                  ? `Epoch ${Math.floor(batchTrained / Math.max(1, totalOnlineSamples)) + 1}`
                  : `${remaining} left`}
              </span>
            </div>

            {/* Progress Bar */}
            <div className="w-full h-1 bg-slate-800/80 rounded-full overflow-hidden">
              <div
                className={`h-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-150 rounded-full ${
                  isLoop ? 'animate-pulse' : ''
                }`}
                style={{ width: `${progressPct}%` }}
              />
            </div>

            {/* Target Presets: Clean 6-Column Segmented Row */}
            <div className="grid grid-cols-6 gap-1 pt-0.5">
              {[
                { val: 100, label: '100' },
                { val: 500, label: '500' },
                { val: 1000, label: '1k' },
                { val: 2500, label: '2.5k' },
                { val: 5000, label: '5k' },
                { val: Infinity, label: '∞' },
              ].map(({ val, label }) => (
                <button
                  key={label}
                  type="button"
                  onClick={() => onChangeTargetSamples?.(val)}
                  className={`py-0.5 text-center rounded font-mono text-[9px] border transition-all cursor-pointer ${
                    targetSamples === val
                      ? 'bg-cyan-500/25 text-cyan-300 border-cyan-500/50 font-bold shadow-sm shadow-cyan-500/20'
                      : 'bg-slate-900/80 text-slate-400 border-slate-800 hover:text-slate-200 hover:border-slate-700'
                  }`}
                  title={`Target: ${label === '∞' ? 'Continuous Infinite Loop' : val} samples`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </>
      )}

      {/* 3. Big Clean Numpad with Live Percentage on Every Button (0-9) */}
      {showNumpad && (
        <div className="grid grid-cols-5 gap-1.5">
        {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => {
          const prob = prediction?.probabilities[digit] ?? 0;
          const pct = Math.round(prob * 100);
          const isWinner = prediction !== null && digit === predictedDigit && pct > 0;
          const hasPrediction = prediction !== null;

          return (
            <button
              key={digit}
              type="button"
              data-testid={`btn-train-digit-${digit}`}
              disabled={disabled}
              onClick={() => onTrainDigit(digit)}
              className={`relative overflow-hidden flex flex-col items-center justify-center h-12 rounded-xl font-mono transition-all cursor-pointer active:scale-95 border ${
                isWinner
                  ? 'bg-emerald-950/60 border-emerald-500 shadow-md shadow-emerald-500/20 text-emerald-300 ring-1 ring-emerald-500/40'
                  : 'bg-slate-950/90 border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white'
              }`}
              title={`Train digit ${digit} (Key: ${digit})`}
            >
              {/* Subtle Probability Fill from Bottom */}
              {pct > 0 && (
                <div
                  style={{ height: `${pct}%` }}
                  className={`absolute inset-x-0 bottom-0 pointer-events-none transition-all duration-150 ${
                    isWinner ? 'bg-emerald-500/25' : 'bg-cyan-500/10'
                  }`}
                />
              )}

              {/* Big Digit Number */}
              <span
                className={`text-lg sm:text-xl font-bold leading-none z-10 ${
                  isWinner ? 'text-emerald-300' : 'text-slate-100'
                }`}
              >
                {digit}
              </span>

              {/* Percentage Probability (Clean & visible when active, no messy dashes when idle) */}
              {hasPrediction && (
                <span
                  className={`text-[9px] font-mono font-medium leading-none z-10 mt-1 ${
                    isWinner
                      ? 'text-emerald-400 font-bold'
                      : pct > 5
                      ? 'text-cyan-400/90'
                      : 'text-slate-500/60'
                  }`}
                >
                  {pct}%
                </span>
              )}
            </button>
          );
        })}
      </div>
      )}
    </div>
  );
};
