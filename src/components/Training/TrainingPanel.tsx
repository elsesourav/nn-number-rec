import React from 'react';
import { Play, Pause, StepForward, Globe, ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react';
import { Button, ParameterSlider } from '@elsesourav/react-ui';
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
  onResetToFirst?: () => void;
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
  onResetToFirst,
  showTrainingControls = true,
  showNumpad = true,
  disabled = false,
}) => {
  const predictedDigit = prediction?.digit ?? null;
  const effectiveTarget = Math.min(targetSamples, totalOnlineSamples);
  const remaining = Math.max(0, effectiveTarget - batchTrained);
  const progressPct = Math.min(100, (batchTrained / Math.max(1, effectiveTarget)) * 100);

  return (
    <div className="flex flex-col gap-2 p-2.5 sm:p-3 bg-slate-900/90 rounded-xl border border-slate-800 shadow-md">
      {/* 1. Live Training Control Bar (Play/Pause, Step, Speed) */}
      {showTrainingControls && (
        <>
          <div className="flex items-center justify-between gap-2 pb-2 border-b border-slate-800/80">
            <div className="flex items-center gap-1.5">
              {/* Live Train Play / Pause Button */}
              <Button
                size="sm"
                variant={isLiveTraining ? 'danger' : 'primary'}
                data-testid="btn-toggle-live-train"
                onClick={onToggleLiveTraining}
                disabled={disabled}
                leftIcon={
                  isLiveTraining ? (
                    <Pause size={12} className="fill-current shrink-0" />
                  ) : (
                    <Play size={12} className="fill-current shrink-0" />
                  )
                }
                className={`h-8 px-3 text-xs font-semibold inline-flex items-center gap-1.5 whitespace-nowrap [&>span]:inline-flex [&>span]:items-center [&>span]:gap-1.5 cursor-pointer ${
                  isLiveTraining
                    ? 'bg-rose-500 hover:bg-rose-600 text-white shadow-xs animate-pulse'
                    : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-xs'
                }`}
                title="Auto Train Animation (Space)"
              >
                <span>{isLiveTraining ? 'Pause' : 'Live Train'}</span>
              </Button>

              {/* Single Step Button */}
              <Button
                size="sm"
                variant="outline"
                data-testid="btn-step-train"
                onClick={onStepTraining}
                disabled={disabled || isLiveTraining}
                leftIcon={<StepForward size={12} className="shrink-0" />}
                className="h-8 px-2.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-950 border-slate-800 hover:bg-slate-800 inline-flex items-center gap-1.5 whitespace-nowrap [&>span]:inline-flex [&>span]:items-center [&>span]:gap-1.5 cursor-pointer"
                title="Train 1 Image Step"
              >
                <span>Step</span>
              </Button>
            </div>

            {/* Speed ParameterSlider */}
            <div className="flex-1 max-w-[145px]">
              <ParameterSlider
                label="Speed"
                value={trainSpeed}
                onChange={onChangeSpeed}
                min={1}
                max={20}
                step={1}
                unit="x"
                ticks={5}
                className="h-8 text-xs bg-slate-950 border-slate-800"
              />
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
                      data-testid="btn-prev-sample"
                      onClick={onPrevSample}
                      disabled={isLiveTraining}
                      className="w-4.5 h-4.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white flex items-center justify-center cursor-pointer disabled:opacity-30 border border-slate-800"
                      title="Previous Online Image & Train"
                    >
                      <ChevronLeft size={10} />
                    </button>
                  )}
                  {onNextSample && (
                    <button
                      type="button"
                      data-testid="btn-next-sample"
                      onClick={onNextSample}
                      disabled={isLiveTraining}
                      className="w-4.5 h-4.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white flex items-center justify-center cursor-pointer disabled:opacity-30 border border-slate-800"
                      title="Next Online Image & Train"
                    >
                      <ChevronRight size={10} />
                    </button>
                  )}
                  {onResetToFirst && (
                    <button
                      type="button"
                      data-testid="btn-reset-sample"
                      onClick={onResetToFirst}
                      disabled={isLiveTraining}
                      className="w-4.5 h-4.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-cyan-300 flex items-center justify-center cursor-pointer disabled:opacity-30 border border-slate-800"
                      title="Reset to First Image (#1)"
                    >
                      <RotateCcw size={9} />
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
                <span className="text-slate-300">{effectiveTarget}</span>
              </div>
              <span className="text-[10px] text-cyan-400 font-medium">
                {remaining === 0 ? 'Target Reached' : `${remaining} left`}
              </span>
            </div>

            {/* Progress Bar */}
            <div className="w-full h-1 bg-slate-800/80 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-150 rounded-full"
                style={{ width: `${progressPct}%` }}
              />
            </div>

            {/* Target Presets: Strictly based on available presets, maximum is All, NO infinite */}
            <div className="grid grid-cols-5 gap-1 pt-0.5">
              {[
                { val: 50, label: '50' },
                { val: 100, label: '100' },
                { val: 250, label: '250' },
                { val: 500, label: '500' },
                { val: totalOnlineSamples, label: `All (${totalOnlineSamples})` },
              ].map(({ val, label }) => (
                <button
                  key={label}
                  type="button"
                  onClick={() => onChangeTargetSamples?.(val)}
                  className={`py-0.5 text-center rounded font-mono text-[9px] border transition-all cursor-pointer ${
                    effectiveTarget === val
                      ? 'bg-cyan-500/25 text-cyan-300 border-cyan-500/50 font-bold shadow-sm shadow-cyan-500/20'
                      : 'bg-slate-900/80 text-slate-400 border-slate-800 hover:text-slate-200 hover:border-slate-700'
                  }`}
                  title={`Target: ${val} samples`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </>
      )}

      {/* 3. The 10 Section: Sleek 10-Class Probability Bars (0-9) - Replaces bulky numpad */}
      {showNumpad && (
        <div
          data-testid="ten-probability-section"
          className="bg-slate-950/90 rounded-xl border border-slate-800/90 p-2 shadow-inner flex flex-col gap-1"
        >
          <div className="grid grid-cols-10 gap-1 sm:gap-1.5 items-end h-16 pt-1">
            {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => {
              const prob = prediction?.probabilities[digit] ?? 0;
              const pct = Math.round(prob * 100);
              const isWinner = prediction !== null && digit === predictedDigit && pct > 0;

              return (
                <button
                  key={digit}
                  type="button"
                  data-testid={`btn-train-digit-${digit}`}
                  disabled={disabled}
                  onClick={() => onTrainDigit(digit)}
                  className={`group relative flex flex-col items-center justify-end h-full w-full rounded-lg transition-all cursor-pointer p-0.5 active:scale-95 select-none ${
                    isWinner
                      ? 'bg-emerald-950/40 ring-1 ring-emerald-500/50'
                      : 'hover:bg-slate-900/60'
                  }`}
                  title={`Digit ${digit}: ${pct}% (Click to train or press ${digit})`}
                >
                  {/* Vertical Probability Bar Container */}
                  <div className="relative w-full flex-1 max-h-10 bg-slate-900/90 rounded-t-md rounded-b-xs overflow-hidden flex flex-col justify-end border border-slate-800/60 group-hover:border-slate-700">
                    <div
                      style={{ height: `${pct}%` }}
                      className={`w-full transition-all duration-150 rounded-t-xs ${
                        isWinner
                          ? 'bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.7)]'
                          : pct > 0
                          ? 'bg-cyan-600/70'
                          : 'bg-transparent'
                      }`}
                    />
                  </div>

                  {/* Digit Label */}
                  <span
                    className={`mt-1 text-xs sm:text-sm font-mono transition-colors ${
                      isWinner
                        ? 'text-emerald-400 font-bold drop-shadow-[0_0_6px_rgba(52,211,153,0.8)]'
                        : 'text-slate-400 group-hover:text-slate-200'
                    }`}
                  >
                    {digit}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
