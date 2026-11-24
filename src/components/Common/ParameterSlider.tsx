import React, { useRef, useState, useCallback } from 'react';

export interface ParameterSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  format?: (value: number) => string;
  ticks?: number;
  disabled?: boolean;
  className?: string;
}

export const ParameterSlider: React.FC<ParameterSliderProps> = ({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  unit = '',
  format,
  ticks = 9,
  disabled = false,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  // Round value to appropriate step precision
  const roundToStep = useCallback(
    (val: number): number => {
      const clamped = Math.min(max, Math.max(min, val));
      const stepsCount = Math.round((clamped - min) / step);
      const steppedVal = min + stepsCount * step;
      const stepStr = step.toString();
      const decimals = stepStr.includes('.') ? stepStr.split('.')[1].length : 0;
      return Number(steppedVal.toFixed(decimals));
    },
    [min, max, step]
  );

  const thumbWidth = 5;
  const paddingX = 14; // inset horizontal boundary for slider thumb travel

  const updateFromClientX = useCallback(
    (clientX: number) => {
      if (!containerRef.current || disabled) return;
      const rect = containerRef.current.getBoundingClientRect();
      const usableWidth = Math.max(1, rect.width - 2 * paddingX);
      const offsetX = clientX - rect.left - paddingX;
      const fraction = Math.min(1, Math.max(0, offsetX / usableWidth));
      const rawVal = min + fraction * (max - min);
      onChange(roundToStep(rawVal));
    },
    [disabled, min, max, roundToStep, onChange]
  );

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (disabled || (e.pointerType === 'mouse' && e.button !== 0)) return;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // Ignored if capture unsupported
    }
    e.currentTarget.focus({ preventScroll: true });
    setIsDragging(true);
    updateFromClientX(e.clientX);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDragging || disabled) return;
    updateFromClientX(e.clientX);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (isDragging) {
      setIsDragging(false);
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // Ignored
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    const mult = e.shiftKey ? 5 : 1;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
      e.preventDefault();
      onChange(roundToStep(value + step * mult));
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
      e.preventDefault();
      onChange(roundToStep(value - step * mult));
    } else if (e.key === 'Home') {
      e.preventDefault();
      onChange(min);
    } else if (e.key === 'End') {
      e.preventDefault();
      onChange(max);
    }
  };

  const range = max - min || 1;
  const progressRatio = Math.min(1, Math.max(0, (value - min) / range));
  const displayValue = format ? format(value) : `${value}${unit}`;

  return (
    <div
      ref={containerRef}
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label={label}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      aria-valuetext={displayValue}
      aria-disabled={disabled}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onKeyDown={handleKeyDown}
      className={`group relative flex items-center justify-between h-10 px-3.5 rounded-xl bg-slate-950/90 border border-slate-800 shadow-xs select-none cursor-ew-resize outline-none transition-all duration-150 overflow-hidden ${
        isDragging
          ? 'border-cyan-500/70 shadow-[0_0_12px_rgba(6,182,212,0.25)] ring-1 ring-cyan-500/30'
          : 'hover:border-slate-700'
      } ${disabled ? 'opacity-50 pointer-events-none cursor-not-allowed' : ''} ${className}`}
    >
      {/* 1. Background Tick Marks (Centered vertically on container midline) */}
      <div className="absolute inset-x-3.5 inset-y-0 flex items-center justify-between pointer-events-none z-0">
        {Array.from({ length: ticks }).map((_, idx) => (
          <span
            key={idx}
            className="w-[1px] h-2.5 rounded-full bg-slate-800/80 group-hover:bg-slate-700/80 transition-colors"
          />
        ))}
      </div>

      {/* 2. Text Labels (Centered vertically on container midline) */}
      <div className="relative z-10 w-full flex items-center justify-between pointer-events-none">
        <span className="text-xs font-medium text-slate-300 truncate max-w-[55%]">
          {label}
        </span>
        <span className="text-xs font-mono font-bold text-slate-100 tabular-nums shrink-0">
          {displayValue}
        </span>
      </div>

      {/* 3. Slider Thumb Bar (Guaranteed vertically centered in container flexbox layer) */}
      <div className="absolute inset-y-0 left-0 right-0 flex items-center pointer-events-none z-20">
        <div
          style={{
            left: `calc(${paddingX}px - ${thumbWidth / 2}px + ${progressRatio} * (100% - ${2 * paddingX}px))`,
          }}
          className={`relative h-6 w-[5px] rounded-full bg-white transition-all duration-75 ${
            isDragging
              ? 'scale-y-110 shadow-[0_0_12px_rgba(255,255,255,0.95)]'
              : 'shadow-[0_0_8px_rgba(255,255,255,0.7)] group-hover:shadow-[0_0_10px_rgba(255,255,255,0.85)]'
          }`}
        />
      </div>
    </div>
  );
};
