import React from 'react';
import { Layers, Plus, Trash2, Zap, Save, Upload, Sparkles, RotateCcw } from 'lucide-react';
import { Button } from '@elsesourav/react-ui';
import { ParameterSlider } from '../Common/ParameterSlider';
import type { NetworkConfig, LayerSpec } from '../../types';

interface NetworkConfiguratorProps {
  config: NetworkConfig;
  onConfigChange: (newConfig: NetworkConfig) => void;
  onRebuildNetwork: () => void;
  onLearningRateChange: (lr: number) => void;
  onMomentumChange: (momentum: number) => void;
  onSaveToLocalStorage?: () => void;
  onLoadFromLocalStorage?: () => boolean;
  onLoadPretrained?: () => Promise<boolean>;
  onResetWeights?: () => void;
  disabled?: boolean;
}

export const NetworkConfigurator: React.FC<NetworkConfiguratorProps> = ({
  config,
  onConfigChange,
  onRebuildNetwork,
  onLearningRateChange,
  onMomentumChange,
  onSaveToLocalStorage,
  onLoadFromLocalStorage,
  onLoadPretrained,
  onResetWeights,
  disabled = false,
}) => {
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

  const handlePixelChange = (res: number) => {
    onConfigChange({
      ...config,
      pixelResolution: res,
      inputSize: res * res,
    });
  };

  const handleAddLayer = () => {
    const newLayer: LayerSpec = {
      id: Math.random().toString(36).substring(2, 7),
      neurons: 64,
    };
    onConfigChange({
      ...config,
      hiddenLayers: [...config.hiddenLayers, newLayer],
    });
  };

  const handleRemoveLayer = (index: number) => {
    if (config.hiddenLayers.length <= 1) return;
    onConfigChange({
      ...config,
      hiddenLayers: config.hiddenLayers.filter((_, i) => i !== index),
    });
  };

  const handleNeuronsDelta = (index: number, delta: number) => {
    const updated = config.hiddenLayers.map((l, i) =>
      i === index ? { ...l, neurons: Math.max(8, Math.min(256, l.neurons + delta)) } : l
    );
    onConfigChange({ ...config, hiddenLayers: updated });
  };

  const handleSetNeurons = (index: number, count: number) => {
    const updated = config.hiddenLayers.map((l, i) =>
      i === index ? { ...l, neurons: count } : l
    );
    onConfigChange({ ...config, hiddenLayers: updated });
  };

  return (
    <div className="flex flex-col gap-3 p-3 bg-slate-900/90 rounded-xl border border-slate-800 shadow-md">
      {/* 1. Header with Parameter Count */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-1.5">
          <Layers className="text-cyan-400" size={15} />
          <span className="text-xs font-semibold text-slate-200">Architecture & Hyperparameters</span>
        </div>
        <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/70 border border-cyan-800/40 px-2 py-0.5 rounded-full font-semibold">
          {calculateTotalParams().toLocaleString()} params
        </span>
      </div>

      {/* 2. Input Layer: Pixel Resolution Selection */}
      <div className="flex items-center justify-between gap-2 bg-slate-950/70 p-2 rounded-lg border border-slate-800/60">
        <span className="text-[11px] text-slate-400 font-medium">Input Layer:</span>
        <div className="flex items-center gap-1">
          {[14, 16, 20, 28].map((res) => (
            <button
              key={res}
              type="button"
              onClick={() => handlePixelChange(res)}
              className={`px-2 py-1 rounded text-[10px] font-mono transition-all border ${
                config.pixelResolution === res
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50 font-bold'
                  : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
            >
              {res}×{res}
            </button>
          ))}
        </div>
      </div>

      {/* 3. Hidden Layers (2 to N) with Quick Increment Buttons */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between text-[11px] font-medium text-slate-300">
          <span>Hidden Layers ({config.hiddenLayers.length}):</span>
          <button
            type="button"
            onClick={handleAddLayer}
            disabled={disabled}
            className="flex items-center gap-1 text-[10px] text-cyan-400 hover:text-cyan-300 bg-cyan-500/10 hover:bg-cyan-500/20 px-2 py-0.5 rounded border border-cyan-500/30 transition-all cursor-pointer"
          >
            <Plus size={11} /> Add Layer
          </button>
        </div>

        <div className="flex flex-col gap-1.5 max-h-[170px] overflow-y-auto pr-0.5">
          {config.hiddenLayers.map((layer, idx) => (
            <div
              key={layer.id}
              className="flex items-center justify-between gap-2 bg-slate-950/80 px-2.5 py-1.5 rounded-lg border border-slate-800/70"
            >
              <div className="flex items-center gap-1.5">
                <span className="text-[9px] font-mono text-cyan-400 bg-cyan-950 px-1 py-0.2 rounded font-bold">
                  L{idx + 1}
                </span>
                <span className="text-[11px] text-slate-300 font-medium w-12 font-mono">
                  {layer.neurons} nodes
                </span>
              </div>

              {/* Increment & Presets */}
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => handleNeuronsDelta(idx, -16)}
                  className="w-5 h-5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs flex items-center justify-center cursor-pointer"
                >
                  –
                </button>
                <button
                  type="button"
                  onClick={() => handleNeuronsDelta(idx, 16)}
                  className="w-5 h-5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs flex items-center justify-center cursor-pointer"
                >
                  +
                </button>
                <div className="w-px h-3 bg-slate-800 mx-0.5"></div>
                {[32, 64, 128].map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => handleSetNeurons(idx, s)}
                    className={`px-1.5 py-0.5 rounded text-[9px] font-mono ${
                      layer.neurons === s
                        ? 'bg-cyan-500/30 text-cyan-300 font-bold'
                        : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    {s}
                  </button>
                ))}
                <button
                  type="button"
                  onClick={() => handleRemoveLayer(idx)}
                  disabled={config.hiddenLayers.length <= 1}
                  className="text-slate-600 hover:text-rose-400 disabled:opacity-20 p-0.5 ml-1 transition-all cursor-pointer"
                  title="Remove layer"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 4. Hyperparameters with ParameterSlider from @elsesourav/react-ui */}
      <div className="flex flex-col gap-2.5 bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/60 text-[11px]">
        {/* LR */}
        <ParameterSlider
          label="Learning Rate"
          value={config.learningRate}
          onChange={(val) => {
            onConfigChange({ ...config, learningRate: val });
            onLearningRateChange(val);
          }}
          min={0.01}
          max={1.5}
          step={0.01}
          ticks={7}
          format={(v) => v.toFixed(2)}
          className="w-full bg-slate-950 border-slate-800"
        />

        {/* Momentum */}
        <ParameterSlider
          label="Momentum"
          value={config.momentum}
          onChange={(val) => {
            onConfigChange({ ...config, momentum: val });
            onMomentumChange(val);
          }}
          min={0.0}
          max={0.98}
          step={0.02}
          ticks={7}
          format={(v) => v.toFixed(2)}
          className="w-full bg-slate-950 border-slate-800"
        />

        {/* Activation */}
        <div className="col-span-2 flex items-center justify-between pt-1 border-t border-slate-800/60">
          <span className="text-slate-400 text-[10px]">Activation:</span>
          <div className="flex items-center gap-1">
            {(['leaky_relu', 'relu', 'sigmoid', 'tanh'] as const).map((act) => (
              <button
                key={act}
                type="button"
                onClick={() => onConfigChange({ ...config, hiddenActivation: act })}
                className={`px-1.5 py-0.5 rounded text-[9px] font-mono border ${
                  config.hiddenActivation === act
                    ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50 font-bold'
                    : 'bg-slate-900 text-slate-500 border-slate-800 hover:text-slate-300'
                }`}
              >
                {act}
              </button>
            ))}
          </div>
        </div>

        {/* Layer Biases Toggle */}
        <div className="col-span-2 flex items-center justify-between pt-1 border-t border-slate-800/60">
          <div className="flex flex-col">
            <span className="text-slate-300 text-[10px] font-medium">Layer Biases (b):</span>
            <span className="text-[9px] text-slate-500">Trainable bias offset per neuron</span>
          </div>
          <button
            type="button"
            onClick={() => onConfigChange({ ...config, useBias: !config.useBias })}
            className={`px-2.5 py-0.5 rounded-md text-[10px] font-mono border transition-all cursor-pointer ${
              config.useBias
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50 font-bold shadow-sm shadow-emerald-500/20'
                : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
          >
            {config.useBias ? 'Enabled (ON)' : 'Disabled (OFF)'}
          </button>
        </div>

        {/* Auto-Fit After Drawing Toggle */}
        <div className="col-span-2 flex items-center justify-between pt-1.5 border-t border-slate-800/60">
          <div className="flex flex-col">
            <span className="text-slate-300 text-[10px] font-medium">Auto-Fit on Draw:</span>
            <span className="text-[9px] text-slate-500">Auto-center & scale digit after drawing</span>
          </div>
          <button
            type="button"
            data-testid="toggle-autofit"
            onClick={() => onConfigChange({ ...config, autoFitOnDraw: config.autoFitOnDraw === false ? true : false })}
            className={`px-2.5 py-0.5 rounded-md text-[10px] font-mono border transition-all cursor-pointer ${
              config.autoFitOnDraw !== false
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50 font-bold shadow-sm shadow-cyan-500/20'
                : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
          >
            {config.autoFitOnDraw !== false ? 'Enabled (ON)' : 'Disabled (OFF)'}
          </button>
        </div>
      </div>

      {/* 5. Rebuild Button & Compact Quick Actions */}
      <div className="flex flex-col gap-1.5 pt-1">
        <button
          type="button"
          onClick={onRebuildNetwork}
          disabled={disabled}
          className="w-full flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-semibold text-xs uppercase tracking-wider shadow-md shadow-cyan-500/10 active:scale-[0.99] transition-all cursor-pointer"
        >
          <Zap size={13} className="fill-slate-950" />
          Apply & Initialize Network
        </button>

        {/* Quick Model Utility Bar */}
        <div className="flex items-center justify-between gap-1 text-[10px] text-slate-400 pt-0.5">
          {onSaveToLocalStorage && (
            <button
              type="button"
              onClick={onSaveToLocalStorage}
              className="flex items-center gap-1 hover:text-cyan-300 transition-colors cursor-pointer"
            >
              <Save size={10} /> Save
            </button>
          )}
          {onLoadFromLocalStorage && (
            <button
              type="button"
              onClick={onLoadFromLocalStorage}
              className="flex items-center gap-1 hover:text-cyan-300 transition-colors cursor-pointer"
            >
              <Upload size={10} /> Load
            </button>
          )}
          {onLoadPretrained && (
            <button
              type="button"
              onClick={onLoadPretrained}
              className="flex items-center gap-1 text-cyan-400 hover:text-cyan-200 font-medium transition-colors cursor-pointer"
            >
              <Sparkles size={10} /> Pretrained
            </button>
          )}
          {onResetWeights && (
            <button
              type="button"
              onClick={onResetWeights}
              className="flex items-center gap-1 text-rose-400/80 hover:text-rose-300 transition-colors cursor-pointer"
            >
              <RotateCcw size={10} /> Reset
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
