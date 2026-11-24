import React, { useRef, useState } from 'react';
import { useToast } from '@elsesourav/react-ui';
import {
  Save,
  Download,
  Upload,
  RotateCcw,
  Sparkles,
  Trash2,
  Bookmark,
  Layers,
} from 'lucide-react';
import type { SerializedModel } from '../../wasm/types';
import type { SavedPreset, NetworkConfig } from '../../types';

interface ModelManagerProps {
  onSaveToLocalStorage: () => void;
  onLoadFromLocalStorage: () => boolean;
  onExportModel: () => void;
  onImportModel: (model: SerializedModel) => boolean;
  onLoadPretrained: () => Promise<boolean>;
  onResetWeights: () => void;
  namedPresets?: SavedPreset[];
  onSaveNamedPreset?: (name: string) => SavedPreset | null;
  onLoadNamedPreset?: (id: string) => boolean;
  onDeleteNamedPreset?: (id: string) => void;
  onApplyTemplate?: (template: Partial<NetworkConfig>) => void;
  disabled?: boolean;
}

export const ModelManager: React.FC<ModelManagerProps> = ({
  onSaveToLocalStorage,
  onLoadFromLocalStorage,
  onExportModel,
  onImportModel,
  onLoadPretrained,
  onResetWeights,
  namedPresets = [],
  onSaveNamedPreset,
  onLoadNamedPreset,
  onDeleteNamedPreset,
  onApplyTemplate,
  disabled = false,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isLoadingPretrained, setIsLoadingPretrained] = useState(false);
  const [presetNameInput, setPresetNameInput] = useState('');
  const { addToast } = useToast();

  const showToast = (msg: string, type: 'info' | 'success' | 'warning' | 'error' = 'success') => {
    addToast({
      title: msg,
      type,
      duration: 2500,
    });
  };

  const handleSaveNamed = (e: React.FormEvent) => {
    e.preventDefault();
    if (!onSaveNamedPreset) return;
    const name = presetNameInput.trim() || `Preset ${namedPresets.length + 1}`;
    const saved = onSaveNamedPreset(name);
    if (saved) {
      setPresetNameInput('');
      showToast(`Saved preset "${name}"!`);
    }
  };

  const handleLoadPreset = (id: string, name: string) => {
    if (!onLoadNamedPreset) return;
    const ok = onLoadNamedPreset(id);
    if (ok) {
      showToast(`Loaded preset "${name}"!`);
    } else {
      showToast(`Failed to load preset "${name}".`);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        const ok = onImportModel(parsed);
        if (ok) {
          showToast('Model imported successfully!');
        } else {
          showToast('Failed to import model: structure incompatible.');
        }
      } catch (err) {
        showToast('Invalid JSON file format.');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleLoadPretrainedClick = async () => {
    setIsLoadingPretrained(true);
    try {
      const ok = await onLoadPretrained();
      if (ok) {
        showToast('Pre-trained model loaded successfully!');
      } else {
        showToast('Pre-trained model not found or incompatible.');
      }
    } finally {
      setIsLoadingPretrained(false);
    }
  };

  return (
    <div className="flex flex-col gap-3 p-3 sm:p-4 bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl text-xs">
      {/* 1. Header with Persistence Indicator */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <span className="font-semibold text-slate-200 flex items-center gap-1.5">
          <Bookmark size={15} className="text-cyan-400" />
          Model Storage & Presets
        </span>
      </div>

      {/* 2. Save Custom Named Preset */}
      <form onSubmit={handleSaveNamed} className="flex flex-col gap-1.5 bg-slate-950/70 p-2.5 rounded-xl border border-slate-800/80">
        <span className="text-[11px] font-medium text-slate-300 flex items-center gap-1">
          <Save size={12} className="text-cyan-400" />
          Save Current Model as Named Preset
        </span>
        <div className="flex items-center gap-1.5">
          <input
            type="text"
            placeholder="e.g. 14x14 Fast Net, MNIST Deep..."
            value={presetNameInput}
            onChange={(e) => setPresetNameInput(e.target.value)}
            disabled={disabled}
            className="flex-1 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
          <button
            type="submit"
            disabled={disabled}
            className="px-3 py-1.5 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/50 font-semibold transition-all cursor-pointer flex-shrink-0"
          >
            Save
          </button>
        </div>
      </form>

      {/* 3. Saved Custom Presets List */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between text-[11px] font-medium text-slate-300">
          <span>Saved Presets ({namedPresets.length}):</span>
        </div>

        {namedPresets.length === 0 ? (
          <div className="text-[11px] text-slate-500 italic p-3 text-center bg-slate-950/40 rounded-xl border border-slate-800/40">
            No custom presets saved yet. Name and save your network configuration above!
          </div>
        ) : (
          <div className="flex flex-col gap-1.5 max-h-[190px] overflow-y-auto pr-0.5">
            {namedPresets.map((preset) => (
              <div
                key={preset.id}
                className="flex items-center justify-between gap-2 p-2 rounded-xl bg-slate-950/80 border border-slate-800/80 hover:border-slate-700 transition-all"
              >
                <div className="flex flex-col min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="font-semibold text-slate-200 truncate">{preset.name}</span>
                    <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800/40">
                      {preset.config.pixelResolution}×{preset.config.pixelResolution}
                    </span>
                  </div>
                  <div className="text-[10px] font-mono text-slate-400 truncate">
                    [{preset.config.inputSize}, {preset.config.hiddenLayers.map((l) => l.neurons).join(', ')}, {preset.config.outputSize}]
                    {preset.config.useBias ? ' • Bias ON' : ' • Bias OFF'}
                  </div>
                </div>

                <div className="flex items-center gap-1 flex-shrink-0">
                  <button
                    type="button"
                    onClick={() => handleLoadPreset(preset.id, preset.name)}
                    className="px-2 py-1 rounded-md text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30 border border-emerald-500/40 transition-all cursor-pointer"
                  >
                    Load
                  </button>
                  <button
                    type="button"
                    onClick={() => onDeleteNamedPreset?.(preset.id)}
                    className="p-1 rounded-md text-slate-500 hover:text-rose-400 hover:bg-rose-950/30 transition-all cursor-pointer"
                    title="Delete preset"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 4. Quick Architecture Presets */}
      <div className="flex flex-col gap-1.5 pt-1 border-t border-slate-800/70">
        <span className="text-[11px] font-medium text-slate-400">Quick-Start Architectures:</span>
        <div className="grid grid-cols-3 gap-1.5 text-[10px] font-mono">
          <button
            type="button"
            onClick={() =>
              onApplyTemplate?.({
                pixelResolution: 28,
                inputSize: 784,
                hiddenLayers: [
                  { id: '1', neurons: 64 },
                  { id: '2', neurons: 64 },
                ],
                useBias: true,
              })
            }
            className="p-1.5 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 text-center transition-all cursor-pointer"
          >
            <div className="font-bold text-cyan-400">MNIST 28×28</div>
            <div className="text-slate-500 text-[9px]">[64, 64]</div>
          </button>

          <button
            type="button"
            onClick={() =>
              onApplyTemplate?.({
                pixelResolution: 14,
                inputSize: 196,
                hiddenLayers: [{ id: '1', neurons: 32 }],
                useBias: true,
              })
            }
            className="p-1.5 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 text-center transition-all cursor-pointer"
          >
            <div className="font-bold text-emerald-400">Light 14×14</div>
            <div className="text-slate-500 text-[9px]">[32]</div>
          </button>

          <button
            type="button"
            onClick={() =>
              onApplyTemplate?.({
                pixelResolution: 20,
                inputSize: 400,
                hiddenLayers: [
                  { id: '1', neurons: 96 },
                  { id: '2', neurons: 48 },
                  { id: '3', neurons: 24 },
                ],
                useBias: true,
              })
            }
            className="p-1.5 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 text-center transition-all cursor-pointer"
          >
            <div className="font-bold text-purple-400">Deep 20×20</div>
            <div className="text-slate-500 text-[9px]">[96, 48, 24]</div>
          </button>
        </div>
      </div>

      {/* 5. Standard Storage Actions & File Export/Import */}
      <div className="grid grid-cols-2 gap-2 pt-1 border-t border-slate-800/70">
        <button
          type="button"
          disabled={disabled}
          onClick={onExportModel}
          className="flex items-center justify-center gap-1.5 py-1.5 px-2.5 rounded-xl bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 font-medium transition-all cursor-pointer"
        >
          <Download size={13} className="text-emerald-400" />
          Export JSON
        </button>

        <button
          type="button"
          disabled={disabled}
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center justify-center gap-1.5 py-1.5 px-2.5 rounded-xl bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 font-medium transition-all cursor-pointer"
        >
          <Upload size={13} className="text-emerald-400" />
          Import JSON
        </button>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".json"
          className="hidden"
        />

        {/* Load Pre-trained weights */}
        <button
          type="button"
          disabled={disabled || isLoadingPretrained}
          onClick={handleLoadPretrainedClick}
          className="col-span-2 flex items-center justify-center gap-1.5 py-2 px-3 rounded-xl bg-gradient-to-r from-cyan-950 to-blue-950 hover:from-cyan-900 hover:to-blue-900 border border-cyan-700/40 text-cyan-200 font-semibold transition-all cursor-pointer"
        >
          <Sparkles size={13} className="text-cyan-400" />
          {isLoadingPretrained ? 'Loading Pre-trained...' : 'Load Pre-trained Model (28×28)'}
        </button>

        {/* Reset Weights */}
        <button
          type="button"
          disabled={disabled}
          onClick={onResetWeights}
          className="col-span-2 flex items-center justify-center gap-1.5 py-1 px-3 rounded-xl text-rose-400/80 hover:text-rose-300 hover:bg-rose-950/20 text-[11px] transition-all cursor-pointer"
        >
          <RotateCcw size={11} />
          Reset Weights to Random
        </button>
      </div>
    </div>
  );
};
