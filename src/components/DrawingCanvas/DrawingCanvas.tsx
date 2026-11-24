import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useImperativeHandle,
  forwardRef,
} from 'react';
import { PenTool, Eraser, Trash2, Sparkles, Maximize2 } from 'lucide-react';
import { Button } from '@elsesourav/react-ui';
import { processCanvasToInput } from '../../utils/imageProcessing';
import { renderSyntheticDigit } from '../../utils/digitDataset';
import { renderMnistImageToCanvas, adjustMnistStrokeWidth } from '../../utils/onlineDataset';

export interface DrawingCanvasHandle {
  clear: () => void;
  autoFit: () => void;
  drawSample: (digit: number, strokeWidth?: number) => Float32Array;
  drawMnistSample: (pixels784: Float32Array, strokeWidth?: number) => Float32Array;
  setTool: (tool: 'brush' | 'eraser') => void;
}

interface DrawingCanvasProps {
  pixelResolution: number; // e.g. 28 -> 28x28 = 784
  onPixelDataChange: (pixels: Float32Array) => void;
  onClear?: () => void;
  disabled?: boolean;
}

interface Point {
  x: number;
  y: number;
  pressure: number;
}

export const DrawingCanvas = forwardRef<DrawingCanvasHandle, DrawingCanvasProps>(
  ({ pixelResolution = 28, onPixelDataChange, onClear, disabled = false }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const offscreenRef = useRef<HTMLCanvasElement | null>(null);

    const [isDrawing, setIsDrawing] = useState(false);
    const [tool, setTool] = useState<'brush' | 'eraser'>('brush');
    const [brushSize, setBrushSize] = useState<number>(24);
    const [hasContent, setHasContent] = useState<boolean>(false);
    const pointsRef = useRef<Point[]>([]);

    // Track active sample on screen to dynamically adjust its stroke width when slider moves
    const currentSampleRef = useRef<{
      type: 'mnist' | 'synthetic';
      pixels?: Float32Array;
      digit?: number;
    } | null>(null);

    // Initialize or resize canvas preserving offscreen content
    const initCanvas = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Always run in authentic 28x28 pixel space for 1:1 MNIST neural input mapping
      if (canvas.width !== pixelResolution || canvas.height !== pixelResolution) {
        // Save current canvas to offscreen if it has content
        if (offscreenRef.current && hasContent) {
          canvas.width = pixelResolution;
          canvas.height = pixelResolution;
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(offscreenRef.current, 0, 0);
        } else {
          canvas.width = pixelResolution;
          canvas.height = pixelResolution;
          ctx.imageSmoothingEnabled = false;
          ctx.fillStyle = '#000000';
          ctx.fillRect(0, 0, pixelResolution, pixelResolution);
        }
      }
    }, [pixelResolution, hasContent]);

    useEffect(() => {
      initCanvas();
    }, [initCanvas]);

    // Save state to offscreen backup
    const backupToOffscreen = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      if (!offscreenRef.current) {
        offscreenRef.current = document.createElement('canvas');
      }
      offscreenRef.current.width = canvas.width;
      offscreenRef.current.height = canvas.height;
      const offCtx = offscreenRef.current.getContext('2d');
      if (offCtx) {
        offCtx.drawImage(canvas, 0, 0);
      }
    }, []);

    // Extract processed pixels: direct 1:1 pixel array
    const syncProcessedPixels = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return new Float32Array(pixelResolution * pixelResolution);

      const ctx = canvas.getContext('2d');
      if (!ctx) return new Float32Array(pixelResolution * pixelResolution);

      const imgData = ctx.getImageData(0, 0, pixelResolution, pixelResolution);
      const pixels = new Float32Array(pixelResolution * pixelResolution);
      let foundAny = false;
      for (let i = 0; i < pixels.length; i++) {
        const val = Math.max(imgData.data[i * 4], imgData.data[i * 4 + 1], imgData.data[i * 4 + 2]) / 255.0;
        pixels[i] = val;
        if (val > 0.05) foundAny = true;
      }
      setHasContent(foundAny);
      backupToOffscreen();
      onPixelDataChange(pixels);
      return pixels;
    }, [pixelResolution, onPixelDataChange, backupToOffscreen]);

    // Explicit Auto-fit & center hand drawing when user clicks [F]
    const autoFitDrawing = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const { pixels } = processCanvasToInput(canvas, {
        pixelResolution,
        centeringMethod: 'centerOfMass',
        autoFitSides: true,
      });

      // Render centered normalized pixels onto canvas
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, pixelResolution, pixelResolution);
        const imgData = ctx.createImageData(pixelResolution, pixelResolution);
        for (let i = 0; i < pixels.length; i++) {
          const v = Math.round(pixels[i] * 255);
          imgData.data[i * 4] = v;
          imgData.data[i * 4 + 1] = v;
          imgData.data[i * 4 + 2] = v;
          imgData.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
      }

      currentSampleRef.current = null;
      setHasContent(true);
      backupToOffscreen();
      onPixelDataChange(pixels);
    }, [pixelResolution, onPixelDataChange, backupToOffscreen]);

    const clearCanvas = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, pixelResolution, pixelResolution);

      if (offscreenRef.current) {
        const offCtx = offscreenRef.current.getContext('2d');
        offCtx?.fillRect(0, 0, pixelResolution, pixelResolution);
      }

      pointsRef.current = [];
      currentSampleRef.current = null;
      setHasContent(false);

      onPixelDataChange(new Float32Array(pixelResolution * pixelResolution));
      onClear?.();
    }, [pixelResolution, onPixelDataChange, onClear]);

    // Draw synthetic digit on canvas with dynamic stroke width
    const drawSample = useCallback(
      (digit: number, strokeWidth?: number) => {
        const canvas = canvasRef.current;
        if (!canvas) return new Float32Array(pixelResolution * pixelResolution);

        const ctx = canvas.getContext('2d');
        if (!ctx) return new Float32Array(pixelResolution * pixelResolution);

        const activeStroke = strokeWidth !== undefined ? strokeWidth : brushSize;
        currentSampleRef.current = { type: 'synthetic', digit };

        const off = document.createElement('canvas');
        off.width = 240;
        off.height = 240;
        const offCtx = off.getContext('2d');
        if (offCtx) {
          renderSyntheticDigit(offCtx, 240, 240, digit, activeStroke);
          ctx.fillStyle = '#000000';
          ctx.fillRect(0, 0, pixelResolution, pixelResolution);
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(off, 0, 0, pixelResolution, pixelResolution);
        }

        setHasContent(true);
        backupToOffscreen();

        const { pixels } = processCanvasToInput(canvas, {
          pixelResolution,
          centeringMethod: 'centerOfMass',
          autoFitSides: true,
        });

        onPixelDataChange(pixels);
        return pixels;
      },
      [pixelResolution, brushSize, onPixelDataChange, backupToOffscreen]
    );

    // Draw authentic online MNIST digit sample onto canvas with dynamic stroke width
    const drawMnistSample = useCallback(
      (pixels28x28: Float32Array, strokeWidth?: number) => {
        const canvas = canvasRef.current;
        if (!canvas) return new Float32Array(pixelResolution * pixelResolution);

        const ctx = canvas.getContext('2d');
        if (!ctx) return new Float32Array(pixelResolution * pixelResolution);

        const activeStroke = strokeWidth !== undefined ? strokeWidth : brushSize;
        currentSampleRef.current = { type: 'mnist', pixels: pixels28x28 };

        // Render directly using current canvas dimensions
        renderMnistImageToCanvas(
          ctx,
          pixelResolution,
          pixelResolution,
          pixels28x28,
          activeStroke,
          'low_pixel'
        );
        setHasContent(true);
        backupToOffscreen();

        const strokeAdjusted = adjustMnistStrokeWidth(pixels28x28, activeStroke);
        onPixelDataChange(strokeAdjusted);
        return strokeAdjusted;
      },
      [pixelResolution, brushSize, onPixelDataChange, backupToOffscreen]
    );

    // Live stroke width change handler
    const handleBrushSizeChange = (newSize: number) => {
      setBrushSize(newSize);

      const sample = currentSampleRef.current;
      if (sample) {
        if (sample.type === 'mnist' && sample.pixels) {
          drawMnistSample(sample.pixels, newSize);
        } else if (sample.type === 'synthetic' && sample.digit !== undefined) {
          drawSample(sample.digit, newSize);
        }
      }
    };

    // Expose handle to parent via ref
    useImperativeHandle(
      ref,
      () => ({
        clear: clearCanvas,
        autoFit: autoFitDrawing,
        drawSample,
        drawMnistSample,
        setTool,
      }),
      [clearCanvas, autoFitDrawing, drawSample, drawMnistSample, setTool]
    );

    // Keyboard shortcuts for canvas actions
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

        const key = e.key.toLowerCase();
        const code = e.code;

        if (key === 'b' || code === 'KeyB') {
          e.preventDefault();
          setTool('brush');
        } else if (key === 'e' || code === 'KeyE') {
          e.preventDefault();
          setTool('eraser');
        } else if (key === 'c' || code === 'KeyC') {
          e.preventDefault();
          clearCanvas();
        } else if (key === 'f' || code === 'KeyF') {
          e.preventDefault();
          autoFitDrawing();
        }
      };

      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }, [clearCanvas, autoFitDrawing]);

    // Pointer events for drawing
    const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (disabled) return;
      const canvas = canvasRef.current;
      if (!canvas) return;

      canvas.setPointerCapture(e.pointerId);
      setIsDrawing(true);
      currentSampleRef.current = null;

      const rect = canvas.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * pixelResolution;
      const y = ((e.clientY - rect.top) / rect.height) * pixelResolution;
      const pressure = e.pressure > 0 ? e.pressure : 0.5;

      pointsRef.current = [{ x, y, pressure }];

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.fillStyle = tool === 'brush' ? '#ffffff' : '#000000';
      const radius = Math.max(0.7, (brushSize / 240) * pixelResolution * 0.7 * (0.8 + pressure * 0.4));
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    };

    const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!isDrawing || disabled) return;
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const rect = canvas.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * pixelResolution;
      const y = ((e.clientY - rect.top) / rect.height) * pixelResolution;
      const pressure = e.pressure > 0 ? e.pressure : 0.5;

      const pts = pointsRef.current;
      pts.push({ x, y, pressure });

      if (pts.length < 2) return;

      const p0 = pts[pts.length - 2];
      const p1 = pts[pts.length - 1];
      const midX = (p0.x + p1.x) / 2;
      const midY = (p0.y + p1.y) / 2;
      const effectivePressure = (p0.pressure + p1.pressure) / 2;

      ctx.strokeStyle = tool === 'brush' ? '#ffffff' : '#000000';
      const lineWidth = Math.max(1.3, (brushSize / 240) * pixelResolution * 1.4 * (0.85 + effectivePressure * 0.3));
      ctx.lineWidth = lineWidth;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      if (pts.length === 2) {
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(midX, midY);
      } else {
        const pPrev = pts[pts.length - 3];
        const prevMidX = (pPrev.x + p0.x) / 2;
        const prevMidY = (pPrev.y + p0.y) / 2;
        ctx.moveTo(prevMidX, prevMidY);
        ctx.quadraticCurveTo(p0.x, p0.y, midX, midY);
      }
      ctx.stroke();
    };

    const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!isDrawing) return;
      const canvas = canvasRef.current;
      if (canvas && canvas.hasPointerCapture(e.pointerId)) {
        canvas.releasePointerCapture(e.pointerId);
      }
      setIsDrawing(false);
      pointsRef.current = [];
      // Sync pixels on end of stroke - NEVER auto-clear!
      syncProcessedPixels();
    };

    return (
      <div className="flex flex-col gap-2 p-3 bg-slate-900/90 rounded-xl border border-slate-800 shadow-md">
        {/* Clean Header: Simple title with no extra size text + Unified Tools */}
        <div className="flex items-center justify-between gap-1.5 min-w-0">
          <span className="text-xs font-semibold text-slate-200 tracking-wide whitespace-nowrap">
            Drawing Pad
          </span>

          {/* Unified Tool Actions: Brush, Eraser, Auto-Fit, Clear */}
          <div className="flex items-center bg-slate-950/90 p-0.5 rounded-lg border border-slate-800/80 shadow-inner gap-0.5 flex-shrink-0">
            {/* Brush Button */}
            <button
              type="button"
              data-testid="btn-brush"
              onClick={() => setTool('brush')}
              className={`flex items-center gap-1 px-1.5 py-1 rounded-md text-xs font-medium transition-all cursor-pointer ${
                tool === 'brush'
                  ? 'bg-cyan-500/25 text-cyan-300 shadow-xs'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
              title="Brush Tool (Press B)"
            >
              <PenTool size={12} className={tool === 'brush' ? 'text-cyan-400' : 'text-slate-400'} />
              <span className="hidden md:inline text-[9px] font-mono font-bold text-cyan-400/80">B</span>
            </button>

            {/* Eraser Button */}
            <button
              type="button"
              data-testid="btn-eraser"
              onClick={() => setTool('eraser')}
              className={`flex items-center gap-1 px-1.5 py-1 rounded-md text-xs font-medium transition-all cursor-pointer ${
                tool === 'eraser'
                  ? 'bg-cyan-500/25 text-cyan-300 shadow-xs'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
              title="Eraser Tool (Press E)"
            >
              <Eraser size={12} className={tool === 'eraser' ? 'text-cyan-400' : 'text-slate-400'} />
              <span className="hidden md:inline text-[9px] font-mono font-bold text-cyan-400/80">E</span>
            </button>

            <div className="w-px h-3 bg-slate-800 mx-0.5" />

            {/* Auto-Fit / Normalize Button */}
            <button
              type="button"
              data-testid="btn-autofit"
              onClick={autoFitDrawing}
              className="flex items-center gap-1 px-1.5 py-1 rounded-md text-xs font-medium text-slate-400 hover:text-cyan-300 hover:bg-slate-900 transition-all cursor-pointer active:scale-95"
              title="Auto-Fit / Normalize Drawing to Standard Size (Press F)"
            >
              <Maximize2 size={12} />
              <span className="hidden md:inline text-[9px] font-mono font-bold text-cyan-400/80">F</span>
            </button>

            {/* Clear / Delete Button */}
            <button
              type="button"
              data-testid="btn-clear"
              onClick={clearCanvas}
              className="flex items-center gap-1 px-1.5 py-1 rounded-md text-xs font-medium text-slate-400 hover:text-rose-400 hover:bg-slate-900 transition-all cursor-pointer active:scale-95"
              title="Clear Canvas (Press C)"
            >
              <Trash2 size={12} />
              <span className="hidden md:inline text-[9px] font-mono font-bold text-rose-400/80">C</span>
            </button>
          </div>
        </div>

        {/* Main Square Drawing Area (Compact size for mobile) */}
        <div
          className="relative aspect-square w-full max-w-[200px] sm:max-w-[240px] mx-auto rounded-xl overflow-hidden border border-slate-800 bg-black shadow-inner cursor-crosshair touch-none"
        >
          <canvas
            ref={canvasRef}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            style={{
              imageRendering: 'pixelated',
              width: '100%',
              height: '100%',
            }}
            className="w-full h-full block"
          />

          {!hasContent && (
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none text-slate-600 gap-1 select-none">
              <Sparkles size={16} className="text-slate-600/70" />
              <span className="text-[10px]">Draw digit (0–9)</span>
            </div>
          )}
        </div>

        {/* Clean Footer: Compact Stroke Slider (15px to 40px) */}
        <div className="flex items-center justify-between gap-2.5 text-xs pt-1 border-t border-slate-800/60">
          <span className="text-slate-400 text-[10px] font-medium flex-shrink-0">Stroke:</span>
          <input
            type="range"
            min={15}
            max={40}
            value={brushSize}
            onChange={(e) => handleBrushSizeChange(Number(e.target.value))}
            className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
            title={`Stroke Size: ${brushSize}px`}
          />
          <span className="font-mono text-slate-400 text-[10px] w-7 text-right flex-shrink-0">
            {brushSize}px
          </span>
        </div>
      </div>
    );
  }
);
DrawingCanvas.displayName = 'DrawingCanvas';
