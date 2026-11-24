import React, { useRef, useEffect, useCallback } from 'react';
import type { NeuralNetworkBridge } from '../../wasm/wasmBridge';

interface NetworkVisualizerProps {
  bridge: NeuralNetworkBridge | null;
  animating?: boolean;
  animationDuration?: number;
  activeDigit?: number | null;
}

export const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({
  bridge,
  animating = false,
  animationDuration = 1800,
  activeDigit = null,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);

  const drawNetwork = useCallback(
    (ctx: CanvasRenderingContext2D, width: number, height: number, progress: number) => {
      if (!bridge) return;

      const numLayers = bridge.getNumLayers();
      if (numLayers < 2) return;

      // Dynamic margins scaled to width & height with tight mobile optimization
      const isNarrowMobile = width < 400;
      const marginXLeft = isNarrowMobile ? 20 : Math.max(32, width * 0.08);
      // Give Output layer probability tags enough space on the right (at least 38px)
      const marginXRight = isNarrowMobile ? 38 : Math.max(36, width * 0.09);
      // Push topmost nodes down so they NEVER collide with top floating controls (Active Flow & Switcher)
      const marginYTop = isNarrowMobile ? 48 : Math.max(36, Math.min(44, height * 0.09));
      const marginYBottom = isNarrowMobile ? 44 : Math.max(50, Math.min(64, height * 0.15));
      const layerSpacing = (width - marginXLeft - marginXRight) / (numLayers - 1);
      const availableH = height - marginYTop - marginYBottom;
      const maxNodesPerLayer = Math.min(26, Math.floor(availableH / (isNarrowMobile ? 12 : 14)));

      // Helper to sample indices for dense layers (with intelligent active input pixel sampling)
      const getLayerVisualIndices = (layerIdx: number) => {
        const total = bridge.getLayerSize(layerIdx);

        // Output layer ALWAYS displays all classes (all 10 digits 0–9) without truncation!
        if (layerIdx === numLayers - 1) {
          const indices: number[] = [];
          for (let i = 0; i < total; i++) indices.push(i);
          return { indices, hasHidden: false, total, limit: total };
        }

        const limit = layerIdx === 0 ? Math.min(22, maxNodesPerLayer) : Math.min(18, maxNodesPerLayer);

        if (total <= limit) {
          const indices = [];
          for (let i = 0; i < total; i++) indices.push(i);
          return { indices, hasHidden: false, total, limit };
        }

        // For Input Layer (784 pixels), prioritize active drawn pixels with dot gaps
        if (layerIdx === 0) {
          const activeIndices: number[] = [];
          for (let p = 0; p < total; p++) {
            if (bridge.getNeuronValue(0, p) > 0.05) {
              activeIndices.push(p);
            }
          }

          const indices: number[] = [];
          const half = Math.floor(limit / 2);

          if (activeIndices.length > 0) {
            // Include top active pixels plus boundary baselines
            const sampledActive: number[] = [];
            const step = Math.max(1, Math.floor(activeIndices.length / (limit - 4)));
            for (let k = 0; k < activeIndices.length && sampledActive.length < limit - 4; k += step) {
              sampledActive.push(activeIndices[k]);
            }

            // Top baseline
            indices.push(0, 14);
            // Drawn active pixels
            indices.push(...sampledActive);
            // Bottom baseline
            indices.push(total - 15, total - 1);
          } else {
            // Default split
            for (let i = 0; i < half; i++) indices.push(i * 2);
            for (let i = total - half * 2; i < total; i += 2) indices.push(i);
          }

          return { indices: indices.slice(0, limit), hasHidden: true, total, limit };
        }

        // For hidden layers
        const indices: number[] = [];
        const half = Math.floor(limit / 2);
        for (let i = 0; i < half; i++) indices.push(i);
        for (let i = total - half; i < total; i++) indices.push(i);
        return { indices, hasHidden: true, total, limit };
      };

      const getY = (
        visIdx: number,
        count: number,
        hasHidden: boolean,
        limit: number
      ) => {
        const gap = 1.4;
        const totalPositions = hasHidden ? limit + gap : count;
        const availableHeight = height - marginYTop - marginYBottom;
        const spacing = availableHeight / (totalPositions > 1 ? totalPositions - 1 : 1);

        let pos = visIdx;
        if (hasHidden && visIdx >= Math.floor(limit / 2)) {
          pos += gap;
        }

        if (totalPositions <= 1) return (marginYTop + availableHeight) / 2;
        return marginYTop + pos * spacing;
      };

      ctx.clearRect(0, 0, width, height);

      // 1. Draw Weights / Connections (Realistic weight-based glow without fake lines)
      const numTransitions = numLayers - 1;

      for (let l = 0; l < numTransitions; l++) {
        const curVis = getLayerVisualIndices(l);
        const nextVis = getLayerVisualIndices(l + 1);

        const x1 = marginXLeft + l * layerSpacing;
        const x2 = marginXLeft + (l + 1) * layerSpacing;

        // Is this inter-layer interval currently carrying the active animation wave?
        const transStart = l / numTransitions;
        const transEnd = (l + 1) / numTransitions;
        const isCurrentWaveInterval = progress >= transStart && progress <= transEnd;
        const localProg = isCurrentWaveInterval
          ? (progress - transStart) / (transEnd - transStart)
          : 0;

        curVis.indices.forEach((i, visI) => {
          const y1 = getY(visI, curVis.indices.length, curVis.hasHidden, curVis.limit);
          const fromVal = bridge.getNeuronValue(l, i);

          nextVis.indices.forEach((j, visJ) => {
            const y2 = getY(visJ, nextVis.indices.length, nextVis.hasHidden, nextVis.limit);

            const w = bridge.getWeightValue(l, i, j);
            if (Math.abs(w) < 0.02) return;

            const alpha = Math.min(Math.abs(w) * 0.45, 0.6);

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);

            // Emerald for positive weight (+Weight), Fuchsia for negative weight (-Weight)
            // Stroke line width scales dynamically with connection weight magnitude
            if (w > 0) {
              ctx.strokeStyle = `rgba(16, 185, 129, ${alpha})`;
            } else {
              ctx.strokeStyle = `rgba(217, 70, 239, ${alpha})`;
            }

            ctx.lineWidth = Math.min(2.0, 0.6 + Math.abs(w) * 1.1);
            ctx.stroke();

            // Real weight & activation based light-up: only glow if actual non-zero signal passes!
            const signal = fromVal * Math.abs(w);
            if (isCurrentWaveInterval && fromVal > 0.04 && signal > 0.012) {
              const px = x1 + (x2 - x1) * localProg;
              const py = y1 + (y2 - y1) * localProg;
              const tailProg = Math.max(0, localProg - 0.25);
              const tx = x1 + (x2 - x1) * tailProg;
              const ty = y1 + (y2 - y1) * tailProg;

              const beamAlpha = Math.min(0.98, 0.45 + signal * 1.6);
              const beamWidth = Math.min(3.4, 1.2 + signal * 2.6);

              // Glowing signal beam trail
              ctx.beginPath();
              ctx.moveTo(tx, ty);
              ctx.lineTo(px, py);
              ctx.strokeStyle = w > 0 ? `rgba(52, 211, 153, ${beamAlpha})` : `rgba(232, 121, 249, ${beamAlpha})`;
              ctx.lineWidth = beamWidth;
              ctx.stroke();

              // Photon head with halo proportional to signal strength
              const headRadius = Math.min(3.8, 1.6 + signal * 2.2);
              ctx.beginPath();
              ctx.arc(px, py, headRadius, 0, Math.PI * 2);
              ctx.fillStyle = '#ffffff';
              ctx.shadowColor = w > 0 ? '#10b981' : '#e879f9';
              ctx.shadowBlur = Math.min(12, 5 + signal * 9);
              ctx.fill();
              ctx.shadowBlur = 0;
            }
          });
        });
      }

      // 2. Draw Neurons with Sequential Layer-to-Layer Illumination
      for (let l = 0; l < numLayers; l++) {
        const vis = getLayerVisualIndices(l);
        const x = marginXLeft + l * layerSpacing;
        const isOutputLayer = l === numLayers - 1;
        const isInputLayer = l === 0;

        // Compact node sizes for hidden layers, dynamically scaled for output to never overlap
        const radius = isOutputLayer
          ? Math.min(13, Math.max(7.5, availableH / (isNarrowMobile ? 26 : 24)))
          : Math.min(5.5, Math.max(2.8, availableH / (vis.limit * 3.5)));

        // Determine if wave has reached, is at, or has passed this layer
        // Layer l active emission starts at l / numTransitions
        const layerArrival = (l - 1) / numTransitions;
        const layerStart = l / numTransitions;
        const layerEnd = (l + 1) / numTransitions;

        let layerState: 'idle' | 'resting' | 'arriving' | 'ignited' | 'active';

        if (progress < 0) {
          layerState = 'idle'; // Normal static display
        } else if (isInputLayer) {
          layerState = progress <= layerEnd ? 'ignited' : 'active';
        } else if (isOutputLayer) {
          if (progress < layerArrival) {
            layerState = 'resting'; // Dim resting before wave arrives
          } else if (progress < 0.95) {
            layerState = 'arriving';
          } else {
            layerState = 'ignited';
          }
        } else {
          // Hidden layers
          if (progress < layerArrival) {
            layerState = 'resting'; // Downstream layer, wave has not reached yet
          } else if (progress < layerStart) {
            layerState = 'arriving'; // Photons streaming towards it
          } else if (progress <= layerEnd) {
            layerState = 'ignited'; // Active emitting stage: full flash and glow!
          } else {
            layerState = 'active'; // Post-emission steady state
          }
        }

        // Find winning neuron on output layer
        let maxOutputIdx = -1;
        let maxOutputVal = -1;
        if (isOutputLayer) {
          for (let j = 0; j < vis.total; j++) {
            const v = bridge.getNeuronValue(l, j);
            if (v > maxOutputVal) {
              maxOutputVal = v;
              maxOutputIdx = j;
            }
          }
        }

        vis.indices.forEach((i, visIdx) => {
          const y = getY(visIdx, vis.indices.length, vis.hasHidden, vis.limit);
          const val = bridge.getNeuronValue(l, i);

          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);

          if (isOutputLayer) {
            // Output Layer
            const showWinnerGlow =
              (layerState === 'idle' || layerState === 'ignited' || (layerState === 'arriving' && progress > 0.85)) &&
              (i === maxOutputIdx || i === activeDigit) &&
              val > 0.04;

            if (showWinnerGlow) {
              ctx.fillStyle = '#10b981';
              ctx.shadowColor = '#10b981';
              ctx.shadowBlur = 24;
              ctx.strokeStyle = '#34d399';
              ctx.lineWidth = 2.5;
            } else if (layerState === 'resting') {
              // Dim before wave arrival
              ctx.fillStyle = '#0b1329';
              ctx.strokeStyle = '#1e293b';
              ctx.lineWidth = 1.2;
              ctx.shadowBlur = 0;
            } else if (val > 0.05) {
              ctx.fillStyle = `rgba(6, 182, 212, ${Math.min(val * 0.7 + 0.15, 0.85)})`;
              ctx.strokeStyle = '#0891b2';
              ctx.lineWidth = 1.5;
              ctx.shadowBlur = 0;
            } else {
              ctx.fillStyle = '#0f172a';
              ctx.strokeStyle = '#334155';
              ctx.lineWidth = 1.5;
              ctx.shadowBlur = 0;
            }

            ctx.fill();
            ctx.stroke();
            ctx.shadowBlur = 0;

            // Draw bold digit label
            const isWinner = (i === maxOutputIdx || i === activeDigit) && val > 0.04;
            const digitColor = showWinnerGlow
              ? '#022c22'
              : layerState === 'resting'
              ? '#475569'
              : val > 0.25
              ? '#ffffff'
              : '#94a3b8';

            const nodeFontSize = Math.max(8.5, Math.min(13, Math.round(radius * 0.95)));
            ctx.fillStyle = digitColor;
            ctx.font = `bold ${nodeFontSize}px monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(i.toString(), x, y);

            // Probability percentage next to the output node
            if (val > 0.02 && (layerState === 'idle' || layerState === 'ignited' || progress > 0.8)) {
              ctx.fillStyle = isWinner ? '#34d399' : '#64748b';
              ctx.font = isWinner
                ? `bold ${isNarrowMobile ? 9 : 11}px monospace`
                : `${isNarrowMobile ? 8.5 : 10}px monospace`;
              ctx.textAlign = 'left';
              ctx.textBaseline = 'middle';
              ctx.fillText(`${Math.round(val * 100)}%`, x + radius + (isNarrowMobile ? 3 : 5), y);
            }
          } else if (isInputLayer) {
            // Input Layer
            const brightness = Math.floor(Math.min(1.0, val) * 255);
            if (layerState === 'ignited' && val > 0.05) {
              // Radiant glow at inception
              ctx.fillStyle = '#38bdf8';
              ctx.shadowColor = '#06b6d4';
              ctx.shadowBlur = 12;
              ctx.strokeStyle = '#e0f2fe';
              ctx.lineWidth = 1.8;
            } else {
              ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;
              ctx.strokeStyle = '#334155';
              ctx.lineWidth = 1;
              ctx.shadowBlur = 0;
            }
            ctx.fill();
            ctx.stroke();
            ctx.shadowBlur = 0;
          } else {
            // Hidden Layers
            const brightness = Math.floor(Math.min(1.0, val) * 255);

            if (layerState === 'resting') {
              // Dim downstream layer before wave reaches it
              ctx.fillStyle = '#0b1329';
              ctx.strokeStyle = '#1e293b';
              ctx.lineWidth = 1;
              ctx.shadowBlur = 0;
            } else if (layerState === 'arriving') {
              // Soft fade-in as photons arrive
              const arrProg = Math.max(0, (progress - layerArrival) / (layerStart - layerArrival));
              const fade = Math.min(1.0, arrProg * 1.5);
              const b = Math.floor(brightness * fade);
              ctx.fillStyle = `rgb(${b}, ${b}, ${Math.floor(b * 1.2)})`;
              ctx.strokeStyle = fade > 0.5 ? '#0284c7' : '#1e293b';
              ctx.lineWidth = 1;
              ctx.shadowBlur = 0;
            } else if (layerState === 'ignited') {
              // Layer in ACTIVE IGNITION flash: bright cyan glow
              ctx.fillStyle = val > 0.08 ? '#38bdf8' : `rgb(${brightness}, ${brightness}, ${Math.floor(brightness * 1.2)})`;
              ctx.shadowColor = '#06b6d4';
              ctx.shadowBlur = 14;
              ctx.strokeStyle = '#e0f2fe';
              ctx.lineWidth = 2.0;
            } else {
              // Active steady state after wave has passed
              ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${Math.floor(brightness * 1.2)})`;
              ctx.strokeStyle = '#334155';
              ctx.lineWidth = 1;
              ctx.shadowBlur = 0;
            }

            ctx.fill();
            ctx.stroke();
            ctx.shadowBlur = 0;
          }
        });

        // Draw ellipsis for truncated hidden layers
        if (vis.hasHidden) {
          const half = Math.floor(vis.limit / 2);
          const yTop = getY(half - 1, vis.indices.length, vis.hasHidden, vis.limit);
          const yBot = getY(half, vis.indices.length, vis.hasHidden, vis.limit);
          const yMid = (yTop + yBot) / 2;

          ctx.fillStyle = '#64748b';
          ctx.font = 'bold 15px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('⋮', x, yMid);
        }

        // Layer Title & Node count inside canvas boundary
        ctx.fillStyle = '#94a3b8';
        ctx.font = `600 ${isNarrowMobile ? 9.5 : 11}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        let layerLabel = `Hidden ${l}`;
        if (isNarrowMobile) layerLabel = `H${l}`;
        if (l === 0) layerLabel = isNarrowMobile ? 'In' : 'Input';
        if (l === numLayers - 1) layerLabel = isNarrowMobile ? 'Out' : 'Output';
        ctx.fillText(layerLabel, x, height - (isNarrowMobile ? 32 : 44));
        ctx.fillStyle = '#64748b';
        ctx.font = `${isNarrowMobile ? 8.5 : 10}px monospace`;
        ctx.fillText(`(${bridge.getLayerSize(l)})`, x, height - (isNarrowMobile ? 20 : 30));
      }
    },
    [bridge]
  );

  // Resize handler & render cycle
  const renderCanvas = useCallback(
    (progress: number = -1) => {
      const container = containerRef.current;
      const canvas = canvasRef.current;
      if (!container || !canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const rect = container.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      ctx.save();
      ctx.scale(dpr, dpr);
      drawNetwork(ctx, rect.width, rect.height, progress);
      ctx.restore();
    },
    [drawNetwork]
  );

  // ResizeObserver to automatically resize on window or drawer changes
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver(() => {
      renderCanvas(-1);
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, [renderCanvas]);

  // Animation trigger on feedforward
  useEffect(() => {
    if (animating) {
      if (animationDuration <= 0) {
        renderCanvas(-1);
        return;
      }

      const duration = animationDuration;
      startTimeRef.current = performance.now();

      const animate = (time: number) => {
        const elapsed = time - (startTimeRef.current || time);
        const progress = Math.min(1.0, elapsed / duration);

        renderCanvas(progress);

        if (progress < 1.0) {
          animFrameRef.current = requestAnimationFrame(animate);
        } else {
          renderCanvas(-1);
        }
      };

      animFrameRef.current = requestAnimationFrame(animate);

      return () => {
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      };
    } else {
      renderCanvas(-1);
    }
  }, [animating, animationDuration, renderCanvas, activeDigit]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full flex-1 flex flex-col justify-between overflow-hidden bg-slate-950/80 rounded-2xl"
    >
      <canvas ref={canvasRef} className="w-full h-full block" />
    </div>
  );
};
