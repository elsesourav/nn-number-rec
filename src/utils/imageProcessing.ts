/**
 * Preprocesses a high-resolution canvas drawing into an anti-aliased,
 * size-normalized, and centered Float32Array suitable for feeding into the neural network.
 * Follows the standard MNIST specification: extracts the tight bounding box,
 * preserves aspect ratio, fits into a 20x20 box inside the 28x28 canvas (auto-fitting extra sides),
 * and fine-tunes centering by center of mass.
 */

export interface ProcessOptions {
  pixelResolution: number; // e.g. 28 -> 28x28 = 784
  centeringMethod?: 'centerOfMass' | 'boundingBox';
  threshold?: number;
  autoFitSides?: boolean; // default true: eliminates extra side margins and scales to normal size
}

export function processCanvasToInput(
  sourceCanvas: HTMLCanvasElement,
  options: ProcessOptions
): {
  pixels: Float32Array;
  previewCanvas: HTMLCanvasElement;
} {
  const {
    pixelResolution = 28,
    centeringMethod = 'centerOfMass',
    threshold = 0.05,
    autoFitSides = true,
  } = options;
  const numInputs = pixelResolution * pixelResolution;

  // Create an offscreen canvas at the target pixel resolution
  const targetCanvas = document.createElement('canvas');
  targetCanvas.width = pixelResolution;
  targetCanvas.height = pixelResolution;
  const ctx = targetCanvas.getContext('2d', { willReadFrequently: true });

  if (!ctx) {
    return {
      pixels: new Float32Array(numInputs),
      previewCanvas: targetCanvas,
    };
  }

  // 1. Scan source canvas to detect non-empty drawing bounds (eliminating extra sides)
  const srcW = sourceCanvas.width;
  const srcH = sourceCanvas.height;
  const srcCtx = sourceCanvas.getContext('2d', { willReadFrequently: true });

  let minX = srcW;
  let minY = srcH;
  let maxX = 0;
  let maxY = 0;
  let hasData = false;

  if (srcCtx && srcW > 0 && srcH > 0) {
    // Sample with step to maintain fast interactive performance
    const step = Math.max(1, Math.floor(srcW / 120));
    const srcImgData = srcCtx.getImageData(0, 0, srcW, srcH);
    const data = srcImgData.data;

    for (let y = 0; y < srcH; y += step) {
      for (let x = 0; x < srcW; x += step) {
        const idx = (y * srcW + x) * 4;
        const val = Math.max(data[idx], data[idx + 1], data[idx + 2]) / 255.0;
        if (val > threshold) {
          hasData = true;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
  }

  // If blank canvas, return zeros
  if (!hasData || minX > maxX || minY > maxY) {
    return {
      pixels: new Float32Array(numInputs),
      previewCanvas: targetCanvas,
    };
  }

  // 2. Auto-Fit to Normal Size:
  // If digit has extra side padding or is off-center, crop the bounding box
  // and scale it proportionally to fit inside a 20x20 region (71.4% of 28x28)
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, pixelResolution, pixelResolution);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  const boxW = Math.max(1, maxX - minX + 1);
  const boxH = Math.max(1, maxY - minY + 1);

  if (autoFitSides && boxW > 4 && boxH > 4) {
    // Standard MNIST ratio: 20 pixels max dimension inside 28x28 (ratio ~0.714)
    const targetSize = Math.round(pixelResolution * 0.72);
    const scale = Math.min(targetSize / boxH, targetSize / boxW);
    const fittedW = Math.max(1, Math.round(boxW * scale));
    const fittedH = Math.max(1, Math.round(boxH * scale));

    const destX = Math.round((pixelResolution - fittedW) / 2);
    const destY = Math.round((pixelResolution - fittedH) / 2);

    ctx.drawImage(sourceCanvas, minX, minY, boxW, boxH, destX, destY, fittedW, fittedH);
  } else {
    ctx.drawImage(sourceCanvas, 0, 0, pixelResolution, pixelResolution);
  }

  // 3. Read scaled image and compute Center of Mass for sub-pixel centering
  const imgData = ctx.getImageData(0, 0, pixelResolution, pixelResolution);
  const rawData = imgData.data;
  const rawGrayscale = new Float32Array(numInputs);

  let totalMass = 0;
  let massX = 0;
  let massY = 0;

  for (let y = 0; y < pixelResolution; y++) {
    for (let x = 0; x < pixelResolution; x++) {
      const idx = (y * pixelResolution + x) * 4;
      const val = Math.max(rawData[idx], rawData[idx + 1], rawData[idx + 2]) / 255.0;
      const pixelIdx = y * pixelResolution + x;
      rawGrayscale[pixelIdx] = val;

      if (val > threshold) {
        totalMass += val;
        massX += x * val;
        massY += y * val;
      }
    }
  }

  // 4. Fine-tune centering by Center of Mass (LeCun standard)
  const centeredGrayscale = new Float32Array(numInputs);
  let shiftX = 0;
  let shiftY = 0;

  if (centeringMethod === 'centerOfMass' && totalMass > 0) {
    const cx = massX / totalMass;
    const cy = massY / totalMass;
    // Clamp shift to ±3 pixels so auto-fit stays stably within bounds
    shiftX = Math.max(-3, Math.min(3, Math.round((pixelResolution - 1) / 2 - cx)));
    shiftY = Math.max(-3, Math.min(3, Math.round((pixelResolution - 1) / 2 - cy)));
  }

  for (let y = 0; y < pixelResolution; y++) {
    for (let x = 0; x < pixelResolution; x++) {
      const val = rawGrayscale[y * pixelResolution + x];
      if (val <= 0) continue;

      const newX = x + shiftX;
      const newY = y + shiftY;

      if (newX >= 0 && newX < pixelResolution && newY >= 0 && newY < pixelResolution) {
        centeredGrayscale[newY * pixelResolution + newX] = val;
      }
    }
  }

  // 5. Update preview canvas with final centered & normalized image
  const previewImgData = ctx.createImageData(pixelResolution, pixelResolution);
  for (let i = 0; i < numInputs; i++) {
    const v = Math.round(centeredGrayscale[i] * 255);
    const pIdx = i * 4;
    previewImgData.data[pIdx] = v;
    previewImgData.data[pIdx + 1] = v;
    previewImgData.data[pIdx + 2] = v;
    previewImgData.data[pIdx + 3] = 255;
  }
  ctx.putImageData(previewImgData, 0, 0);

  return {
    pixels: centeredGrayscale,
    previewCanvas: targetCanvas,
  };
}
