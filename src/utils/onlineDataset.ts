/**
 * Online MNIST Dataset Loader & Renderer
 * Fetches and parses real handwritten digit images from the online MNIST dataset
 * and renders them onto the drawing canvas for authentic one-by-one training.
 */

export interface MnistSample {
  id: number;
  label: number;
  pixels: Float32Array; // 784 normalized values in [0, 1]
}

let cachedSamples: MnistSample[] | null = null;
let loadPromise: Promise<MnistSample[]> | null = null;

/**
 * Loads real MNIST handwritten digit dataset from the online dataset endpoint.
 */
export async function loadOnlineMnistDataset(): Promise<MnistSample[]> {
  if (cachedSamples && cachedSamples.length > 0) {
    return cachedSamples;
  }

  if (loadPromise) {
    return loadPromise;
  }

  loadPromise = (async () => {
    try {
      const response = await fetch('/dataset/mnist_samples.bin');
      if (!response.ok) {
        throw new Error(`Failed to fetch online MNIST dataset: ${response.statusText}`);
      }

      const buffer = await response.arrayBuffer();
      const bytes = new Uint8Array(buffer);
      const sampleSize = 785; // 1 byte label + 784 bytes pixels
      const count = Math.floor(bytes.length / sampleSize);

      const samples: MnistSample[] = [];

      for (let i = 0; i < count; i++) {
        const offset = i * sampleSize;
        const label = bytes[offset];
        const rawPixels = bytes.subarray(offset + 1, offset + sampleSize);

        const floatPixels = new Float32Array(784);
        for (let j = 0; j < 784; j++) {
          floatPixels[j] = rawPixels[j] / 255.0;
        }

        samples.push({
          id: i,
          label,
          pixels: floatPixels,
        });
      }

      cachedSamples = samples;
      return samples;
    } catch (err) {
      console.warn('Failed to load local MNIST dataset, falling back to empty', err);
      return [];
    } finally {
      loadPromise = null;
    }
  })();

  return loadPromise;
}

/**
 * Dynamically adjusts MNIST pixel stroke width based on the stroke slider.
 * Baseline stroke width is 18px (range 15px to 30px, default 18px).
 * When strokeWidth > 18, dilates the strokes smoothly to create a thicker pen mark.
 * When strokeWidth < 18, erodes/attenuates the edges to create a finer pen mark.
 */
export function adjustMnistStrokeWidth(
  pixels28x28: Float32Array | number[],
  strokeWidth: number = 18
): Float32Array {
  // If at default baseline (18px ± 0.5), return direct copy
  if (Math.abs(strokeWidth - 18) < 0.5) {
    return new Float32Array(pixels28x28);
  }

  const out = new Float32Array(784);
  const getP = (x: number, y: number): number =>
    x >= 0 && x < 28 && y >= 0 && y < 28 ? pixels28x28[y * 28 + x] : 0;

  if (strokeWidth > 18) {
    // Dilate (thicken stroke): expand mass towards neighboring pixels (range 18..30)
    const factor = Math.min(1.0, (strokeWidth - 18) / 12); // 0..1
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const cur = getP(x, y);
        const maxCard = Math.max(getP(x - 1, y), getP(x + 1, y), getP(x, y - 1), getP(x, y + 1));
        const maxDiag = Math.max(
          getP(x - 1, y - 1),
          getP(x + 1, y - 1),
          getP(x - 1, y + 1),
          getP(x + 1, y + 1)
        );
        const dilated = Math.max(cur, maxCard * 0.82, maxDiag * 0.58);
        out[y * 28 + x] = Math.min(1.0, cur * (1.0 - factor * 0.75) + dilated * (factor * 0.75));
      }
    }
  } else {
    // Erode (thin stroke): attenuate faint outer edges (range 15..18)
    const factor = Math.min(1.0, (18 - strokeWidth) / 3.5); // 0..1
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const cur = getP(x, y);
        const minCard = Math.min(getP(x - 1, y), getP(x + 1, y), getP(x, y - 1), getP(x, y + 1));
        const thinned = Math.pow(cur, 1.0 + factor * 1.35);
        out[y * 28 + x] = Math.max(0.0, thinned * (1.0 - factor * 0.3) + minCard * (factor * 0.3));
      }
    }
  }

  return out;
}

/**
 * Renders an authentic 28x28 MNIST handwritten digit onto a canvas context.
 * In 'low_pixel' mode (default), renders crisp, pixelated square blocks with zero blur,
 * exactly like the original MNIST raster dataset.
 * In 'smooth' mode, applies anti-aliasing interpolation.
 * Supports dynamic stroke width adjustment matching the user's stroke slider.
 */
export function renderMnistImageToCanvas(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  pixels28x28: Float32Array | number[],
  strokeWidth: number = 18,
  renderMode: 'low_pixel' | 'smooth' = 'low_pixel'
): void {
  ctx.save();

  // Apply dynamic stroke width adjustment if different from baseline
  const activePixels = adjustMnistStrokeWidth(pixels28x28, strokeWidth);

  // 1. Clear canvas with deep black
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, width, height);

  // 2. Create offscreen 28x28 canvas to draw raw pixels
  const offscreen = document.createElement('canvas');
  offscreen.width = 28;
  offscreen.height = 28;
  const offCtx = offscreen.getContext('2d');
  if (!offCtx) {
    ctx.restore();
    return;
  }

  const imgData = offCtx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const val = Math.min(255, Math.max(0, Math.round(activePixels[i] * 255)));
    const idx = i * 4;
    imgData.data[idx + 0] = val; // Red
    imgData.data[idx + 1] = val; // Green
    imgData.data[idx + 2] = val; // Blue
    imgData.data[idx + 3] = 255; // Alpha
  }
  offCtx.putImageData(imgData, 0, 0);

  if (width === 28 && height === 28) {
    ctx.putImageData(imgData, 0, 0);
  } else if (renderMode === 'low_pixel') {
    // Low-pixel mode: Crisp nearest-neighbor pixel blocks (exact original MNIST look!)
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0, width, height);
  } else {
    // Smooth HD mode: bilinear anti-aliased interpolation
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    const pad = width * 0.04;
    const renderSize = width - pad * 2;
    ctx.drawImage(offscreen, pad, pad, renderSize, renderSize);
  }

  ctx.restore();
}
