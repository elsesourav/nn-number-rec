/**
 * Draws the Neural Network using the Wasm object directly.
 * Handles large layers by truncating the middle nodes.
 * Optimized to use scalar getters from Wasm to avoid object creation overhead.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} nn - The Wasm NeuralNetwork object
 * @param {number} width
 * @param {number} height
 * @param {number} progress - Animation progress (0.0 to 1.0), or -1 for no animation
 */
function drawNetwork(ctx, nn, width, height, progress) {
   const numLayers = nn.getNumLayers();
   const margin = width * 0.05;
   const layerSpacing = (width - 2 * margin) / (numLayers - 1);

   // Calculate the current "scanline" X position
   // If progress is -1 (not animating), show everything (limitX = width)
   // If animating, limitX goes from margin (input) to width-margin (output)
   // const limitX =
   //    progress >= 0
   //       ? margin + progress * (width - 2 * margin)
   //       : width;

   const maxNodes = 15; // Maximum nodes to show per layer

   const nodeRadius = height / maxNodes / 2.5;

   // Helper to get max nodes for a specific layer
   const getMaxNodesForLayer = (layerIndex) => {
      return layerIndex === 0 ? Math.floor(maxNodes * 1.4) : maxNodes;
   };

   // Helper to get visible indices for a layer
   const getVisibleIndices = (layerIndex) => {
      const cols = nn.getLayerSize(layerIndex);
      const indices = [];
      const limit = getMaxNodesForLayer(layerIndex);

      if (cols <= limit) {
         for (let i = 0; i < cols; i++) indices.push(i);
         return { indices, hasHidden: false, total: cols, limit };
      }

      if (layerIndex === 0) {
         // Sample evenly for input layer
         const step = cols / limit;
         for (let i = 0; i < limit; i++) {
            indices.push(Math.floor(i * step));
         }
         // Always show hidden indicator for input layer if it's larger than limit
         return { indices, hasHidden: true, total: cols, limit };
      } else {
         // Truncate middle for other layers
         const k = Math.floor(limit / 2);
         for (let i = 0; i < k; i++) indices.push(i);
         for (let i = cols - k; i < cols; i++) indices.push(i);
         return { indices, hasHidden: true, total: cols, limit };
      }
   };

   // Helper to get Y position based on visual index
   const getY = (visualIndex, count, hasHidden, limit) => {
      const gapSize = 2;
      const visualTotal = hasHidden ? limit + gapSize : count;
      const spacing = (height - 2 * margin) / (visualTotal - 1);

      let posIndex = visualIndex;
      if (hasHidden) {
         const k = Math.floor(limit / 2);
         if (visualIndex >= k) {
            posIndex += gapSize;
         }
      }

      return margin + posIndex * spacing;
   };

   ctx.clearRect(0, 0, width, height);

   // 1. Draw Weights
   for (let l = 0; l < numLayers - 1; l++) {
      const currentVis = getVisibleIndices(l);
      const nextVis = getVisibleIndices(l + 1);

      const x1 = margin + l * layerSpacing;
      const x2 = margin + (l + 1) * layerSpacing;

      currentVis.indices.forEach((i, visI) => {
         const y1 = getY(
            visI,
            currentVis.indices.length,
            currentVis.hasHidden,
            currentVis.limit
         );

         nextVis.indices.forEach((j, visJ) => {
            const y2 = getY(
               visJ,
               nextVis.indices.length,
               nextVis.hasHidden,
               nextVis.limit
            );

            // Use optimized getter
            const w = nn.getWeightVal(l, i, j);

            // Performance optimization: skip very small weights
            if (Math.abs(w) < 0.01) return;

            const alpha = Math.min(Math.abs(w), 1.0);

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);

            // Green only, opacity based on weight magnitude
            ctx.strokeStyle = `rgba(0, 255, 0, ${alpha})`;
            ctx.lineWidth = width * 0.0006;
            ctx.stroke();

            // Animation: Pulse effect
            if (progress >= 0) {
               const layerDuration = 1.0 / (numLayers - 1);
               const layerStart = l * layerDuration;
               const layerEnd = (l + 1) * layerDuration;

               if (progress >= layerStart && progress <= layerEnd) {
                  const localProgress = (progress - layerStart) / layerDuration;

                  // Draw a moving bright segment
                  const px = x1 + (x2 - x1) * localProgress;
                  const py = y1 + (y2 - y1) * localProgress;

                  ctx.beginPath();
                  // Draw a segment centered at current progress
                  const tailLen = 0.1; // Length of the pulse tail
                  const pStart = Math.max(0, localProgress - tailLen);

                  const sx = x1 + (x2 - x1) * pStart;
                  const sy = y1 + (y2 - y1) * pStart;

                  ctx.moveTo(sx, sy);
                  ctx.lineTo(px, py);

                  // Bright yellow/white pulse
                  ctx.strokeStyle = `rgba(255, 255, 200, 0.8)`;
                  ctx.lineWidth = width * 0.0015; // Thicker than normal line
                  ctx.stroke();
               }
            }
         });
      });
   }

   // 2. Draw Nodes
   ctx.setLineDash([]); // Reset dashes for nodes
   for (let l = 0; l < numLayers; l++) {
      const vis = getVisibleIndices(l);
      const x = margin + l * layerSpacing;

      // Find max index for output layer to highlight the winner
      let maxInd = -1;
      if (l === numLayers - 1) {
         let maxVal = -1;
         for (let j = 0; j < vis.total; j++) {
            const v = nn.getNeuronVal(l, j);
            if (v > maxVal) {
               maxVal = v;
               maxInd = j;
            }
         }
      }

      vis.indices.forEach((i, visIndex) => {
         const y = getY(visIndex, vis.indices.length, vis.hasHidden, vis.limit);

         // Use optimized getter
         let val = nn.getNeuronVal(l, i);

         // Animation: If the pulse hasn't reached this layer yet, show as inactive (black)
         if (progress >= 0 && l > 0) {
            const layerDuration = 1.0 / (numLayers - 1);
            const activationTime = l * layerDuration;
            if (progress < activationTime) {
               val = 0;
            }
         }

         let r = l === 0 ? nodeRadius * 0.6 : nodeRadius;
         if (l === numLayers - 1) {
            r = nodeRadius * 1.6;
            if (r > nodeRadius * 8) r = nodeRadius * 8;
         }

         ctx.beginPath();
         ctx.arc(x, y, r, 0, Math.PI * 2);
         const v = Math.floor(val * 255);
         ctx.fillStyle = `rgba(${v}, ${v}, ${v}, 1)`;
         ctx.strokeStyle = "white";
         ctx.lineWidth = width * 0.0012;
         ctx.fill();
         ctx.stroke();

         if (l === numLayers - 1) {
            if (i === maxInd && val !== 0) ctx.fillStyle = "#1500ffff";
            else ctx.fillStyle = val > 0.5 ? "black" : "white";
            ctx.font = `Semi Bold ${
               r * 1.2
            }px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(i, x, y);
         }
      });

      // Draw "..." if hidden
      if (vis.hasHidden) {
         const limit = vis.limit;
         const k = Math.floor(limit / 2);
         // Position between the last top node and first bottom node
         // We need visual indices for the gap calculation
         // The gap is between visual index k-1 and k
         const yTop = getY(k - 1, vis.indices.length, vis.hasHidden, limit);
         const yBot = getY(k, vis.indices.length, vis.hasHidden, limit);
         const yMid = (yTop + yBot) / 2;

         ctx.fillStyle = "gray";
         ctx.font = `${height * 0.05}px Arial`;
         ctx.textAlign = "center";
         ctx.textBaseline = "middle";
         ctx.fillText("⋮", x, yMid);
      }
   }
}
