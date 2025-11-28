class NNView {
   /**
    * @param {NeuralNetwork} nn
    * @param {number} width
    * @param {number} height
    * @param {object} options
    *   - minRadiusInput
    *   - minRadiusHidden
    *   - minRadiusOutput
    */
   constructor(nn, width, height, options = {}) {
      this.nn = nn;
      this.width = width;
      this.height = height;

      this.c = new Canvas(this.width, this.height, ID("view-cvs"));
      this.fps = 60;
      this.ani = new Animation(this.fps);

      this.margin = 10;

      // This was your old vertical scale, keeping as base
      this.numCol = 17;
      this.sclH = this.height / this.numCol;
      this.rBase = this.sclH / 3;

      // Optional control of minimum radius (if too many neurons)
      const {
         minRadiusInput = this.rBase / 4,
         minRadiusHidden = this.rBase / 3,
         minRadiusOutput = this.rBase * 1.2,
      } = options;

      this.minRadiusInput = minRadiusInput;
      this.minRadiusHidden = minRadiusHidden;
      this.minRadiusOutput = minRadiusOutput;

      // Horizontal spacing based on number of layers in NN
      this.colSpacing =
         (this.width - 2 * this.margin) / (this.nn.numLayers - 1);

      // Animation timing
      this.f = 10;
      this.runTime = this.f * 4;
      this.totalRunTime = this.runTime;

      // Geometry of each layer
      // layerNodes[layerIndex] = [ {x,y,r,index?}, ... ]
      this.layerNodes = [];

      // Cached activations per layer for drawing
      this.layerActs = [];

      // Aliases for convenience / backwards-style use
      this.inps = [];
      this.outs = [];

      // Precompute node positions
      this.setInps();
      this.setHiddenAndOuts();

      this.inpPnt = undefined;
      this.maxIndex = undefined;
   }

   // ----------------- PUBLIC: start / stop animation -----------------
   aniStart() {
      this.runTime = this.f * 4;
      this.totalRunTime = this.runTime;

      const L = this.nn.numLayers - 1; // last layer index

      // ---- Inputs: compress raw input array into N points like before ----
      // const rawInput = this.nn.inputs.data[0]; // Matrix row
      const inputs = this.nn.inputs;
      const rawInput = [];
      const inCols = inputs.getCols();
      for (let i = 0; i < inCols; i++) rawInput.push(inputs.at(0, i));

      this.inpPnt = getInputPoints(rawInput, this.layerNodes[0].length);
      this.layerActs[0] = this.inpPnt;

      // ---- Hidden + Output activations ----
      for (let i = 1; i <= L; i++) {
         // const row = this.nn.layers[i].data[0].slice();
         const layer = this.nn.layers[i];
         const cols = layer.getCols();
         const row = [];
         for (let k = 0; k < cols; k++) row.push(layer.at(0, k));

         // clamp to [0,1] for visualization
         this.layerActs[i] = row.map((v) => {
            if (v > 1) return 1;
            if (v < 0) return 0;
            return v;
         });
      }

      // ---- Max index on output layer ----
      this.out = this.layerActs[L];
      this.maxIndex = getMaximumIndex(this.out);

      this.ani.start(() => {
         this.draw();
         this.drawVal();
         this.runTime--;
         if (this.runTime <= 0) this.aniStop();
      });
   }

   aniStop() {
      this.ani.stop();
   }

   // ----------------- MAIN DRAW -----------------
   draw() {
      this.c.clrScr();
      this.drawConnections();
      this.drawNodesSkeleton();
   }

   drawVal() {
      // Color nodes based on activations (green intensity)

      const L = this.nn.numLayers - 1;

      // All layers
      for (let li = 0; li <= L; li++) {
         const layer = this.layerNodes[li];
         const acts = this.layerActs[li] || [];

         layer.forEach((node, i) => {
            const a = acts[i] || 0;
            this.c.arc(node.x, node.y, node.r > 2 ? node.r - 1 : node.r);
            this.c.fill(0, 255 * a, 0);
         });
      }

      // Special handling for output labels & max highlight
      const outLayer = this.layerNodes[L];
      if (!outLayer) return;

      outLayer.forEach((v, i) => {
         // Draw index & highlight max
         this.c.fillStyle(255);
         this.c.textMode(this.c.CENTER);
         this.c.font(this.sclH * 0.9, "Arial");

         if (this.maxIndex === i) {
            // Bright label
            this.c.fillText(v.index, v.x + v.r * 3, v.y + 2);

            // Small arrow line
            this.c.moveTo(v.x + v.r * 2, v.y + v.r);
            this.c.lineTo(v.x + v.r * 4, v.y + v.r);
            this.c.stroke(0, 255, 0);
         } else {
            // Dim label
            this.c.fillStyle(255 * 0.3);
            this.c.fillText(v.index, v.x + v.r * 3, v.y + 2);
         }
      });
   }

   // ----------------- NODE DRAW (OUTLINES) -----------------
   drawNodesSkeleton() {
      const L = this.nn.numLayers - 1;

      for (let li = 0; li <= L; li++) {
         const layer = this.layerNodes[li];
         layer.forEach((v) => {
            this.c.arc(v.x, v.y, v.r);
            this.c.lineWidth(2);
            this.c.stroke(0, 255, 255);
            this.c.fill(0);
         });
      }
   }

   // ----------------- CONNECTIONS -----------------
   drawConnections() {
      const L = this.nn.numLayers - 1;

      for (let li = 0; li < L; li++) {
         const layerA = this.layerNodes[li];
         const layerB = this.layerNodes[li + 1];

         layerA.forEach((a) => {
            layerB.forEach((b) => {
               this.c.moveTo(a.x, a.y);
               this.c.lineTo(b.x, b.y);
               this.c.lineWidth(this.sclH / 100);
               this.c.stroke(255);
            });
         });
      }
   }

   // ----------------- GEOMETRY / LAYOUT -----------------

   // Input “summary” nodes (kept similar to your old design)
   setInps() {
      this.inps = [];
      for (let i = 0; i < this.numCol; i++) {
         if (i === Math.round(this.numCol / 2) - 1) {
            // middle split into 2 small nodes
            let gap = this.sclH / 4;

            this.inps.push({
               x: this.margin + this.rBase,
               y: this.sclH * i + this.rBase * 1.5 - gap,
               r: this.rBase / 4,
            });
            this.inps.push({
               x: this.margin + this.rBase,
               y: this.sclH * i + this.rBase * 1.5 + gap,
               r: this.rBase / 4,
            });
         } else {
            this.inps.push({
               x: this.margin + this.rBase,
               y: this.sclH * i + this.rBase * 1.5,
               r: this.rBase,
            });
         }
      }

      this.layerNodes[0] = this.inps;
   }

   // Hidden + Output layers
   setHiddenAndOuts() {
      const L = this.nn.numLayers - 1;
      this.outs = [];

      for (let li = 1; li <= L; li++) {
         const count = this.nn.layerSizes[li];
         const x = this.margin + this.colSpacing * li;
         const isOutput = li === L;

         const nodes = this.createLayerNodes(
            count,
            x,
            isOutput ? "output" : "hidden"
         );

         // attach indices for outputs (digits)
         if (isOutput) {
            nodes.forEach((n, idx) => (n.index = idx));
            this.outs = nodes;
         }

         this.layerNodes[li] = nodes;
      }
   }

   // Auto-fit a layer vertically based on count
   createLayerNodes(count, x, type = "hidden") {
      const availableH = this.height - 2 * this.margin;
      const spacing = availableH / (count + 1);

      let baseR = this.rBase;
      if (type === "output") baseR *= 1.8;
      if (type === "hidden") baseR *= 1.0;
      if (type === "input") baseR *= 1.0;

      let minR =
         type === "output"
            ? this.minRadiusOutput
            : type === "hidden"
            ? this.minRadiusHidden
            : this.minRadiusInput;

      // radius shrinks with spacing but not below minR
      let radiusBySpacing = spacing * 0.4;
      let r = Math.min(baseR, Math.max(radiusBySpacing, minR));

      const nodes = [];
      for (let i = 0; i < count; i++) {
         const y = this.margin + spacing * (i + 1);
         nodes.push({ x, y, r });
      }
      return nodes;
   }
}
