const pixel = 28;
const inpNums = $$(".number");
const numInp = pixel * pixel;
const numHid0 = 64;
const numHid1 = 64;
const numOut = 10;
let pencilSize = pixel / 8;

let _min_ = Math.floor(minSize / 100);
_min_ = window.innerHeight < 600 ? _min_ * 40 : _min_ * 60;
const SIZE = 500;
const CANVAS_SCALE = 2;

cssRoot.style.setProperty("--scale-n", _min_ / 120);
cssRoot.style.setProperty("--scale", `${_min_ / 30}px`);

cssRoot.style.setProperty("--view-w", `${_min_ * 1.4}px`);
cssRoot.style.setProperty("--view-h", `${_min_ * 1.1}px`);

const c = new Canvas(pixel, pixel, ID("cvs"));
let nn;
let wasmModule;

createMathModule().then((module) => {
   wasmModule = module;

   // Instantiate Wasm NeuralNetwork
   // Constructor: (numInp, hiddenSizes, numOut, lrnRate)
   const hiddenSizes = new wasmModule.vectorInt();
   hiddenSizes.push_back(numHid0);
   hiddenSizes.push_back(numHid1);

   nn = new wasmModule.NeuralNetwork(numInp, hiddenSizes, numOut, 1.0);
   hiddenSizes.delete();

   // Initialize with zeros to ensure clean state
   const zeroInput = new wasmModule.vector1d();
   for (let i = 0; i < numInp; i++) zeroInput.push_back(0.0);
   const initOut = nn.feedForwardArray(zeroInput);
   initOut.delete();
   zeroInput.delete();

   // Explicitly reset all activations to 0 (black)
   nn.resetActivations();

   // --- Auto Load Logic ---
   const savedLR = getDataFromLocalStorage("sb-nn-lr");
   if (savedLR) {
      nn.lrnRate = parseFloat(savedLR);
      nn.lrStep = parseFloat(savedLR);
      const learnRateInput = document.getElementById("learn-rate");
      const lrShow = document.getElementById("lr-show");
      if (learnRateInput) learnRateInput.value = savedLR;
      if (lrShow) lrShow.innerText = savedLR;
   }

   const savedData = getDataFromLocalStorage("sb-nn-data");
   if (savedData) {
      const numLayers = nn.getNumLayers();
      for (let i = 0; i < numLayers - 1; i++) {
         if (savedData[`bias${i}`]) {
            const m = new wasmModule.Matrix(savedData[`bias${i}`]);
            nn.setBiases(i, m);
            m.delete();
         }
         if (savedData[`weights${i}`]) {
            const m = new wasmModule.Matrix(savedData[`weights${i}`]);
            nn.setWeights(i, m);
            m.delete();
         }
      }
      console.log("Network restored from auto-save");
   }
   // -----------------------

   let ctxView = null;
   let cvsView = document.getElementById("view-cvs");

   if (cvsView.tagName !== "CANVAS") {
      // If it's a div, create a canvas inside
      const newCanvas = document.createElement("canvas");
      // Double the resolution for sharper rendering
      newCanvas.width = SIZE * 1.4 * CANVAS_SCALE;
      newCanvas.height = SIZE * 1.1 * CANVAS_SCALE;
      // Use CSS to keep the display size the same
      // newCanvas.style.width = `${SIZE * 1.4}px`;
      // newCanvas.style.height = `${SIZE * 1}px`;
      cvsView.appendChild(newCanvas);
      cvsView = newCanvas;
   } else {
      cvsView.width = SIZE * 1.4 * CANVAS_SCALE;
      cvsView.height = SIZE * 1.1 * CANVAS_SCALE;
      // cvsView.style.width = `${SIZE * 1.4}px`;
      // cvsView.style.height = `${SIZE * 1}px`;
   }
   ctxView = cvsView.getContext("2d");

   // Initial draw
   if (nn && ctxView) {
      drawNetwork(ctxView, nn, cvsView.width, cvsView.height, -1);
   }

   // Exposed animation trigger
   window.triggerNNAnimation = () => {
      const startTime = Date.now();
      const duration = 1000; // 1 second animation

      const animate = () => {
         if (!nn || !ctxView) return;

         const elapsed = Date.now() - startTime;
         let progress = elapsed / duration;

         if (progress >= 1) {
            // Animation finished, draw final state
            drawNetwork(ctxView, nn, cvsView.width, cvsView.height, -1);
            return;
         }

         drawNetwork(ctxView, nn, cvsView.width, cvsView.height, progress);
         requestAnimationFrame(animate);
      };
      animate();
   };

   // Exposed static draw (no animation)
   window.drawNNStatic = () => {
      if (nn && ctxView) {
         drawNetwork(ctxView, nn, cvsView.width, cvsView.height, -1);
      }
   };
});

function draw(x, y, r) {
   c.arc(x, y, r / 3);
   c.fill(255);
}

let index,
   ary = [];
function getCanvasData() {
   ary = [];
   const cvsData = c.getImageData(0, 0, pixel, pixel).data;
   let inputs = [];

   // 1. Extract raw grayscale values
   for (let i = 0; i < numInp * 4; i += 4) {
      inputs.push(Math.map(cvsData[i], 0, 255, 0, 1));
   }

   // 2. Find Bounding Box
   let minX = pixel,
      minY = pixel,
      maxX = 0,
      maxY = 0;
   let hasData = false;

   for (let y = 0; y < pixel; y++) {
      for (let x = 0; x < pixel; x++) {
         if (inputs[y * pixel + x] > 0.1) {
            // Threshold for "content"
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            hasData = true;
         }
      }
   }

   if (!hasData) {
      ary = inputs; // Empty image
      return;
   }

   // 3. Center the image (Bounding Box Centering)
   const width = maxX - minX + 1;
   const height = maxY - minY + 1;

   const cx = minX + width / 2;
   const cy = minY + height / 2;

   const newInputs = new Array(numInp).fill(0);

   const shiftX = Math.round(pixel / 2 - cx);
   const shiftY = Math.round(pixel / 2 - cy);

   for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
         const val = inputs[y * pixel + x];
         const newX = x + shiftX;
         const newY = y + shiftY;

         if (newX >= 0 && newX < pixel && newY >= 0 && newY < pixel) {
            newInputs[newY * pixel + newX] = val;
         }
      }
   }

   ary = newInputs;
}

function setOutputIndex(index) {
   let ary = [];
   for (let i = 0; i < numOut; i++) {
      if (i != index) ary.push(0);
      else ary.push(1);
   }
   return ary;
}

function training() {
   if (index != undefined) {
      getCanvasData();
      let outAry = setOutputIndex(index);

      // Convert JS arrays to Wasm vectors
      const inputVec = new wasmModule.vector1d();
      ary.forEach((v) => inputVec.push_back(v));

      const targetVec = new wasmModule.vector1d();
      outAry.forEach((v) => targetVec.push_back(v));

      nn.trainArray(inputVec, targetVec);

      inputVec.delete();
      targetVec.delete();

      if (window.triggerNNAnimation) window.triggerNNAnimation();

      console.clear();
   }
}

function clearBoard() {
   c.clrScr();
   c.background(0);
   index = undefined;
}
clearBoard();
