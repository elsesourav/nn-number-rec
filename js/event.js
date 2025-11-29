/* ----------- event listener ----------- */
const mainDiv = $("main");
const menuButton = ID("menu-icon");
const clearBtn = ID("clear");
const trainingBtn = ID("training");
const trainBtn = ID("train");
const numInpBtn = ID("num-btn");
const learnRate = ID("learn-rate");
const lrShow = ID("lr-show");
const saveBtn = ID("save");
const resetBtn = ID("reset");
const hoverClass = $$(".hover");

// draw events for mouse and touch
let offsetX,
   offsetY,
   isDraw = false;
let cvsOffset = c.cvs.getBoundingClientRect();

window.addEventListener("resize", () => {
   cvsOffset = c.cvs.getBoundingClientRect();
})

c.on("click", (e) => {
   offsetX = Math.map(
      e.clientX,
      cvsOffset.left,
      cvsOffset.left + c.cvs.clientWidth,
      0,
      pixel
   );
   offsetY = Math.map(
      e.clientY,
      cvsOffset.top,
      cvsOffset.top + c.cvs.clientHeight,
      0,
      pixel
   );
   draw(Math.round(offsetX), Math.round(offsetY), Math.round(pixel / 6));
});
c.on("mousemove", (e) => {
   if (isDraw) {
      offsetX = Math.map(
         e.clientX,
         cvsOffset.left,
         cvsOffset.left + c.cvs.clientWidth,
         0,
         pixel
      );
      offsetY = Math.map(
         e.clientY,
         cvsOffset.top,
         cvsOffset.top + c.cvs.clientHeight,
         0,
         pixel
      );
      draw(Math.round(offsetX), Math.round(offsetY), Math.round(pixel / 6));
   }
});
c.on("touchmove", (e) => {
   if (isDraw) {
      offsetX = Math.map(
         e.touches[0].clientX,
         cvsOffset.left,
         cvsOffset.left + c.cvs.clientWidth,
         0,
         pixel
      );
      offsetY = Math.map(
         e.touches[0].clientY,
         cvsOffset.top,
         cvsOffset.top + c.cvs.clientHeight,
         0,
         pixel
      );
      draw(Math.round(offsetX), Math.round(offsetY), Math.round(pixel / 6));
   }
});

c.on("mousedown", () => (isDraw = true));
document.body.addEventListener("mouseup", () => {
   isDraw = false;
});
let numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
c.on("mouseup", () => {
   getCanvasData();

   // Convert JS array to Wasm vector
   const inputVec = new wasmModule.vector1d();
   ary.forEach((v) => inputVec.push_back(v));

   let outputMatrix = nn.feedForwardArray(inputVec);
   inputVec.delete();

   // Trigger animation
   if (window.triggerNNAnimation) window.triggerNNAnimation();

   let result = [];
   let cols = outputMatrix.getCols();
   for (let i = 0; i < cols; i++) result.push(outputMatrix.at(0, i));
   outputMatrix.delete(); // Clean up result matrix

   result = result.map((x) => Math.round(x));
   // outputs.innerText = numbers[getIndexWhereOne(result)];
});

c.on("touchstart", () => (isDraw = true));
c.on("touchend", () => {
   isDraw = false;
   getCanvasData();

   // Convert JS array to Wasm vector
   const inputVec = new wasmModule.vector1d();
   ary.forEach((v) => inputVec.push_back(v));

   let outputMatrix = nn.feedForwardArray(inputVec);
   inputVec.delete();

   // Trigger animation
   if (window.triggerNNAnimation) window.triggerNNAnimation();

   let result = [];
   let cols = outputMatrix.getCols();
   for (let i = 0; i < cols; i++) result.push(outputMatrix.at(0, i));
   outputMatrix.delete(); // Clean up result matrix

   result = result.map((x) => Math.round(x));
   //     outputs.innerText = numbers[getIndexWhereOne(result)];
});

inpNums.each((e, i) => {
   e.addEventListener("click", () => {
      removeClass(inpNums, "on");
      index = i;
      addClass(e, "on");
   });
});

clearBtn.on("click", () => {
   clearBoard();
   removeClass(inpNums, "on");

   if (nn.resetActivations) nn.resetActivations();
   if (window.drawNNStatic) window.drawNNStatic();
});
trainingBtn.on("click", () => {
   trainingBtn.classList.toggle("on");
   numInpBtn.classList.toggle("active");
   trainBtn.classList.toggle("active");
});
trainBtn.on("click", () => {
   training();
   trainBtn.classList.toggle("on");
});

if (isMobile) {
   cssRoot.style.setProperty("--cursor", "auto");
}

menuButton.on("click", () => {
   mainDiv.classList.toggle("active");
});

learnRate.on("input", (e) => {
   let val = parseFloat(e.target.value);
   nn.lrStep = val;
   val = parseFloat(val.toFixed(3));
   lrShow.innerText = nn.lrnRate = val;
});
saveBtn.on("click", () => {
   const getMData = (m) => {
      const rows = m.getRows();
      const cols = m.getCols();
      const data = [];
      for (let i = 0; i < rows; i++) {
         const row = [];
         for (let j = 0; j < cols; j++) row.push(m.at(i, j));
         data.push(row);
      }
      return data;
   };

   // Assuming 3 layers of weights/biases for a 4-layer network (Input -> H1 -> H2 -> Output)
   // Or check getNumLayers()
   const numLayers = nn.getNumLayers();
   const obj = {
      lrnRate: nn.lrnRate,
      lrStep: nn.lrStep,
   };

   // Save all weights and biases
   for (let i = 0; i < numLayers - 1; i++) {
      obj[`bias${i}`] = getMData(nn.getBiases(i));
      obj[`weights${i}`] = getMData(nn.getWeights(i));
   }

   setDataFromLocalStorage("sb-nn", obj);
});
resetBtn.on("click", () => {
   setDataFromLocalStorage("sb-nn", "");
});
