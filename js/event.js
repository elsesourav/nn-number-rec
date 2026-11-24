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
const loadJsonBtn = ID("load-json-btn");
const loadJsonInput = ID("load-json-input");
const resetBtn = ID("reset");
const hoverClass = $$(".hover");

// draw events for mouse and touch
let offsetX,
   offsetY,
   isDraw = false;
let cvsOffset = c.cvs.getBoundingClientRect();

window.addEventListener("resize", () => {
   cvsOffset = c.cvs.getBoundingClientRect();
});

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
   draw(Math.round(offsetX), Math.round(offsetY), Math.round(pencilSize));
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
      draw(Math.round(offsetX), Math.round(offsetY), Math.round(pencilSize));
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
      draw(Math.round(offsetX), Math.round(offsetY), Math.round(pencilSize));
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
   document.body.classList.toggle("menu-open");
});

learnRate.on("input", (e) => {
   let val = parseFloat(e.target.value);
   nn.lrStep = val;
   val = parseFloat(val.toFixed(2));
   lrShow.innerText = nn.lrnRate = val;
   setDataFromLocalStorage("sb-nn-lr", val);
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

   const numLayers = nn.getNumLayers();
   const obj = {};

   // Save all weights and biases
   for (let i = 0; i < numLayers - 1; i++) {
      obj[`bias${i}`] = getMData(nn.getBiases(i));
      obj[`weights${i}`] = getMData(nn.getWeights(i));
   }

   setDataFromLocalStorage("sb-nn-data", obj);
   showToast("Network Saved!");
});

function loadModel(obj) {
   const numLayers = nn.getNumLayers();

   for (let i = 0; i < numLayers - 1; i++) {
      if (obj[`weights${i}`] && obj[`bias${i}`]) {
         // Load Weights
         const wData = obj[`weights${i}`];
         const wRows = wData.length;
         const wCols = wData[0].length;

         const wVec2d = new wasmModule.vector2d();
         wData.forEach((row) => {
            const v = new wasmModule.vector1d();
            row.forEach((val) => v.push_back(val));
            wVec2d.push_back(v);
            v.delete();
         });

         const wMatrix = new wasmModule.Matrix(wRows, wCols, wVec2d);
         nn.setWeights(i, wMatrix);

         wMatrix.delete();
         wVec2d.delete();

         // Load Biases
         const bData = obj[`bias${i}`];
         const bRows = bData.length;
         const bCols = bData[0].length;

         const bVec2d = new wasmModule.vector2d();
         bData.forEach((row) => {
            const v = new wasmModule.vector1d();
            row.forEach((val) => v.push_back(val));
            bVec2d.push_back(v);
            v.delete();
         });

         const bMatrix = new wasmModule.Matrix(bRows, bCols, bVec2d);
         nn.setBiases(i, bMatrix);

         bMatrix.delete();
         bVec2d.delete();
      }
   }

   // Also save to local storage so it persists
   setDataFromLocalStorage("sb-nn-data", obj);
   showToast("Model Loaded!");

   // Redraw
   if (window.drawNNStatic) window.drawNNStatic();
}

loadJsonBtn.on("click", () => {
   fetch("model.json")
      .then((res) => {
         if (!res.ok) throw new Error("Failed to fetch model.json");
         return res.json();
      })
      .then((obj) => {
         loadModel(obj);
      })
      .catch((err) => {
         console.warn(
            "Auto-load failed (likely due to file:// protocol), opening file picker...",
            err
         );
         loadJsonInput.click();
      });
});

loadJsonInput.on("change", (e) => {
   const file = e.target.files[0];
   if (!file) return;

   const reader = new FileReader();
   reader.onload = (event) => {
      try {
         const obj = JSON.parse(event.target.result);
         loadModel(obj);
      } catch (err) {
         console.error(err);
         showToast("Error loading JSON");
      }
   };
   reader.readAsText(file);
   // Reset input so same file can be selected again
   e.target.value = "";
});

resetBtn.on("click", () => {
   setDataFromLocalStorage("sb-nn-data", "");
   setDataFromLocalStorage("sb-nn-lr", "");
   location.reload();
});

// Close menu when clicking outside
document.addEventListener("click", (e) => {
   if (
      document.body.classList.contains("menu-open") &&
      !mainDiv.contains(e.target) &&
      !menuButton.contains(e.target)
   ) {
      mainDiv.classList.remove("active");
      document.body.classList.remove("menu-open");
   }
});

function showToast(msg) {
   const toast = document.getElementById("toast");
   toast.innerText = msg;
   toast.classList.add("show");
   setTimeout(() => {
      toast.classList.remove("show");
   }, 3000);
}
