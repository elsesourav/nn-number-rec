const fs = require("fs");
const zlib = require("zlib");
const path = require("path");
const createMathModule = require("./wasmJs/wasm.js");

const IMAGES_BASE = "train-images-idx3-ubyte";
const LABELS_BASE = "train-labels-idx1-ubyte";
const OUTPUT_FILE = "model.json";

// Configuration
const TARGET_PIXEL = 28;
const NUM_INP = TARGET_PIXEL * TARGET_PIXEL;
const NUM_HID0 = 64;
const NUM_HID1 = 64;
const NUM_OUT = 10;
const LEARNING_RATE = 0.1;
const BATCH_SIZE = 1; // Train in batches
const EPOCHS = 3;

function loadFile(baseName) {
   if (fs.existsSync(baseName + ".gz")) {
      console.log(`Loading ${baseName}.gz...`);
      return zlib.gunzipSync(fs.readFileSync(baseName + ".gz"));
   }
   if (fs.existsSync(baseName)) {
      console.log(`Loading ${baseName}...`);
      return fs.readFileSync(baseName);
   }
   return null;
}

async function loadMNIST() {
   const imagesBuffer = loadFile(IMAGES_BASE);
   const labelsBuffer = loadFile(LABELS_BASE);

   if (!imagesBuffer) {
      if (
         fs.existsSync("t10k-images.idx3-ubyte") ||
         fs.existsSync("t10k-images.idx3-ubyte.zip") ||
         fs.existsSync("t10k-images.idx3-ubyte.gz")
      ) {
         throw new Error(
            "Found 't10k-images' (Test Set) but expected 'train-images' (Training Set). Please download 'train-images-idx3-ubyte.gz'."
         );
      }
      throw new Error(`File not found: ${IMAGES_BASE}.gz (or uncompressed)`);
   }
   if (!labelsBuffer) {
      throw new Error(`File not found: ${LABELS_BASE}.gz (or uncompressed)`);
   }

   console.log("Parsing MNIST data...");

   // Parse Images
   // Magic: 4 bytes, Count: 4 bytes, Rows: 4 bytes, Cols: 4 bytes
   const imgCount = imagesBuffer.readUInt32BE(4);
   const rows = imagesBuffer.readUInt32BE(8);
   const cols = imagesBuffer.readUInt32BE(12);

   console.log(`Found ${imgCount} images (${rows}x${cols})`);

   // Parse Labels
   // Magic: 4 bytes, Count: 4 bytes
   const lblCount = labelsBuffer.readUInt32BE(4);
   console.log(`Found ${lblCount} labels`);

   if (imgCount !== lblCount) {
      throw new Error(
         `Image count (${imgCount}) and label count (${lblCount}) do not match. Ensure you are using the matching train/test sets.`
      );
   }

   const images = [];
   const labels = [];

   let imgOffset = 16;
   let lblOffset = 8;

   for (let i = 0; i < imgCount; i++) {
      const pixels = [];
      for (let j = 0; j < rows * cols; j++) {
         let val = imagesBuffer[imgOffset + j];
         let normVal = val / 255.0;
         // Use grayscale (0.0 to 1.0) to match anti-aliased canvas input better
         pixels.push(normVal);
      }
      imgOffset += rows * cols;
      images.push(pixels);

      // Read label
      const label = labelsBuffer[lblOffset++];
      // One-hot encode
      const target = new Array(NUM_OUT).fill(0);
      target[label] = 1;
      labels.push(target);
   }

   return { images, labels };
}

// Shuffle helper
function shuffle(images, labels) {
   for (let i = images.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [images[i], images[j]] = [images[j], images[i]];
      [labels[i], labels[j]] = [labels[j], labels[i]];
   }
}

async function main() {
   try {
      const wasmModule = await createMathModule();
      console.log("Wasm module loaded.");

      const { images, labels } = await loadMNIST();

      // Initialize Neural Network
      const hiddenSizes = new wasmModule.vectorInt();
      hiddenSizes.push_back(NUM_HID0);
      hiddenSizes.push_back(NUM_HID1);

      const nn = new wasmModule.NeuralNetwork(
         NUM_INP,
         hiddenSizes,
         NUM_OUT,
         LEARNING_RATE
      );
      console.log("Neural Network initialized.");

      // Pre-allocate vectors for batch training to avoid GC overhead
      const inputsVec = new wasmModule.vector1d();
      const targetsVec = new wasmModule.vector1d();

      // Resize once (fill with zeros)
      inputsVec.resize(BATCH_SIZE * NUM_INP, 0);
      targetsVec.resize(BATCH_SIZE * NUM_OUT, 0);

      // Training Loop
      console.log(`Starting training for ${EPOCHS} epochs...`);

      for (let epoch = 0; epoch < EPOCHS; epoch++) {
         console.log(`Epoch ${epoch + 1}/${EPOCHS}`);

         // Shuffle data each epoch
         shuffle(images, labels);

         for (let i = 0; i < images.length; i += BATCH_SIZE) {
            const currentBatchSize = Math.min(BATCH_SIZE, images.length - i);

            // Fill the pre-allocated vectors
            for (let j = 0; j < currentBatchSize; j++) {
               const idx = i + j;
               const img = images[idx];
               const lbl = labels[idx];

               // Direct set is faster than push_back
               for (let k = 0; k < NUM_INP; k++) {
                  inputsVec.set(j * NUM_INP + k, img[k]);
               }
               for (let k = 0; k < NUM_OUT; k++) {
                  targetsVec.set(j * NUM_OUT + k, lbl[k]);
               }
            }

            nn.trainBatch(inputsVec, targetsVec, currentBatchSize);

            if ((i + currentBatchSize) % 1000 === 0) {
               process.stdout.write(
                  `\rProcessed ${i + currentBatchSize}/${images.length} images`
               );
            }
         }
         console.log("\nEpoch complete.");
      }

      inputsVec.delete();
      targetsVec.delete();

      // Save Model
      console.log("Saving model...");
      const numLayers = nn.getNumLayers();
      const obj = {};

      // Helper to convert Matrix to JS array
      const getMData = (m) => {
         const data = [];
         const rows = m.getRows();
         const cols = m.getCols();
         for (let i = 0; i < rows; i++) {
            const row = [];
            for (let j = 0; j < cols; j++) {
               row.push(m.at(i, j));
            }
            data.push(row);
         }
         return data;
      };

      for (let i = 0; i < numLayers - 1; i++) {
         obj[`bias${i}`] = getMData(nn.getBiases(i));
         obj[`weights${i}`] = getMData(nn.getWeights(i));
      }

      fs.writeFileSync(OUTPUT_FILE, JSON.stringify(obj));
      console.log(`Model saved to ${OUTPUT_FILE}`);
   } catch (err) {
      console.error("Error:", err);
   }
}

main();
