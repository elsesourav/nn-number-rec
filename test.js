let _ = null;

createMathModule()
   .then((module) => {
      _ = module;
      document.getElementById("status").textContent =
         "✓ _ loaded successfully!";
      document.getElementById("status").style.color = "green";
      document.getElementById("controls").style.display = "block";
   })
   .catch((err) => {
      document.getElementById("status").textContent =
         "✗ Failed to load: " + err;
      document.getElementById("status").style.color = "red";
   });

// Matrix test
function testMatrix() {
   if (!_) return;

   try {
      let output = "<strong>Matrix Test:</strong><br>";

      // 1. Random Weights
      const m1 = new _.Matrix(2, 2);
      m1.randomWeights();
      output += "Matrix 1 (Random):<br>" + matrixToString(m1) + "<br>";

      // 2. Random Weights Rounded
      const m2 = new _.Matrix(2, 2);
      m2.randomWeightsRounded(true);
      output += "Matrix 2 (Random Rounded):<br>" + matrixToString(m2) + "<br>";

      // 3. Arithmetic (Add, Subtract, Multiply)
      const mA = new _.Matrix([
         [1, 2],
         [3, 4],
      ]);
      const mB = new _.Matrix([
         [10, 20],
         [30, 40],
      ]);

      output += "Matrix A:<br>" + matrixToString(mA) + "<br>";
      output += "Matrix B:<br>" + matrixToString(mB) + "<br>";

      mA.add(mB);
      output += "A.add(B):<br>" + matrixToString(mA) + "<br>";

      mA.subtract(mB);
      output +=
         "A.subtract(B) (Back to original):<br>" + matrixToString(mA) + "<br>";

      const mC = new _.Matrix([
         [2, 2],
         [2, 2],
      ]);
      mA.multiply(mC);
      output += "A.multiply([[2,2],[2,2]]):<br>" + matrixToString(mA) + "<br>";

      // 4. Transpose
      const mT = new _.Matrix([
         [1, 2, 3],
         [4, 5, 6],
      ]);
      output += "Matrix T (2x3):<br>" + matrixToString(mT) + "<br>";

      mT.transpose();
      output += "T.transpose() (3x2):<br>" + matrixToString(mT) + "<br>";

      mT.show();

      // Cleanup
      m1.delete();
      m2.delete();
      mA.delete();
      mB.delete();
      mC.delete();
      mT.delete();

      document.getElementById("result").innerHTML = output;
   } catch (e) {
      console.error(e);
      document.getElementById("result").innerHTML = "Error: " + e.message;
   }
}

// Tensor test
function testTensor() {
   if (!_) return;

   try {
      let output = "<strong>Tensor Test:</strong><br>";

      // 1. Create 3D Tensor from JS Array
      const t = new _.Tensor([
         [
            [1, 2],
            [3, 4],
         ],
         [
            [5, 6],
            [7, 8],
         ],
      ]);
      output += "Created 3D Tensor (2x2x2).<br>";
      output += tensorToString(t) + "<br>";

      // 2. Push a new 2D slice
      t.push([
         [9, 10],
         [11, 12],
      ]);
      output += "Pushed [[9, 10], [11, 12]]:<br>";
      output += tensorToString(t) + "<br>";

      // 3. Insert at index 1
      t.insert(1, [
         [13, 14],
         [15, 16],
      ]);
      output += "Inserted [[13, 14], [15, 16]] at index 1:<br>";
      output += tensorToString(t) + "<br>";

      // 4. Pop
      t.pop();
      output += "Popped last element:<br>";
      output += tensorToString(t) + "<br>";

      // 5. Get value
      const val = t.get([0, 1, 0]); // Should be 3
      output += `Value at [0, 1, 0]: <strong>${val}</strong><br>`;

      // 6. Set value
      t.set([0, 1, 0], 99.9);
      output += `Set [0, 1, 0] to 99.9. New value: <strong>${t.get([
         0, 1, 0,
      ])}</strong><br>`;

      // 7. Slice
      const sliced = t.slice(1, 3);
      output += "Slice(1, 3):<br>";
      output += tensorToString(sliced) + "<br>";

      document.getElementById("result").innerHTML = output;

      t.delete();
      sliced.delete();
   } catch (e) {
      console.error(e);
      document.getElementById("result").innerHTML = "Error: " + e.message;
   }
}

function tensorToString(t) {
   const shapeVec = t.getShape();
   const shape = [];
   for (let i = 0; i < shapeVec.size(); i++) {
      shape.push(shapeVec.get(i));
   }

   const data = t.getData();
   let str = `Shape: [${shape.join(", ")}]<br>Data: [`;

   // Simple flat data print for now, formatting N-D is complex
   for (let i = 0; i < data.size(); i++) {
      str += data.get(i).toFixed(1);
      if (i < data.size() - 1) str += ", ";
   }
   str += "]";

   // Clean up vectors returned by value
   shapeVec.delete();
   data.delete();

   return str;
}
function matrixToString(m) {
   let str = "[<br>";
   const rows = m.getRows();
   const cols = m.getCols();

   for (let i = 0; i < rows; i++) {
      str += "&nbsp;&nbsp;[";
      for (let j = 0; j < cols; j++) {
         str += m.at(i, j).toFixed(2);
         if (j < cols - 1) str += ", ";
      }
      str += "]";
      if (i < rows - 1) str += ",<br>";
   }
   str += "<br>]";
   return str;
}
