# Matrix Class Documentation

The `Matrix` class is a high-performance, 2D dense matrix implementation written in C++ and compiled to WebAssembly. It is designed for neural network computations, supporting essential mathematical operations like addition, subtraction, Hadamard (element-wise) multiplication, and transposition.

## Constructors

### `new Matrix(rows, cols)`

Creates a new matrix with the specified dimensions, initialized with zeros.

-  **Parameters:**
   -  `rows` (number): The number of rows.
   -  `cols` (number): The number of columns.
-  **Example:**
   ```javascript
   const m = new wasmModule.Matrix(3, 4); // 3x4 matrix of zeros
   console.log(m.getRows()); // 3
   console.log(m.getCols()); // 4
   m.delete();
   ```

### `new Matrix(jsArray)`

Creates a new matrix from a JavaScript array of arrays (2D array).

-  **Parameters:**
   -  `jsArray` (Array<Array<number>>): A 2D JavaScript array.
-  **Example:**
   ```javascript
   const m = new wasmModule.Matrix([
      [1, 2, 3],
      [4, 5, 6],
   ]);
   console.log(m.at(1, 0)); // 4
   m.delete();
   ```

---

## Methods

### `getRows()`

Returns the number of rows in the matrix.

-  **Returns:** `number`
-  **Example:**
   ```javascript
   const m = new wasmModule.Matrix(5, 10);
   const rows = m.getRows();
   console.log(rows); // 5
   m.delete();
   ```

### `getCols()`

Returns the number of columns in the matrix.

-  **Returns:** `number`
-  **Example:**
   ```javascript
   const m = new wasmModule.Matrix(5, 10);
   const cols = m.getCols();
   console.log(cols); // 10
   m.delete();
   ```

### `getData()`

Retrieves the underlying data of the matrix as a C++ `vector2d` object.

-  **Returns:** `vector2d` (A C++ proxy object for `std::vector<std::vector<double>>`).
-  **Important:** The returned object is a C++ instance living in WebAssembly memory. You **must** call `.delete()` on it when you are done to prevent memory leaks.
-  **Example:**

   ```javascript
   const m = new wasmModule.Matrix([
      [1, 2],
      [3, 4],
   ]);
   const dataVec = m.getData();

   // Access dataVec using .get(index) which returns a vector1d
   const row0 = dataVec.get(0);
   console.log(row0.get(1)); // 2

   // Cleanup
   row0.delete(); // vector1d returned by value needs cleanup? (Usually emscripten handles primitives, but objects need care)
   dataVec.delete();
   m.delete();
   ```

### `setData(data)`

Sets the matrix data from a C++ `vector2d` object.

-  **Parameters:**
   -  `data` (vector2d): The new data.
-  **Example:**
   ```javascript
   // Advanced usage: usually you construct with array
   // This is mostly for internal C++ interop
   ```

### `randomWeights()`

Fills the matrix with random floating-point numbers between -1.0 and 1.0. This is commonly used for initializing neural network weights.

-  **Returns:** `void` (Modifies the matrix in-place).
-  **Example:**
   ```javascript
   const m = new wasmModule.Matrix(2, 2);
   m.randomWeights();
   m.show();
   // Output might be:
   // [[0.12, -0.54],
   //  [0.99, -0.11]]
   m.delete();
   ```

### `randomWeightsRounded(round)`

Fills the matrix with random numbers. If `round` is true, the numbers are rounded to the nearest integer.

-  **Parameters:**
   -  `round` (boolean): Whether to round the values.
-  **Returns:** `void` (Modifies the matrix in-place).
-  **Example:**
   ```javascript
   const m = new wasmModule.Matrix(2, 2);
   m.randomWeightsRounded(true);
   m.show();
   // Output might be:
   // [[1, 0],
   //  [-1, 1]]
   m.delete();
   ```

### `add(otherMatrix)`

Performs element-wise addition with another matrix. The operation is performed in-place, meaning the current matrix is updated with the result.

-  **Parameters:**
   -  `otherMatrix` (Matrix): The matrix to add. Must have the same dimensions.
-  **Returns:** `void`
-  **Example:**

   ```javascript
   const m1 = new wasmModule.Matrix([
      [1, 2],
      [3, 4],
   ]);
   const m2 = new wasmModule.Matrix([
      [10, 10],
      [10, 10],
   ]);

   m1.add(m2);

   // m1 is now [[11, 12], [13, 14]]
   console.log(m1.at(0, 0)); // 11

   m1.delete();
   m2.delete();
   ```

### `subtract(otherMatrix)`

Performs element-wise subtraction. The operation is performed in-place.

-  **Parameters:**
   -  `otherMatrix` (Matrix): The matrix to subtract. Must have the same dimensions.
-  **Returns:** `void`
-  **Example:**

   ```javascript
   const m1 = new wasmModule.Matrix([
      [10, 20],
      [30, 40],
   ]);
   const m2 = new wasmModule.Matrix([
      [1, 2],
      [3, 4],
   ]);

   m1.subtract(m2);

   // m1 is now [[9, 18], [27, 36]]
   console.log(m1.at(0, 0)); // 9

   m1.delete();
   m2.delete();
   ```

### `multiply(otherMatrix)`

Performs element-wise multiplication (Hadamard product). The operation is performed in-place.

-  **Parameters:**
   -  `otherMatrix` (Matrix): The matrix to multiply with. Must have the same dimensions.
-  **Returns:** `void`
-  **Example:**

   ```javascript
   const m1 = new wasmModule.Matrix([
      [1, 2],
      [3, 4],
   ]);
   const m2 = new wasmModule.Matrix([
      [2, 0.5],
      [2, 0.5],
   ]);

   m1.multiply(m2);

   // m1 is now [[2, 1], [6, 2]]
   console.log(m1.at(0, 1)); // 1

   m1.delete();
   m2.delete();
   ```

### `transpose()`

Transposes the matrix (swaps rows and columns). The operation is performed in-place.

-  **Returns:** `void`
-  **Example:**

   ```javascript
   const m = new wasmModule.Matrix([
      [1, 2, 3],
      [4, 5, 6],
   ]);
   // Shape is 2x3

   m.transpose();
   // Shape is now 3x2
   // [[1, 4],
   //  [2, 5],
   //  [3, 6]]

   console.log(m.getRows()); // 3
   console.log(m.at(0, 1)); // 4
   m.delete();
   ```

### `at(row, col)`

Retrieves the value at a specific row and column index.

-  **Parameters:**
   -  `row` (number): The row index (0-based).
   -  `col` (number): The column index (0-based).
-  **Returns:** `number`
-  **Example:**
   ```javascript
   const m = new wasmModule.Matrix([
      [10, 20],
      [30, 40],
   ]);
   const val = m.at(1, 1);
   console.log(val); // 40
   m.delete();
   ```

### `show()`

Prints the matrix contents to the standard output. In a web browser environment, this output appears in the **developer console** (not on the web page itself).

-  **Returns:** `void`
-  **Example:**

```javascript
const m = new wasmModule.Matrix(2, 2);
m.show();
// Open your browser's developer console to see:
// 0.00 0.00
// 0.00 0.00
m.delete();
```

---

## Memory Management

Since `Matrix` is a C++ class bound to JavaScript, instances are not automatically garbage collected by the JS engine in the same way native JS objects are.

**You must manually delete Matrix instances when they are no longer needed.**

```javascript
const m = new wasmModule.Matrix(100, 100);
// ... perform operations ...
m.delete(); // Frees the memory in the WebAssembly heap
```
