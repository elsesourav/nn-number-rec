# Tensor Class Documentation

The `Tensor` class provides a flexible N-dimensional array container. Unlike `Matrix`, which is strictly 2D and optimized for math, `Tensor` is designed for data manipulation, supporting arbitrary dimensions and dynamic resizing along the first axis (axis 0).

## Constructors

### `new Tensor(jsArray)`

Creates a new Tensor from a nested JavaScript array. The constructor recursively parses the array to determine the shape and data.

-  **Parameters:**
   -  `jsArray` (Array): A nested JavaScript array (e.g., `[[1, 2], [3, 4]]`). All sub-arrays at the same level must have the same length (no ragged arrays).
-  **Example:**
   ```javascript
   // Create a 2x2x2 tensor (3D)
   const t = new wasmModule.Tensor([
      [
         [1, 2],
         [3, 4],
      ],
      [
         [5, 6],
         [7, 8],
      ],
   ]);
   console.log(t.get([0, 0, 0])); // 1
   t.delete();
   ```

---

## Methods

### `getShape()`

Returns the shape of the tensor as a C++ `vectorInt`.

-  **Returns:** `vectorInt` (A C++ proxy for `std::vector<int>`).
-  **Important:** You must call `.delete()` on the returned vector.
-  **Example:**

   ```javascript
   const t = new wasmModule.Tensor([
      [1, 2],
      [3, 4],
   ]);
   const shapeVec = t.getShape();

   // Iterate to read shape
   const shape = [];
   for (let i = 0; i < shapeVec.size(); i++) {
      shape.push(shapeVec.get(i));
   }
   console.log(shape); // [2, 2]

   shapeVec.delete();
   t.delete();
   ```

### `getData()`

Returns the flattened data of the tensor as a C++ `vector1d`.

-  **Returns:** `vector1d` (A C++ proxy for `std::vector<double>`).
-  **Important:** You must call `.delete()` on the returned vector.
-  **Example:**

   ```javascript
   const t = new wasmModule.Tensor([
      [1, 2],
      [3, 4],
   ]);
   const dataVec = t.getData();

   console.log(dataVec.get(0)); // 1
   console.log(dataVec.get(3)); // 4

   dataVec.delete();
   t.delete();
   ```

### `get(indices)`

Retrieves a value from the tensor at the specified N-dimensional index.

-  **Parameters:**
   -  `indices` (Array<number>): A JavaScript array of integers representing the index (e.g., `[0, 1, 2]`).
-  **Returns:** `number`
-  **Example:**
   ```javascript
   const t = new wasmModule.Tensor([
      [10, 20],
      [30, 40],
   ]);
   const val = t.get([1, 0]);
   console.log(val); // 30
   t.delete();
   ```

### `set(indices, value)`

Sets the value at the specified N-dimensional index.

-  **Parameters:**
   -  `indices` (Array<number>): A JavaScript array of integers representing the index.
   -  `value` (number): The new value.
-  **Returns:** `void`
-  **Example:**
   ```javascript
   const t = new wasmModule.Tensor([1, 2, 3]);
   t.set([1], 99.5);
   console.log(t.get([1])); // 99.5
   t.delete();
   ```

### `push(item)`

Appends a new sub-tensor to the end of the tensor along the first dimension (axis 0). The `item` must match the shape of the existing sub-tensors (i.e., `shape[1:]`).

-  **Parameters:**
   -  `item` (Array): A nested JavaScript array representing the sub-tensor to push.
-  **Returns:** `void`
-  **Example:**

   ```javascript
   const t = new wasmModule.Tensor([
      [1, 2],
      [3, 4],
   ]); // Shape [2, 2]

   // Push a new row [5, 6]
   t.push([5, 6]);

   // Shape is now [3, 2]
   console.log(t.get([2, 0])); // 5
   t.delete();
   ```

### `pop()`

Removes the last element along the first dimension (axis 0).

-  **Returns:** `void`
-  **Example:**
   ```javascript
   const t = new wasmModule.Tensor([1, 2, 3]);
   t.pop();
   // Shape is now [2], data is [1, 2]
   t.delete();
   ```

### `insert(index, item)`

Inserts a new sub-tensor at the specified index along the first dimension (axis 0).

-  **Parameters:**
   -  `index` (number): The index at which to insert.
   -  `item` (Array): The nested JavaScript array to insert.
-  **Returns:** `void`
-  **Example:**

   ```javascript
   const t = new wasmModule.Tensor([
      [1, 2],
      [5, 6],
   ]);

   // Insert [3, 4] at index 1
   t.insert(1, [3, 4]);

   // Result: [[1, 2], [3, 4], [5, 6]]
   console.log(t.get([1, 0])); // 3
   t.delete();
   ```

### `slice(start, end)`

Creates a new Tensor that is a slice of the original tensor along the first dimension (axis 0).

-  **Parameters:**
   -  `start` (number): The starting index (inclusive).
   -  `end` (number, optional): The ending index (exclusive). If omitted, slices to the end.
-  **Returns:** `Tensor` (A new Tensor instance).
-  **Important:** The returned Tensor is a new C++ object and must be manually deleted.
-  **Example:**

   ```javascript
   const t = new wasmModule.Tensor([10, 20, 30, 40, 50]);

   // Slice from index 1 to 3 (exclusive) -> [20, 30]
   const subTensor = t.slice(1, 3);

   console.log(subTensor.get([0])); // 20

   subTensor.delete();
   t.delete();
   ```

### `show()`

Prints the tensor's shape and data to the standard output. In a web browser environment, this output appears in the **developer console**.

-  **Returns:** `void`
-  **Example:**

```javascript
const t = new wasmModule.Tensor([1, 2, 3]);
t.show();
// Open your browser's developer console to see:
// Tensor shape: [3], size: 3
t.delete();
```

---

## Memory Management

Like `Matrix`, `Tensor` instances are C++ objects.

**You must manually delete Tensor instances when they are no longer needed.**

```javascript
const t = new wasmModule.Tensor([...]);
// ...
t.delete();
```
