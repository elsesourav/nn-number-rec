"use strict";

class Matrix {
  constructor(rows, cols, data = []) {
    this._rows = rows;
    this._cols = cols;
    this._data = data;
    this._tempMatrix = [];

    //  initialise with zeroes when on data provided 
    if (data.length === 0 || data === null) {
      this._data = [];
      for (let i = 0; i < this.rows; i++) {
        this._data[i] = [];
        for (let j = 0; j < this.cols; j++) {
          this._data[i][j] = 0;
        }
      }
    } else {
      //  chack data row and col is equal in given rows and cols 
      if (data.length !== rows || data[0].length !== cols) {
        throw new Error("Incrrect data dimensions!");
      }
    }
  }

  get rows() {
    return this._rows;
  }
  set rows(rows) {
    this._rows = rows;
  }

  get cols() {
    return this._cols;
  }
  set cols(cols) {
    this._cols = cols;
  }

  get data() {
    return this._data;
  }
  set data(data) {
    return this._data = data;
  }

  get tempMatrix() {
    return this._tempMatrix;
  }
  set tempMatrix(matrix) {
    return this._tempMatrix = matrix;
  }

  //  apply random weights between -1 to 1 
  randomWeights() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = Math.random() * 2 - 1;
      }
    }
  }

  //  apply random weights between -1 to 1 (round value)
  randomWeights(round) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = parseFloat((Math.random() * 2 - 1).toFixed(1));
      }
    }
  }

  //  add to matrixs 
  static add(matrix1, matrix2) {
    Matrix.chackDimensions(matrix1, matrix2);
    this.tempMatrix = new Matrix(matrix1.rows, matrix1.cols);
    for (let i = 0; i < this.tempMatrix.rows; i++) {
      for (let j = 0; j < this.tempMatrix.cols; j++) {
        this.tempMatrix.data[i][j] = matrix1.data[i][j] + matrix2.data[i][j];
      }
    }
    return this.tempMatrix;
  }

  add(matrix2) {
    Matrix.chackDimensions(this, matrix2);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] += matrix2.data[i][j];
      }
    }
  }

  //  subtract to matrixs 
  static subtract(matrix1, matrix2) {
    Matrix.chackDimensions(matrix1, matrix2);
    this.tempMatrix = new Matrix(matrix1.rows, matrix1.cols);
    for (let i = 0; i < this.tempMatrix.rows; i++) {
      for (let j = 0; j < this.tempMatrix.cols; j++) {
        this.tempMatrix.data[i][j] = matrix1.data[i][j] - matrix2.data[i][j];
      }
    }
    return this.tempMatrix;
  }

  subtract(matrix2) {
    Matrix.chackDimensions(this, matrix2);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] -= matrix2.data[i][j];
      }
    }
  }

  //   multiply to matrixs (not dot product) 
  static multiply(matrix1, matrix2) {
    Matrix.chackDimensions(matrix1, matrix2);
    this.tempMatrix = new Matrix(matrix1.rows, matrix1.cols);
    for (let i = 0; i < this.tempMatrix.rows; i++) {
      for (let j = 0; j < this.tempMatrix.cols; j++) {
        this.tempMatrix.data[i][j] = matrix1.data[i][j] * matrix2.data[i][j];
      }
    }
    return this.tempMatrix;
  }

  multiply(matrix2) {
    Matrix.chackDimensions(this, matrix2);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] *= matrix2.data[i][j];
      }
    }
  }

  //  dot to matrixs (not dot product) 
  static dot(matrix1, matrix2) {
    if (matrix1.cols !== matrix2.rows) {
      throw new Error("Matices are not dot compatible!");
    }
    this.tempMatrix = new Matrix(matrix1.rows, matrix2.cols);
    for (let i = 0; i < this.tempMatrix.rows; i++) {
      for (let j = 0; j < this.tempMatrix.cols; j++) {
        let sum = 0;
        for (let k = 0; k < matrix1.cols; k++) {
          sum += matrix1.data[i][k] * matrix2.data[k][j];
        }
        this.tempMatrix.data[i][j] = sum;
      }
    }
    return this.tempMatrix;
  }

  //  convart array to one rowed array 
  static convertFromArray(array) {
    return new Matrix(1, array.length, [array]);
  }

  //  apply a function to each cell of the gavin matrix  
  static map(matrix1, matrixFunction) {
    this.tempMatrix = new Matrix(matrix1.rows, matrix1.cols);
    for (let i = 0; i < this.tempMatrix.rows; i++) {
      for (let j = 0; j < this.tempMatrix.cols; j++) {
        this.tempMatrix.data[i][j] = matrixFunction(matrix1.data[i][j]);
      }
    }
    return this.tempMatrix;
  }

  //  apply a function to each cell of the gavin matrix  
  map(matrixFunction) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = matrixFunction(this.data[i][j]);
      }
    }
  }

  //  find transpose of the gevin mateix 
  static transpose(matrix1) {
    this.tempMatrix = new Matrix(matrix1.cols, matrix1.rows);
    for (let i = 0; i < matrix1.rows; i++) {
      for (let j = 0; j < matrix1.cols; j++) {
        this.tempMatrix.data[j][i] = matrix1.data[i][j];
      }
    }
    return this.tempMatrix;
  }

  //  find transpose of the gevin mateix 
  transpose() {
    this.tempMatrix = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.tempMatrix.data[j][i] = this.data[i][j];
      }
    }
    this.data = this.tempMatrix.data;
  }

  //  chack matices have the same dimensions 
  static chackDimensions(matrix1, matrix2) {
    if (matrix1.rows !== matrix2.rows || matrix1.cols !== matrix2.cols) {
      throw new Error("Matrixs are of different dimensions!");
    }
  }

  // show data
  show() {
    console.table(this.data);
  }
}

