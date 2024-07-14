"use strict";

class NeuralNetwork {
    constructor(numInp, numHid0, numHid1, numOut, lrnRate = 1, lrStape, viweW, viewH) {
        this._inputs = [];
        this._hidden0 = [];
        this._hidden1 = [];
        this._outputs = [];

        this._numInp = numInp;
        this._numHid0 = numHid0;
        this._numHid1 = numHid1;
        this._numOut = numOut;
        this._lrnRate = lrnRate;
        this.lrStape = lrStape;

        this._bias0 = new Matrix(1, this._numHid0);
        this._bias1 = new Matrix(1, this._numHid1);
        this._bias2 = new Matrix(1, this._numOut);

        this._weights0 = new Matrix(this._numInp, this._numHid0);
        this._weights1 = new Matrix(this._numHid0, this._numHid1);
        this._weights2 = new Matrix(this._numHid1, this._numOut);

        // randomize the initial weights
        this._bias0.randomWeights();
        this._bias1.randomWeights();
        this._bias2.randomWeights();
        this._weights0.randomWeights();
        this._weights1.randomWeights();
        this._weights2.randomWeights();

        this.$output = undefined;
        this.$targets = undefined;
        this.$outputErrors = undefined;
        this.$outputDerivs = undefined;
        this.$weights2T = undefined;
        this.$hidden1Errors = undefined;
        this.$hidden1Derivs = undefined;
        this.$hidden1Deltas = undefined;
        this.$weights1T = undefined;
        this.$hidden0Errors = undefined;
        this.$hidden0Derivs = undefined;
        this.$hidden0Deltas = undefined;
        this.$hidden1T = undefined;
        this.$hidden0T = undefined;
        this.$inputsT = undefined;

        this.view = new NNView(this, viweW, viewH);
        this.view.draw();
    }

    feedForward(inputArray) {
        // convert input array to a matrix
        this.inputs = Matrix.convertFromArray(inputArray);

        // find the hidden0 values and apply the activation function
        this.hidden0 = Matrix.dot(this.inputs, this.weights0);
        this.hidden0.add(this.bias0); // apply bias
        this.hidden0.map(x => sigmoid(x));
  
        // find the hidden0 values and apply the activation function
        this.hidden1 = Matrix.dot(this.hidden0, this.weights1);
        this.hidden1.add(this.bias1); // apply bias
        this.hidden1.map(x => sigmoid(x));

        // find the output values and apply the activation function
        this.outputs = Matrix.dot(this.hidden1, this.weights2);
        this.outputs.add(this.bias2); // apply bias
        this.outputs.map(x => sigmoid(x));

        this.view.aniStart();
        return this.outputs;
    }

    train(inputArray, targetArray) {
        // feed the input data through the network
        this.$outputs = this.feedForward(inputArray);

        // calculate the output errors (target - output)
        this.$targets = Matrix.convertFromArray(targetArray);
        this.$outputErrors = Matrix.subtract(this.$targets, this.$outputs);

        // calculate the deltas (errors * derivitive of the output)
        this.$outputDerivs = Matrix.map(this.$outputs, x => sigmoid(x, true));
        this.$outputDeltas = Matrix.multiply(this.$outputErrors, this.$outputDerivs);

        // calculate hidden1 layer errors (deltas "dot" transpose of weights2)
        this.$weights2T = Matrix.transpose(this.weights2);
        this.$hidden1Errors = Matrix.dot(this.$outputDeltas, this.$weights2T);

        // calculate the hidden1 deltas (errors * derivitive of hidden1)
        this.$hidden1Derivs = Matrix.map(this.hidden1, x => sigmoid(x, true));
        this.$hidden1Deltas = Matrix.multiply(this.$hidden1Errors, this.$hidden1Derivs);

        
        // calculate hidden layer errors (deltas "dot" transpose of weights1)
        this.$weights1T = Matrix.transpose(this.weights1);
        this.$hidden0Errors = Matrix.dot(this.$hidden1Deltas, this.$weights1T);

        // calculate the hidden deltas (errors * derivitive of hidden)
        this.$hidden0Derivs = Matrix.map(this.hidden0, x => sigmoid(x, true));
        this.$hidden0Deltas = Matrix.multiply(this.$hidden0Errors, this.$hidden0Derivs);

        // update the weights (add transpose of layers "dot" deltas)
        this.$hidden1T = Matrix.transpose(this.hidden1);
        this.weights2.add(Matrix.map(Matrix.dot(this.$hidden1T, this.$outputDeltas), (x) => x * this.lrnRate));

        this.$hidden0T = Matrix.transpose(this.hidden0);
        this.weights1.add(Matrix.map(Matrix.dot(this.$hidden0T, this.$hidden1Deltas), (x) => x * this.lrnRate));

        this.$inputsT = Matrix.transpose(this.inputs);
        this.weights0.add(Matrix.map(Matrix.dot(this.$inputsT, this.$hidden0Deltas), (x) => x * this.lrnRate));

        // update bias
        this.bias2.add(this.$outputDeltas);
        this.bias1.add(this.$hidden1Deltas);
        this.bias0.add(this.$hidden0Deltas);
    }

    get lrnRate() {
        return this._lrnRate;
    }
    set lrnRate(lrnRate) {
        this._lrnRate = lrnRate;
    }

    get inputs() {
        return this._inputs;
    }
    set inputs(inputs) {
        this._inputs = inputs;
    }

    get hidden0() {
        return this._hidden0;
    }
    set hidden0(hidden) {
        this._hidden0 = hidden;
    }

    get hidden1() {
        return this._hidden1;
    }
    set hidden1(hidden) {
        this._hidden1 = hidden;
    }

    get outputs() {
        return this._outputs;
    }
    set outputs(outputs) {
        this._outputs = outputs;
    }

    get bias0() {
        return this._bias0;
    }
    set bias0(bias) {
        this._bias0 = bias;
    }

    get bias1() {
        return this._bias1;
    }
    set bias1(bias) {
        this._bias1 = bias;
    }

    get bias2() {
        return this._bias2;
    }
    set bias2(bias) {
        this._bias2 = bias;
    }

    get weights0() {
        return this._weights0;
    }
    set weights0(weights) {
        this._weights0 = weights;
    }

    get weights1() {
        return this._weights1;
    }
    set weights1(weights) {
        this._weights1 = weights;
    }

    get weights2() {
        return this._weights2;
    }
    set weights2(weights) {
        this._weights2 = weights;
    }
}

function sigmoid(x, derivative = false) {
    if (derivative) return x * (1 - x);
    return 1 / (1 + Math.exp(-x));
}