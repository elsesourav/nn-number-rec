import type { NNEngineModule, WasmNeuralNetwork, SerializedModel } from './types';

export class NeuralNetworkBridge {
  private module: NNEngineModule;
  private nn: WasmNeuralNetwork;
  private topology: number[];
  private inputBufferView: Float32Array;
  private outputBufferView: Float32Array;
  private targetPtr: number = 0;
  private targetBufferView: Float32Array | null = null;
  private isDisposed: boolean = false;
  private _useBias: boolean = true;

  constructor(
    module: NNEngineModule,
    topology: number[],
    learningRate: number = 0.1,
    momentum: number = 0.85,
    hiddenAct: string = 'sigmoid',
    outputAct: string = 'sigmoid',
    useBias: boolean = true
  ) {
    this.module = module;
    this.topology = [...topology];
    this._useBias = useBias;

    // Allocate topology vector in WASM
    const topoVec = new module.VectorInt();
    for (const size of topology) {
      topoVec.push_back(size);
    }

    this.nn = new module.NeuralNetwork(topoVec, learningRate, momentum, hiddenAct, outputAct);
    topoVec.delete();

    // Setup zero-copy Float32Array views over WASM Heap
    const inPtr = this.nn.getInputBufferPtr();
    const outPtr = this.nn.getOutputBufferPtr();
    const inDim = this.nn.getInputSize();
    const outDim = this.nn.getOutputSize();

    this.inputBufferView = new Float32Array(this.module.HEAPF32.buffer, inPtr, inDim);
    this.outputBufferView = new Float32Array(this.module.HEAPF32.buffer, outPtr, outDim);

    // Pre-allocate target buffer for training to avoid allocation in hot loops
    this.targetPtr = this.module._malloc(outDim * Float32Array.BYTES_PER_ELEMENT);
    this.targetBufferView = new Float32Array(this.module.HEAPF32.buffer, this.targetPtr, outDim);

    if (!useBias) {
      this.zeroBiases();
    }
  }

  public get useBias(): boolean {
    return this._useBias;
  }

  public set useBias(val: boolean) {
    this._useBias = val;
    if (!val) {
      this.zeroBiases();
    }
  }

  public zeroBiases(): void {
    if (this.isDisposed) return;
    const numLayers = this.nn.getNumLayers();
    for (let l = 0; l < numLayers - 1; l++) {
      const bPtr = this.nn.getBiasesPtr(l);
      const bSize = this.nn.getBiasesSize(l);
      if (bPtr && bSize > 0) {
        const bView = new Float32Array(this.module.HEAPF32.buffer, bPtr, bSize);
        bView.fill(0);
      }
    }
  }

  public get disposed(): boolean {
    return this.isDisposed;
  }

  /**
   * Refreshes heap views in case WebAssembly memory grew.
   */
  private checkMemoryViews() {
    if (this.isDisposed) return;
    if (this.inputBufferView.buffer !== this.module.HEAPF32.buffer) {
      const inPtr = this.nn.getInputBufferPtr();
      const outPtr = this.nn.getOutputBufferPtr();
      const inDim = this.nn.getInputSize();
      const outDim = this.nn.getOutputSize();

      this.inputBufferView = new Float32Array(this.module.HEAPF32.buffer, inPtr, inDim);
      this.outputBufferView = new Float32Array(this.module.HEAPF32.buffer, outPtr, outDim);
      if (this.targetPtr) {
        this.targetBufferView = new Float32Array(this.module.HEAPF32.buffer, this.targetPtr, outDim);
      }
    }
  }

  /**
   * Zero-copy feed-forward inference.
   * Copies input pixels directly into WASM linear memory and executes C++ kernel.
   */
  public predict(pixels: Float32Array): Float32Array {
    if (this.isDisposed) {
      return new Float32Array(this.topology[this.topology.length - 1]);
    }
    this.checkMemoryViews();
    this.inputBufferView.set(pixels);
    this.nn.feedForward();
    return new Float32Array(this.outputBufferView);
  }

  /**
   * Single-sample training with backpropagation.
   */
  public trainSingle(pixels: Float32Array, target: Float32Array): number {
    if (this.isDisposed) return 0;
    this.checkMemoryViews();
    this.inputBufferView.set(pixels);
    if (this.targetBufferView) {
      this.targetBufferView.set(target);
    }
    const loss = this.nn.train(this.nn.getInputBufferPtr(), this.targetPtr);
    if (!this._useBias) {
      this.zeroBiases();
    }
    return loss;
  }

  /**
   * Mini-batch training for high performance.
   */
  public trainBatch(inputs: Float32Array, targets: Float32Array, batchSize: number): number {
    if (this.isDisposed) return 0;
    this.checkMemoryViews();
    const inBytes = inputs.byteLength;
    const tgtBytes = targets.byteLength;

    const inPtr = this.module._malloc(inBytes);
    const tgtPtr = this.module._malloc(tgtBytes);

    const inView = new Float32Array(this.module.HEAPF32.buffer, inPtr, inputs.length);
    const tgtView = new Float32Array(this.module.HEAPF32.buffer, tgtPtr, targets.length);

    inView.set(inputs);
    tgtView.set(targets);

    const avgLoss = this.nn.trainBatch(inPtr, tgtPtr, batchSize);

    this.module._free(inPtr);
    this.module._free(tgtPtr);

    return avgLoss;
  }

  // Visualizer Introspection Getters
  public getNumLayers(): number {
    if (this.isDisposed) return 0;
    return this.nn.getNumLayers();
  }

  public getLayerSize(layerIdx: number): number {
    if (this.isDisposed) return 0;
    return this.nn.getLayerSize(layerIdx);
  }

  public getNeuronValue(layerIdx: number, neuronIdx: number): number {
    if (this.isDisposed) return 0;
    return this.nn.getNeuronValue(layerIdx, neuronIdx);
  }

  public getWeightValue(layerIdx: number, fromIdx: number, toIdx: number): number {
    if (this.isDisposed) return 0;
    return this.nn.getWeightValue(layerIdx, fromIdx, toIdx);
  }

  public getBiasValue(layerIdx: number, neuronIdx: number): number {
    if (this.isDisposed) return 0;
    return this.nn.getBiasValue(layerIdx, neuronIdx);
  }

  public resetActivations(): void {
    if (this.isDisposed) return;
    this.nn.resetActivations();
  }

  // Hyperparameters
  public get learningRate(): number {
    if (this.isDisposed) return 0.1;
    return this.nn.learningRate;
  }

  public set learningRate(val: number) {
    if (this.isDisposed) return;
    this.nn.learningRate = val;
  }

  public get momentum(): number {
    if (this.isDisposed) return 0.85;
    return this.nn.momentum;
  }

  public set momentum(val: number) {
    if (this.isDisposed) return;
    this.nn.momentum = val;
  }

  public getTopology(): number[] {
    return [...this.topology];
  }

  /**
   * Export the network weights, biases, and topology to a structured JSON object.
   */
  public exportModel(pixelResolution: number): SerializedModel {
    if (this.isDisposed) {
      return {
        version: 2,
        topology: [...this.topology],
        learningRate: 0.1,
        momentum: 0.85,
        pixelResolution,
        layers: [],
      };
    }
    this.checkMemoryViews();
    const numLayers = this.nn.getNumLayers();
    const layers = [];

    for (let l = 0; l < numLayers - 1; l++) {
      const wSize = this.nn.getWeightsSize(l);
      const bSize = this.nn.getBiasesSize(l);
      const wPtr = this.nn.getWeightsPtr(l);
      const bPtr = this.nn.getBiasesPtr(l);

      const wView = new Float32Array(this.module.HEAPF32.buffer, wPtr, wSize);
      const bView = new Float32Array(this.module.HEAPF32.buffer, bPtr, bSize);

      layers.push({
        weights: Array.from(wView),
        biases: Array.from(bView),
      });
    }

    return {
      version: 2,
      topology: [...this.topology],
      learningRate: this.learningRate,
      momentum: this.momentum,
      pixelResolution,
      layers,
    };
  }

  /**
   * Imports weights and biases from either modern format or legacy 2D format.
   */
  public importModel(model: SerializedModel): boolean {
    if (this.isDisposed) return false;
    this.checkMemoryViews();
    const numLayers = this.nn.getNumLayers();

    // 1. Modern Version 2 Format
    if (model.version === 2 && model.layers) {
      for (let l = 0; l < Math.min(numLayers - 1, model.layers.length); l++) {
        const layerData = model.layers[l];
        const wPtr = this.nn.getWeightsPtr(l);
        const bPtr = this.nn.getBiasesPtr(l);
        const wSize = this.nn.getWeightsSize(l);
        const bSize = this.nn.getBiasesSize(l);

        if (layerData.weights && wPtr) {
          const wView = new Float32Array(this.module.HEAPF32.buffer, wPtr, wSize);
          const count = Math.min(layerData.weights.length, wSize);
          for (let i = 0; i < count; i++) wView[i] = layerData.weights[i];
        }

        if (layerData.biases && bPtr) {
          const bView = new Float32Array(this.module.HEAPF32.buffer, bPtr, bSize);
          const count = Math.min(layerData.biases.length, bSize);
          for (let i = 0; i < count; i++) bView[i] = layerData.biases[i];
        }
      }
      if (typeof model.learningRate === 'number') this.learningRate = model.learningRate;
      if (typeof model.momentum === 'number') this.momentum = model.momentum;
      return true;
    }

    // 2. Legacy model format support: { weights0: [[...]], bias0: [[...]], ... }
    let loadedAny = false;
    for (let l = 0; l < numLayers - 1; l++) {
      const wKey = `weights${l}`;
      const bKey = `bias${l}`;

      if (model[wKey] && Array.isArray(model[wKey])) {
        const flatWeights: number[] = [];
        const rawW = model[wKey] as (number[] | number)[];
        for (const row of rawW) {
          if (Array.isArray(row)) {
            for (const val of row) flatWeights.push(val);
          } else {
            flatWeights.push(row);
          }
        }
        const wPtr = this.nn.getWeightsPtr(l);
        const wSize = this.nn.getWeightsSize(l);
        if (wPtr) {
          const wView = new Float32Array(this.module.HEAPF32.buffer, wPtr, wSize);
          const count = Math.min(flatWeights.length, wSize);
          for (let i = 0; i < count; i++) wView[i] = flatWeights[i];
          loadedAny = true;
        }
      }

      if (model[bKey] && Array.isArray(model[bKey])) {
        const flatBiases: number[] = [];
        const rawB = model[bKey] as (number[] | number)[];
        for (const row of rawB) {
          if (Array.isArray(row)) {
            for (const val of row) flatBiases.push(val);
          } else {
            flatBiases.push(row);
          }
        }
        const bPtr = this.nn.getBiasesPtr(l);
        const bSize = this.nn.getBiasesSize(l);
        if (bPtr) {
          const bView = new Float32Array(this.module.HEAPF32.buffer, bPtr, bSize);
          const count = Math.min(flatBiases.length, bSize);
          for (let i = 0; i < count; i++) bView[i] = flatBiases[i];
          loadedAny = true;
        }
      }
    }

    return loadedAny;
  }

  public dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;

    if (this.targetPtr) {
      try {
        this.module._free(this.targetPtr);
      } catch (e) {
        // safe ignore
      }
      this.targetPtr = 0;
      this.targetBufferView = null;
    }

    try {
      if (this.nn && typeof this.nn.delete === 'function') {
        this.nn.delete();
      }
    } catch (e) {
      // safe ignore in case Embind already freed it
    }
  }
}
