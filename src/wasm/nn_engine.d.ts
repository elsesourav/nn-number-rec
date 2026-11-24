import type { NNEngineModule } from './types';

export interface CreateNNEngineOptions {
  locateFile?: (path: string, prefix: string) => string;
  wasmBinary?: ArrayBuffer | Uint8Array;
  onAbort?: (what: unknown) => void;
  print?: (text: string) => void;
  printErr?: (text: string) => void;
}

declare function createNNEngine(options?: CreateNNEngineOptions): Promise<NNEngineModule>;

export default createNNEngine;
