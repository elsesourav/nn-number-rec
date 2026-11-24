import { useState, useEffect } from 'react';
import type { NNEngineModule } from '../wasm/types';
import createNNEngine from '../wasm/nn_engine';

let cachedModulePromise: Promise<NNEngineModule> | null = null;

export function useWasmModule() {
  const [module, setModule] = useState<NNEngineModule | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function init() {
      try {
        if (!cachedModulePromise) {
          cachedModulePromise = createNNEngine({
            locateFile: (path: string) => {
              if (path.endsWith('.wasm')) {
                return `/wasm/${path}`;
              }
              return path;
            },
          });
        }

        const mod = await cachedModulePromise;
        if (isMounted) {
          setModule(mod);
          setLoading(false);
        }
      } catch (err: unknown) {
        console.error('Failed to load WebAssembly module:', err);
        if (isMounted) {
          const message = err instanceof Error ? err.message : 'Failed to initialize WebAssembly engine.';
          setError(message);
          setLoading(false);
        }
      }
    }

    init();

    return () => {
      isMounted = false;
    };
  }, []);

  return { module, loading, error };
}
