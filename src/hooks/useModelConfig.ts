import { useState, useEffect } from 'react';

export interface Model {
  id: string;
  name: string;
}

export interface Provider {
  id: string;
  name: string;
  models: Model[];
  supportsCustomModel?: boolean;
}

export interface ModelConfig {
  providers: Provider[];
  defaultProvider: string;
  // Present only when the deployment is locked to a single provider.
  // Both fields are absent (not null) when the lock is off.
  lockedProvider?: string;
  lockedModel?: string;
}

/**
 * Fetch the deployment's model configuration from `/api/models/config`.
 *
 * Shared by every component that needs the provider list and/or the
 * deployment lock, so the fetch, the loading/error handling and the
 * truthiness guard on the (absent-when-unlocked) lock fields live in one
 * place. Fetches once on mount.
 *
 * `lockedProvider`/`lockedModel` are the empty string when the deployment is
 * unlocked, so callers can guard on truthiness exactly as they would on the
 * raw fields.
 */
export function useModelConfig() {
  const [modelConfig, setModelConfig] = useState<ModelConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetchModelConfig = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch('/api/models/config');

        if (!response.ok) {
          throw new Error(`Error fetching model configurations: ${response.status}`);
        }

        const data: ModelConfig = await response.json();
        if (cancelled) return;
        setModelConfig(data);
      } catch (err) {
        if (cancelled) return;
        console.error('Failed to fetch model configurations:', err);
        setError('Failed to load model configurations. Using default options.');
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    fetchModelConfig();

    return () => {
      cancelled = true;
    };
  }, []);

  return {
    modelConfig,
    isLoading,
    error,
    lockedProvider: modelConfig?.lockedProvider || '',
    lockedModel: modelConfig?.lockedModel || '',
  };
}
