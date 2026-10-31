import { useCallback, useEffect, useRef, useState } from 'react';
import type { FailureRiskRequest } from '../types/component3.types';
import { predictExperimental, type ExperimentalPrediction } from './experimentalApi';

interface State { loading: boolean; result: ExperimentalPrediction | null; error: string | null }
const empty: State = { loading: false, result: null, error: null };

export function useExperimentalPrediction() {
  const [state, setState] = useState<State>(empty);
  const serial = useRef(0);
  const pending = useRef<AbortController | null>(null);
  const reset = useCallback(() => {
    serial.current++;
    pending.current?.abort();
    pending.current = null;
    setState(empty);
  }, []);
  useEffect(() => () => {
    serial.current++;
    pending.current?.abort();
    pending.current = null;
  }, []);

  const predict = useCallback(async (body: FailureRiskRequest, reference?: { modelVersion: string; probabilities?: number[] }) => {
    if (pending.current) return;
    const id = ++serial.current;
    const controller = new AbortController();
    pending.current = controller;
    setState({ loading: true, result: null, error: null });
    try {
      const result = await predictExperimental(body, controller.signal);
      if (reference && (result.model_version !== reference.modelVersion || (reference.probabilities
        && reference.probabilities.some((p, i) => p !== [result.p7, result.p14, result.p30][i])))) {
        throw new Error('Prediction does not match the v3 ranking. Refresh the ranking before retrying.');
      }
      if (serial.current === id) setState({ loading: false, result, error: null });
    } catch (error) {
      if (serial.current === id && !controller.signal.aborted) {
        setState({ loading: false, result: null, error: error instanceof Error ? error.message : 'Experimental prediction failed.' });
      }
    } finally {
      if (serial.current === id) pending.current = null;
    }
  }, []);
  return { ...state, predict, reset };
}
