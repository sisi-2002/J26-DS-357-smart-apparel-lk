import { useCallback, useEffect, useRef, useState } from 'react';
import { predictFailureRisk } from '../api/failureRiskApi';
import type { FailureRiskRequest, FailureRiskResponse } from '../types/component3.types';

interface PredictionState {
  loading: boolean;
  result: FailureRiskResponse | null;
  error: string | null;
}
const emptyState: PredictionState = { loading: false, result: null, error: null };

export function useFailureRisk() {
  const [state, setState] = useState<PredictionState>(emptyState);
  const request = useRef<{ id: number; controller: AbortController } | null>(null);
  const serial = useRef(0);

  const reset = useCallback(() => {
    serial.current++;
    request.current?.controller.abort();
    request.current = null;
    setState(emptyState);
  }, []);

  useEffect(() => () => {
    serial.current++;
    request.current?.controller.abort();
    request.current = null;
  }, []);

  const predict = useCallback(async (body: FailureRiskRequest, ranking?: { modelVersion: string; probability: number }) => {
    if (request.current) return;
    const current = { id: ++serial.current, controller: new AbortController() };
    request.current = current;
    setState({ loading: true, result: null, error: null });
    try {
      const result = await predictFailureRisk(body, current.controller.signal);
      if (ranking && (result.model_version !== ranking.modelVersion
        || Math.abs(result.predicted_failure_probability - ranking.probability) > 0.000001)) {
        throw new Error('The prediction no longer matches this ranking. Refresh the ranking and try again.');
      }
      if (current.id === serial.current) setState({ loading: false, result, error: null });
    } catch (error) {
      if (current.id === serial.current && !current.controller.signal.aborted) {
        setState({ loading: false, result: null, error: error instanceof Error ? error.message : 'Prediction failed. Please retry.' });
      }
    } finally {
      if (request.current?.id === current.id) request.current = null;
    }
  }, []);

  return { ...state, predict, reset };
}
