import { useCallback, useEffect, useRef, useState } from 'react';
import { getTopRiskMachines } from '../api/topRiskMachinesApi';
import type { TopRiskMachinesResponse } from '../types/component3.types';

interface RankingState {
  loading: boolean;
  result: TopRiskMachinesResponse | null;
  error: string | null;
}
const empty: RankingState = { loading: false, result: null, error: null };

export function useTopRiskMachines() {
  const [state, setState] = useState<RankingState>(empty);
  const current = useRef<AbortController | null>(null);
  const serial = useRef(0);
  const reset = useCallback(() => {
    serial.current++;
    current.current?.abort();
    current.current = null;
    setState(empty);
  }, []);

  useEffect(() => () => { serial.current++; current.current?.abort(); current.current = null; }, []);

  const rank = useCallback(async (predictionTime: string) => {
    if (current.current) return;
    const controller = new AbortController();
    const id = ++serial.current;
    current.current = controller;
    setState({ loading: true, result: null, error: null });
    try {
      const result = await getTopRiskMachines(predictionTime, controller.signal);
      if (id === serial.current) setState({ loading: false, result, error: null });
    } catch (error) {
      if (id === serial.current && !controller.signal.aborted) {
        setState({ loading: false, result: null, error: error instanceof Error ? error.message : 'Ranking failed. Please retry.' });
      }
    } finally {
      if (id === serial.current) current.current = null;
    }
  }, []);

  return { ...state, rank, reset };
}
