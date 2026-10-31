import { useCallback, useEffect, useRef, useState } from 'react';
import { rankExperimental, type ExperimentalRanking, type Horizon } from './experimentalApi';

const empty = { loading: false, result: null as ExperimentalRanking | null, error: null as string | null };
export function useExperimentalRanking() {
  const [state, setState] = useState(empty);
  const serial = useRef(0);
  const pending = useRef<AbortController | null>(null);
  const reset = useCallback(() => {
    serial.current++; pending.current?.abort(); pending.current = null; setState(empty);
  }, []);
  useEffect(() => () => { serial.current++; pending.current?.abort(); pending.current = null; }, []);
  const rank = useCallback(async (timestamp: string, horizon: Horizon) => {
    if (pending.current) return;
    const controller = new AbortController(); const id = ++serial.current; pending.current = controller;
    setState({ loading: true, result: null, error: null });
    try {
      const result = await rankExperimental(timestamp, horizon, controller.signal);
      if (id === serial.current) setState({ loading: false, result, error: null });
    } catch (error) {
      if (id === serial.current && !controller.signal.aborted) {
        setState({ loading: false, result: null, error: error instanceof Error ? error.message : 'V3 comparison failed.' });
      }
    } finally { if (id === serial.current) pending.current = null; }
  }, []);
  return { ...state, rank, reset };
}
