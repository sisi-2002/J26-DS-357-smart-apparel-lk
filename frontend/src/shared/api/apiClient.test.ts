import { describe, expect, it, vi } from 'vitest';
import { jsonResponse } from '../../test/fixtures';
import { postJson } from './apiClient';

describe('shared JSON API client', () => {
  it('uses configured base, JSON payload and abort signal', async () => {
    vi.stubEnv('VITE_API_BASE_URL', '/api/v1/');
    const fetch = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    vi.stubGlobal('fetch', fetch);
    expect(await postJson('/component3/predict-risk', { machine_id: 1 })).toEqual({ ok: true });
    expect(fetch).toHaveBeenCalledWith('/api/v1/component3/predict-risk', expect.objectContaining({
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: '{"machine_id":1}', signal: expect.any(AbortSignal),
    }));
  });

  it.each([404, 422, 500, 502, 503, 504])('reports HTTP %i with a useful error', async status => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'Server detail' }, status)));
    await expect(postJson('predict', {})).rejects.toMatchObject({ name: 'ApiError', status });
  });

  it('formats Pydantic validation arrays as readable messages', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: [{ msg: 'Choose a valid timestamp' }] }, 422)));
    await expect(postJson('predict', {})).rejects.toThrow('Choose a valid timestamp');
  });

  it('handles non-JSON proxy errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('Proxy unavailable', { status: 502 })));
    await expect(postJson('predict', {})).rejects.toThrow('proxy could not reach FastAPI');
  });

  it('rejects a successful HTML response instead of treating it as a result', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>Vite fallback</html>')));
    await expect(postJson('predict', {})).rejects.toThrow('invalid JSON response');
  });

  it('reports missing configuration without making a request', async () => {
    vi.stubEnv('VITE_API_BASE_URL', '');
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    await expect(postJson('predict', {})).rejects.toThrow('API configuration is missing');
    expect(fetch).not.toHaveBeenCalled();
  });

  it('reports network failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    await expect(postJson('predict', {})).rejects.toThrow('Cannot reach the prediction API');
  });

  it('propagates caller cancellation', async () => {
    const controller = new AbortController();
    vi.stubGlobal('fetch', vi.fn((_url, options: RequestInit) => new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    })));
    const response = postJson('predict', {}, controller.signal);
    const assertion = expect(response).rejects.toMatchObject({ name: 'AbortError' });
    controller.abort();
    await assertion;
  });

  it('times out a stalled API request', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', vi.fn((_url, options: RequestInit) => new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    })));
    const assertion = expect(postJson('predict', {})).rejects.toThrow('timed out after 30 seconds');
    await vi.advanceTimersByTimeAsync(30_000);
    await assertion;
  });
});
