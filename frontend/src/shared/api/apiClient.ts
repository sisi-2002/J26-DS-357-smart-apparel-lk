export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = 'ApiError';
  }
}

const statusMessages: Record<number, string> = {
  404: 'This machine was not found in the configured synthetic benchmark.',
  422: 'The API rejected this selection. Choose a supported machine and historical timestamp.',
  500: 'The model could not complete this prediction. Try again or check the backend logs.',
  502: 'The API proxy could not reach FastAPI. Check that the backend is running.',
  503: 'The saved model or benchmark data is unavailable. Check the backend configuration.',
  504: 'The API did not respond in time. Check that FastAPI is running and try again.',
};

function errorDetail(value: unknown): string | undefined {
  if (!value || typeof value !== 'object' || !('detail' in value)) return;
  const detail: unknown = value.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.flatMap((item: unknown) => {
      if (item && typeof item === 'object' && 'msg' in item && typeof item.msg === 'string') {
        return [item.msg];
      }
      return [];
    }).join('; ') || undefined;
  }
}

export async function postJson(path: string, body: unknown, signal?: AbortSignal): Promise<unknown> {
  const base = import.meta.env.VITE_API_BASE_URL?.trim().replace(/\/+$/, '');
  if (!base) throw new ApiError('API configuration is missing. Set VITE_API_BASE_URL and restart Vite.');
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) controller.abort();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, 30_000);

  try {
    const response = await fetch(`${base}/${path.replace(/^\/+/, '')}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body), signal: controller.signal,
    });
    const payload: unknown = await response.json().catch(() => undefined);
    if (!response.ok) {
      const message = statusMessages[response.status] ?? errorDetail(payload) ?? `The API returned HTTP ${response.status}.`;
      const detail = response.status === 422 ? errorDetail(payload) : undefined;
      throw new ApiError(detail ? `${message} ${detail}` : message, response.status);
    }
    if (payload === undefined) throw new ApiError('The API returned an invalid JSON response. Check the API proxy configuration.');
    return payload;
  } catch (error) {
    if (timedOut) throw new ApiError('The prediction timed out after 30 seconds. Check the backend and try again.');
    if (signal?.aborted) throw error;
    if (error instanceof ApiError) throw error;
    throw new ApiError('Cannot reach the prediction API. Check FastAPI and the Vite proxy configuration.');
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}
