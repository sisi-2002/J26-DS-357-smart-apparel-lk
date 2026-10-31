const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').trim().replace(/\/+$/, '');

function apiUrl(path: string): string {
  // Component 1 paths already contain /api/v1, and artifact paths are
  // relative to the server origin. Support the shared API-root setting
  // while retaining Component 1's legacy origin-only configuration.
  const prefix = '/api/v1';
  const origin = BASE_URL.endsWith(prefix) ? BASE_URL.slice(0, -prefix.length) : BASE_URL;
  return `${origin}${path}`;
}

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const url = apiUrl(path);
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    throw new ApiError(0, 'Network error — is the backend running?');
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* no JSON body */
    }
    throw new ApiError(res.status, detail);
  }

  return res.json() as Promise<T>;
}

export function artifactUrl(relativePath: string): string {
  return apiUrl(relativePath);
}
