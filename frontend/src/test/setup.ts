import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeEach, vi } from 'vitest';

beforeEach(() => { vi.stubEnv('VITE_API_BASE_URL', '/api/v1'); vi.stubEnv('VITE_C3_EXPERIMENTAL_MULTIHORIZON', 'false'); });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.unstubAllEnvs(); vi.useRealTimers(); });
