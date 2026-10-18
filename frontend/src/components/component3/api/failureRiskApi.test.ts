import { describe, expect, it, vi } from 'vitest';
import { jsonResponse, riskFixture } from '../../../test/fixtures';
import { predictFailureRisk } from './failureRiskApi';

const body = { machine_id: 1, prediction_time: '2015-09-02T06:00:00' };

describe('failure-risk response validation', () => {
  it('accepts the backend contract unchanged', async () => {
    const fixture = riskFixture();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(fixture)));
    expect(await predictFailureRisk(body)).toEqual(fixture);
  });

  it.each([
    null, {}, { horizon_days: 14 }, { predicted_failure_probability: -0.1 },
    { predicted_failure_probability: 1.1 }, { predicted_failure_probability: null },
    { research_only: false }, { data_status: 'factory_live' }, { evidence_profile: {} },
    { note: '' }, { prediction_time: '2026-10-18T06:00:00' },
  ])('rejects incompatible data: %j', async change => {
    const payload = change === null ? null : Object.keys(change).length ? { ...riskFixture(), ...change } : {};
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(payload)));
    await expect(predictFailureRisk(body)).rejects.toThrow('incompatible prediction response');
  });

  it('rejects a response for a different selection', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(riskFixture({ machine_id: 2 }))));
    await expect(predictFailureRisk(body)).rejects.toThrow('different machine or timestamp');
  });
});
