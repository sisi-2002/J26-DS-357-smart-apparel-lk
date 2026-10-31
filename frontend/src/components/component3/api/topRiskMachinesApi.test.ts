import { describe, expect, it, vi } from 'vitest';
import { jsonResponse, rankingFixture } from '../../../test/fixtures';
import { getTopRiskMachines } from './topRiskMachinesApi';

const time = '2015-09-02T06:00:00';

describe('top-risk API contract', () => {
  it('sends one timestamp-only request and preserves returned ordering', async () => {
    const result = rankingFixture();
    const fetch = vi.fn().mockResolvedValue(jsonResponse(result)); vi.stubGlobal('fetch', fetch);
    expect(await getTopRiskMachines(time)).toEqual(result);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith('/api/v1/component3/top-risk-machines', expect.objectContaining({
      body: JSON.stringify({ prediction_time: time }), method: 'POST',
    }));
  });

  it.each([
    { horizon_days: 14 }, { scored_machine_count: 99 }, { returned_machine_count: 9 },
    { research_only: false }, { data_status: 'factory_live' }, { machines: [] }, { model_version: '' },
  ])('rejects invalid metadata: %j', async change => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ ...rankingFixture(), ...change })));
    await expect(getTopRiskMachines(time)).rejects.toThrow('incompatible ranking');
  });

  it.each(['duplicate', 'wrong_order', 'wrong_rank', 'invalid_probability', 'unknown_id'])('rejects invalid rows: %s', async problem => {
    const result = rankingFixture();
    if (problem === 'duplicate') result.machines[1].machine_id = result.machines[0].machine_id;
    if (problem === 'wrong_order') result.machines[1].predicted_failure_probability = 0.99;
    if (problem === 'wrong_rank') result.machines[0].rank = 2;
    if (problem === 'invalid_probability') result.machines[0].predicted_failure_probability = 1.1;
    if (problem === 'unknown_id') result.machines[0].machine_id = 101;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(result)));
    await expect(getTopRiskMachines(time)).rejects.toThrow('incompatible ranking');
  });

  it('accepts equal serialized probabilities without re-sorting by ID', async () => {
    const result = rankingFixture(); result.machines[1].predicted_failure_probability = result.machines[0].predicted_failure_probability;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(result)));
    expect((await getTopRiskMachines(time)).machines[0].machine_id).toBe(20);
  });

  it('rejects another prediction date', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(rankingFixture({ prediction_time: '2015-09-09T06:00:00' }))));
    await expect(getTopRiskMachines(time)).rejects.toThrow('different snapshot');
  });

  it('rejects unsupported input before requesting the API', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    await expect(getTopRiskMachines('2026-10-18T06:00:00')).rejects.toThrow('supported historical snapshot');
    expect(fetch).not.toHaveBeenCalled();
  });
});
