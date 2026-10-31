import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../../../app/App';
import { deferred, jsonResponse, rankingFixture } from '../../../test/fixtures';
import { PREDICTION_TIMESTAMPS } from '../data/demoOptions';
import { predictExperimental, rankExperimental, type ExperimentalPrediction } from './experimentalApi';
import { experimentalPreviewEnabled } from './featureFlag';

const body = { machine_id: 1, prediction_time: PREDICTION_TIMESTAMPS[0] };
// HTTP fixtures only: these probabilities are never bundled into the application.
function fixture(overrides: Partial<ExperimentalPrediction> = {}): ExperimentalPrediction {
  return { ...body, p7: .12345678, p14: .23456789, p30: .45678912,
    model_version: `sha256:${'a'.repeat(64)}`, model_release: 'v3-run_003-unpooled-hgb-sigmoid',
    model_schema: 'unpooled-inclusive-v3', experimental: true, research_only: true,
    data_status: 'synthetic_azure_historical_demo', interval_semantics: '(0,7], (7,14], (14,30]',
    note: 'Synthetic Azure benchmark, not factory validated. Test fixture only.', request_id: 'test-preview-request',
    evidence_profile: { H_observed_maintenance_span_days: 42, M_prior_maintenance_events: 3, F_prior_failure_events: 2,
      R_days_since_latest_maintenance: 1.5, C_published_field_completeness_pct: 100, error_events_prior: 7,
      error_events_7d: 1, error_events_30d: 2, error_events_90d: 3, days_since_last_error: 4 }, ...overrides };
}
function renderPage(path = '/component3/predict-risk') {
  return render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);
}
function mockResponse(response: unknown = fixture(), status = 200) {
  const fetch = vi.fn().mockResolvedValue(jsonResponse(response, status));
  routeFetch(fetch); return fetch;
}

function routeFetch(fetch: (url: string, options: RequestInit) => Promise<Response>) {
  vi.stubGlobal('fetch', (url: string, options: RequestInit) => {
    if (url.endsWith('/experimental/top-risk-machines')) {
      const request = JSON.parse(options.body as string);
      return Promise.resolve(jsonResponse({ ...fixture(), ...request, scored_machine_count: 100, returned_machine_count: 10,
        machines: Array.from({ length: 10 }, (_, i) => ({ rank: i + 1, machine_id: i + 11, p7: .3 - i / 100, p14: .4 - i / 100, p30: .5 - i / 100 })) }));
    }
    return fetch(url, options);
  });
}

describe('isolated experimental multi-horizon preview', () => {
  beforeEach(() => { vi.stubEnv('VITE_C3_EXPERIMENTAL_MULTIHORIZON', 'true'); mockResponse(); });

  it.each([undefined, '', 'false', '1', 'TRUE'])('is off unless explicitly true (%s)', value => {
    vi.stubEnv('VITE_C3_EXPERIMENTAL_MULTIHORIZON', value);
    expect(experimentalPreviewEnabled()).toBe(false);
    const fetch = vi.fn().mockResolvedValue(jsonResponse(rankingFixture())); routeFetch(fetch);
    renderPage();
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Machine Failure-Risk Explorer');
    expect(screen.queryByRole('link', { name: 'Experimental horizons' })).not.toBeInTheDocument();
    expect(screen.queryByTestId('experimental-p7')).not.toBeInTheDocument();
    expect(fetch.mock.calls.every(([url]) => !(url as string).includes('/experimental/'))).toBe(true);
  });

  it('shows the opt-in navigation, provenance and a compact empty state without requesting predictions', () => {
    const fetch = mockResponse(); renderPage();
    expect(screen.getByRole('link', { name: /Machine risk/ })).toHaveAttribute('aria-current', 'page');
    expect(screen.queryByRole('link', { name: 'Experimental horizons' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Prototype information')).toHaveTextContent('Synthetic Demo · Azure 2015 Data');
    expect(screen.getByLabelText('Prototype information')).toHaveTextContent('not validated maintenance alerts');
    expect(screen.getByLabelText('Prototype information')).toBeVisible();
    expect(screen.getByText(/Comparison does not automatically request a prediction/)).toBeVisible();
    expect(screen.getByText('Model & Data Info').closest('details')).not.toHaveAttribute('open');
    expect(fetch).not.toHaveBeenCalled();
  });

  it('uses one experimental API response for every horizon and matching machine name/line', async () => {
    const fetch = mockResponse(fixture({ machine_id: 64, prediction_time: PREDICTION_TIMESTAMPS[12] }));
    renderPage(); const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText('Historical machine'), '64');
    await user.selectOptions(screen.getByLabelText('Prediction snapshot'), PREDICTION_TIMESTAMPS[12]);
    await user.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    expect(await screen.findByTestId('experimental-p7')).toHaveTextContent('12.35%');
    expect(screen.getByTestId('experimental-p14')).toHaveTextContent('23.46%');
    expect(screen.getByTestId('experimental-p30')).toHaveTextContent('45.68%');
    expect(screen.getByRole('heading', { name: 'Machine 064 · Line 07' })).toBeVisible();
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith('/api/v1/component3/experimental/predict-risk', expect.objectContaining({
      body: JSON.stringify({ machine_id: 64, prediction_time: PREDICTION_TIMESTAMPS[12] }),
    }));
    await user.click(screen.getByText('Model & Data Info'));
    expect(screen.getAllByText('test-preview-request')).toHaveLength(2);
    expect(screen.getByText(/not real apparel-factory records/)).toBeVisible();
    expect(screen.getByText(/Thirty-day Top-10 ranking was weaker/)).toBeVisible();
    expect(screen.getByText(/three frozen-model sigmoid calibrators/)).toBeVisible();
  });

  it('disables duplicate submissions while loading and clears results on date changes', async () => {
    const pending = deferred<Response>(); const fetch = vi.fn().mockReturnValue(pending.promise); routeFetch(fetch);
    renderPage(); await userEvent.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    expect(screen.getByRole('status')).toHaveTextContent('saved v3 model');
    const button = screen.getByRole('button', { name: /Predicting/ });
    expect(button).toBeDisabled(); fireEvent.submit(button.closest('form')!);
    expect(fetch).toHaveBeenCalledTimes(1);
    await act(async () => { pending.resolve(jsonResponse(fixture())); });
    expect(await screen.findByTestId('experimental-p30')).toBeVisible();
    await userEvent.selectOptions(screen.getByLabelText('Prediction snapshot'), PREDICTION_TIMESTAMPS[1]);
    expect(screen.queryByTestId('experimental-p7')).not.toBeInTheDocument();
  });

  it('cancels changed selections and ignores late responses, including after unmount', async () => {
    const pending = deferred<Response>();
    const fetch = vi.fn().mockReturnValueOnce(pending.promise).mockResolvedValueOnce(jsonResponse(fixture({ machine_id: 2, p7: .01 })));
    routeFetch(fetch); const rendered = renderPage(); const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    await user.selectOptions(screen.getByLabelText('Historical machine'), '2');
    expect((fetch.mock.calls[0][1] as RequestInit).signal?.aborted).toBe(true);
    await user.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    expect(await screen.findByTestId('experimental-p7')).toHaveTextContent('1.00%');
    await act(async () => { pending.resolve(jsonResponse(fixture())); });
    expect(screen.getByTestId('experimental-p7')).toHaveTextContent('1.00%');
    const third = deferred<Response>(); fetch.mockReturnValueOnce(third.promise);
    await user.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    rendered.unmount(); expect((fetch.mock.calls[2][1] as RequestInit).signal?.aborted).toBe(true);
    await act(async () => { third.resolve(jsonResponse(fixture({ machine_id: 2 }))); });
  });

  it.each([404, 422, 500, 503])('fails safely on HTTP %i without falling back to LR', async status => {
    const fetch = mockResponse({ detail: 'Unavailable' }, status); renderPage();
    await userEvent.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Prediction unavailable');
    expect(screen.getByRole('alert')).toHaveTextContent('No substitute probabilities');
    expect(screen.queryByTestId('experimental-p7')).not.toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Predict selected machine' })).toBeEnabled();
  });

  it('reports network errors without displaying probabilities', async () => {
    routeFetch(vi.fn().mockRejectedValue(new TypeError('Network failed'))); renderPage();
    await userEvent.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Cannot reach');
    expect(screen.queryByTestId('experimental-p30')).not.toBeInTheDocument();
  });

  it.each([
    { p7: .6, p14: .2 }, { p30: 1.1 }, { p7: null }, { p7: '0.2' },
    { machine_id: 2 }, { prediction_time: PREDICTION_TIMESTAMPS[1] }, { experimental: false },
    { model_schema: 'wrong' }, { model_version: 'not-a-version' }, { model_release: 'other-model' },
  ])('rejects incompatible output without rounding or repairing it: %j', async change => {
    mockResponse({ ...fixture(), ...change });
    await expect(predictExperimental(body)).rejects.toThrow('incompatible or incoherent');
  });

  it('preserves the exact full-precision response values in the API layer', async () => {
    mockResponse(); expect(await predictExperimental(body)).toEqual(fixture());
  });
});

function rankFixture(timestamp: string = body.prediction_time, horizon = 7) {
  return { ...fixture(), prediction_time: timestamp, horizon_days: horizon, scored_machine_count: 100, returned_machine_count: 10,
    machines: Array.from({ length: 10 }, (_, i) => ({ rank: i + 1, machine_id: i + 11, p7: .3 - i / 100, p14: .4 - i / 100, p30: .5 - i / 100 })) };
}

describe('unified v3 comparison and evidence workflow', () => {
  beforeEach(() => { vi.stubEnv('VITE_C3_EXPERIMENTAL_MULTIHORIZON', 'true'); });

  it('automatically loads only the v3 ranking, preserving deliberate machine selection', async () => {
    const pending = deferred<Response>(); const fetch = vi.fn().mockReturnValue(pending.promise); vi.stubGlobal('fetch', fetch);
    renderPage();
    expect(fetch).toHaveBeenCalledWith('/api/v1/component3/experimental/top-risk-machines', expect.objectContaining({ body: JSON.stringify({ prediction_time: body.prediction_time, horizon_days: 7 }) }));
    await userEvent.selectOptions(screen.getByLabelText('Historical machine'), '64');
    await act(async () => { pending.resolve(jsonResponse(rankFixture())); });
    expect(screen.getByLabelText('Historical machine')).toHaveValue('64');
    expect(screen.queryByTestId('experimental-p7')).not.toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('selects ranked machines, checks all three probabilities and keeps the ranking visible', async () => {
    const fetch = mockResponse(fixture({ machine_id: 11, p7: .3, p14: .4, p30: .5 })); renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'View prediction for machine 11' }));
    expect(await screen.findByTestId('experimental-p7')).toHaveTextContent('30.00%');
    expect(screen.getByTestId('experimental-p14')).toHaveTextContent('40.00%');
    expect(screen.getByTestId('experimental-p30')).toHaveTextContent('50.00%');
    expect(screen.getByRole('heading', { name: 'Machine 011 · Line 02' })).toBeVisible();
    expect(screen.getAllByRole('button', { name: /View prediction for machine/ })).toHaveLength(10);
    expect(screen.getByTestId('evidence-H')).toHaveTextContent('42 days');
    expect(screen.getByTestId('evidence-errors')).toHaveTextContent('7');
    expect(screen.getByTestId('error-windows')).toHaveTextContent('1 / 2 / 3');
    expect(screen.getByTestId('evidence-error-recency')).toHaveTextContent('4 days');
    expect(screen.getByText(/Evidence sufficiency for v3 has not yet been evaluated/)).toBeVisible();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('rejects a prediction that disagrees with the ranking rather than mixing models', async () => {
    mockResponse(fixture({ machine_id: 11, p7: .1 })); renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'View prediction for machine 11' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('does not match the v3 ranking');
    expect(screen.queryByTestId('experimental-p7')).not.toBeInTheDocument();
  });

  it('switches comparison horizon without clearing the selected prediction', async () => {
    mockResponse(); renderPage();
    await userEvent.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    await screen.findByTestId('experimental-p30');
    await userEvent.selectOptions(screen.getByLabelText('Rank machines by'), '30');
    expect(await screen.findByRole('columnheader', { name: /30-day probability/ })).toBeVisible();
    expect(screen.getByTestId('experimental-p7')).toHaveTextContent('12.35%');
    expect(screen.getByLabelText('Historical machine')).toHaveValue('1');
  });

  it('ignores a late ranking after a date change', async () => {
    const first = deferred<Response>();
    const fetch = vi.fn().mockReturnValueOnce(first.promise).mockResolvedValueOnce(jsonResponse(rankFixture(PREDICTION_TIMESTAMPS[1])));
    vi.stubGlobal('fetch', fetch); renderPage();
    await userEvent.selectOptions(screen.getByLabelText('Prediction snapshot'), PREDICTION_TIMESTAMPS[1]);
    expect((fetch.mock.calls[0][1] as RequestInit).signal?.aborted).toBe(true);
    await screen.findByRole('button', { name: 'View prediction for machine 11' });
    await act(async () => { first.resolve(jsonResponse(rankFixture())); });
    expect(screen.getByRole('table')).toHaveAccessibleName(/9 Sep 2015/);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('allows individual prediction when ranking fails', async () => {
    vi.stubGlobal('fetch', (url: string) => Promise.resolve(url.endsWith('/top-risk-machines')
      ? jsonResponse({ detail: 'Not ready' }, 503) : jsonResponse(fixture())));
    renderPage(); expect(await screen.findByRole('alert')).toHaveTextContent('Ranking unavailable');
    await userEvent.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    expect(await screen.findByTestId('experimental-p7')).toHaveTextContent('12.35%');
  });

  it('renders missing maintenance/error recencies as missing, not zero', async () => {
    const result = fixture(); result.evidence_profile.H_observed_maintenance_span_days = null;
    result.evidence_profile.R_days_since_latest_maintenance = null; result.evidence_profile.days_since_last_error = null;
    mockResponse(result); renderPage(); await userEvent.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    expect(await screen.findByTestId('evidence-H')).toHaveTextContent('No prior maintenance observed');
    expect(screen.getByTestId('evidence-error-recency')).toHaveTextContent('No prior error observed');
  });

  it('redirects the former preview bookmark to the primary product page', async () => {
    mockResponse(); renderPage('/component3/experimental-risk');
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Machine Failure-Risk Explorer');
    expect(screen.getByLabelText('Rank machines by')).toBeVisible();
  });

  it.each(['duplicate','order','horizon','version','count','nan'])('rejects malformed ranking: %s', async defect => {
    const r = rankFixture();
    if (defect === 'duplicate') r.machines[1].machine_id = r.machines[0].machine_id;
    if (defect === 'order') r.machines.reverse();
    if (defect === 'horizon') r.horizon_days = 30;
    if (defect === 'version') r.model_schema = 'wrong' as typeof r.model_schema;
    if (defect === 'count') r.scored_machine_count = 99;
    if (defect === 'nan') r.machines[0].p7 = NaN;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(r)));
    await expect(rankExperimental(body.prediction_time, 7)).rejects.toThrow();
  });
});
