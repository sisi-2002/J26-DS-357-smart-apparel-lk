import { act, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { StrictMode } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import App from '../../../app/App';
import { deferred, jsonResponse, rankingFixture, riskFixture } from '../../../test/fixtures';
import type { TopRiskMachinesResponse } from '../types/component3.types';

function renderPage() { return render(<MemoryRouter initialEntries={['/component3/predict-risk']}><App /></MemoryRouter>); }
function routedFetch(ranking = rankingFixture()) {
  const fetch = vi.fn((url: string, options: RequestInit) => {
    const body = JSON.parse(options.body as string) as { machine_id: number; prediction_time: string };
    if (url.endsWith('top-risk-machines')) return Promise.resolve(jsonResponse({ ...ranking, prediction_time: body.prediction_time }));
    const row = ranking.machines.find(row => row.machine_id === body.machine_id);
    return Promise.resolve(jsonResponse(riskFixture({ ...body, predicted_failure_probability: row?.predicted_failure_probability ?? 0.25 })));
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}

describe('top-risk machine interaction', () => {
  it('automatically fetches the batch ranking without automatically predicting a machine', async () => {
    const fetch = routedFetch(); renderPage();
    expect(screen.getByRole('heading', { name: 'Highest Failure-Risk Machines (Next 7 Days)' })).toBeInTheDocument();
    expect(await screen.findByRole('table')).toBeVisible();
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe('/api/v1/component3/top-risk-machines');
    expect(screen.queryByTestId('failure-probability')).not.toBeInTheDocument();
  });

  it('ranks once, selects details and keeps all rows visible across machine switches', async () => {
    const fetch = routedFetch(); renderPage(); const user = userEvent.setup();
    const table = await screen.findByRole('table');
    expect(within(table).getAllByRole('row')).toHaveLength(11);
    expect(within(table).getAllByRole('row')[1]).toHaveTextContent('Machine 020Line 0280.00%');
    expect(fetch).toHaveBeenCalledTimes(1);
    await user.click(screen.getByRole('button', { name: 'View prediction for machine 20' }));
    expect(await screen.findByTestId('failure-probability')).toHaveTextContent('80.00%');
    expect(screen.getByLabelText('Historical machine')).toHaveValue('20');
    expect(screen.getByTestId('evidence-H')).toHaveTextContent('42 days');
    expect(screen.getByTestId('evidence-M')).toHaveTextContent('3');
    expect(screen.getByRole('table')).toBe(table);
    await user.click(screen.getByRole('button', { name: 'View prediction for machine 19' }));
    expect(await screen.findByTestId('failure-probability')).toHaveTextContent('75.00%');
    expect(screen.getByRole('table')).toBe(table);
    expect(fetch.mock.calls.filter(([url]) => url.endsWith('top-risk-machines'))).toHaveLength(1);
    expect(fetch).toHaveBeenCalledTimes(3);
    await user.selectOptions(screen.getByLabelText('Historical machine'), '99');
    expect(screen.getByRole('table')).toBeVisible();
    expect(screen.queryByTestId('failure-probability')).not.toBeInTheDocument();
  });

  it('automatically reranks on date change, clears details and preserves the selected machine', async () => {
    const fetch = routedFetch(); renderPage(); const user = userEvent.setup();
    await screen.findByRole('table');
    await user.click(screen.getByRole('button', { name: 'View prediction for machine 20' }));
    await screen.findByTestId('failure-probability');
    await user.selectOptions(screen.getByLabelText('Prediction snapshot'), '2015-09-09T06:00:00');
    expect(screen.queryByTestId('failure-probability')).not.toBeInTheDocument();
    await screen.findByRole('table', { name: /9 Sep 2015/ });
    expect(screen.getByLabelText('Historical machine')).toHaveValue('20');
    expect(fetch).toHaveBeenCalledTimes(3);
    expect(fetch.mock.calls.at(-1)?.[1].body).toBe('{"prediction_time":"2015-09-09T06:00:00"}');
  });

  it('shows pending state and blocks duplicate ranking requests', async () => {
    const pending = deferred<Response>(); const fetch = vi.fn().mockReturnValue(pending.promise);
    vi.stubGlobal('fetch', fetch); renderPage();
    const button = screen.getByRole('button', { name: 'Comparing 100 machines…' });
    expect(button).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent('one model batch');
    fireEvent.click(button);
    expect(fetch).toHaveBeenCalledTimes(1);
    await act(async () => { pending.resolve(jsonResponse(rankingFixture())); });
  });

  it('keeps the current ranking when the same date is selected again', async () => {
    const fetch = routedFetch(); renderPage();
    const table = await screen.findByRole('table');
    await userEvent.selectOptions(screen.getByLabelText('Prediction snapshot'), '2015-09-02T06:00:00');
    expect(screen.getByRole('table')).toBe(table);
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('aborts an old date and ignores its late ranking', async () => {
    const pending = deferred<Response>();
    const fetch = vi.fn().mockReturnValueOnce(pending.promise)
      .mockResolvedValueOnce(jsonResponse(rankingFixture({ prediction_time: '2015-09-09T06:00:00' })));
    vi.stubGlobal('fetch', fetch); renderPage(); const user = userEvent.setup();
    const signal = (fetch.mock.calls[0][1] as RequestInit).signal;
    await user.selectOptions(screen.getByLabelText('Prediction snapshot'), '2015-09-09T06:00:00');
    expect(signal?.aborted).toBe(true);
    await screen.findByRole('table', { name: /9 Sep 2015/ });
    await act(async () => { pending.resolve(jsonResponse(rankingFixture())); });
    expect(screen.getByRole('table', { name: /9 Sep 2015/ })).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('aborts pending rankings on unmount', async () => {
    const pending = deferred<Response>(); const fetch = vi.fn().mockReturnValue(pending.promise);
    vi.stubGlobal('fetch', fetch); const rendered = renderPage();
    const signal = (fetch.mock.calls[0][1] as RequestInit).signal;
    rendered.unmount(); expect(signal?.aborted).toBe(true);
    await act(async () => { pending.resolve(jsonResponse(rankingFixture())); });
  });

  it('does not overwrite a deliberate machine choice when automatic ranking finishes', async () => {
    const pending = deferred<Response>();
    const fetch = vi.fn().mockReturnValue(pending.promise);
    vi.stubGlobal('fetch', fetch); renderPage();
    await userEvent.selectOptions(screen.getByLabelText('Historical machine'), '99');
    await act(async () => { pending.resolve(jsonResponse(rankingFixture())); });
    expect(await screen.findByRole('table')).toBeVisible();
    expect(screen.getByLabelText('Historical machine')).toHaveValue('99');
    expect(screen.queryByTestId('failure-probability')).not.toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('refreshes the comparison without clearing an individual prediction or selection', async () => {
    const fetch = routedFetch(); renderPage(); const user = userEvent.setup();
    await screen.findByRole('table');
    await user.click(screen.getByRole('button', { name: 'View prediction for machine 20' }));
    await screen.findByTestId('failure-probability');
    await user.click(screen.getByRole('button', { name: 'Refresh ranking' }));
    await screen.findByRole('table');
    expect(screen.getByLabelText('Historical machine')).toHaveValue('20');
    expect(screen.getByTestId('failure-probability')).toHaveTextContent('80.00%');
    expect(fetch).toHaveBeenCalledTimes(3);
  });

  it('recovers from the development StrictMode effect replay and ignores the aborted response', async () => {
    const pending = deferred<Response>();
    const fetch = vi.fn().mockReturnValueOnce(pending.promise)
      .mockResolvedValueOnce(jsonResponse(rankingFixture()));
    vi.stubGlobal('fetch', fetch);
    render(<StrictMode><MemoryRouter initialEntries={['/component3/predict-risk']}><App /></MemoryRouter></StrictMode>);
    expect((fetch.mock.calls[0][1] as RequestInit).signal?.aborted).toBe(true);
    expect(await screen.findByRole('table')).toBeVisible();
    await act(async () => { pending.resolve(jsonResponse(rankingFixture({ request_id: 'stale-request' }))); });
    expect(screen.queryByText('stale-request')).not.toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it.each([500, 503])('handles HTTP %i and retries without preset results', async status => {
    const fetch = vi.fn().mockResolvedValueOnce(jsonResponse({ detail: 'Test failure' }, status))
      .mockResolvedValueOnce(jsonResponse(rankingFixture()));
    vi.stubGlobal('fetch', fetch); renderPage(); const user = userEvent.setup();
    expect(await screen.findByRole('alert')).toHaveTextContent('Ranking unavailable');
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Refresh ranking' }));
    expect(await screen.findByRole('table')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('rejects details from a changed model rather than associating them with the ranking', async () => {
    const fetch = routedFetch(); renderPage(); const user = userEvent.setup();
    await screen.findByRole('table');
    fetch.mockImplementationOnce(() => Promise.resolve(jsonResponse(riskFixture({ machine_id: 20,
      predicted_failure_probability: 0.8, model_version: 'sha256:changed-model' }))));
    await user.click(screen.getByRole('button', { name: 'View prediction for machine 20' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Refresh the ranking');
    expect(screen.queryByTestId('failure-probability')).not.toBeInTheDocument();
    expect(screen.getByRole('table')).toBeVisible();
  });

  it('shows null H/R values when a ranked machine has no maintenance history', async () => {
    const fetch = routedFetch(); renderPage(); const user = userEvent.setup();
    await screen.findByRole('table');
    const detail = riskFixture({ machine_id: 20, predicted_failure_probability: 0.8 });
    detail.evidence_profile.H_observed_maintenance_span_days = null;
    detail.evidence_profile.R_days_since_latest_maintenance = null;
    fetch.mockImplementationOnce(() => Promise.resolve(jsonResponse(detail)));
    await user.click(screen.getByRole('button', { name: 'View prediction for machine 20' }));
    expect(await screen.findByTestId('evidence-H')).toHaveTextContent('No prior maintenance observed');
    expect(screen.getByTestId('evidence-R')).toHaveTextContent('No prior maintenance observed');
  });

  it('displays server order without sorting equal rounded probabilities in the browser', async () => {
    const result: TopRiskMachinesResponse = rankingFixture();
    result.machines[1].predicted_failure_probability = result.machines[0].predicted_failure_probability;
    routedFetch(result); renderPage();
    const rows = within(await screen.findByRole('table')).getAllByRole('row');
    expect(rows[1]).toHaveTextContent('Machine 020Line 0280.00%');
    expect(rows[2]).toHaveTextContent('Machine 019Line 0280.00%');
  });
});
