import { act, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import App from '../../../app/App';
import { deferred, jsonResponse, rankingFixture, riskFixture } from '../../../test/fixtures';
import type { TopRiskMachinesResponse } from '../types/component3.types';

function renderPage() { return render(<MemoryRouter initialEntries={['/component3/predict-risk']}><App /></MemoryRouter>); }
function routedFetch(ranking = rankingFixture()) {
  const fetch = vi.fn((url: string, options: RequestInit) => {
    if (url.endsWith('top-risk-machines')) return Promise.resolve(jsonResponse(ranking));
    const body = JSON.parse(options.body as string) as { machine_id: number; prediction_time: string };
    const row = ranking.machines.find(row => row.machine_id === body.machine_id);
    return Promise.resolve(jsonResponse(riskFixture({ ...body, predicted_failure_probability: row?.predicted_failure_probability ?? 0.25 })));
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}

describe('top-risk machine interaction', () => {
  it('does not fetch or show a preset ranking on initial render', () => {
    const fetch = routedFetch(); renderPage();
    expect(screen.getByRole('heading', { name: 'Top 10 Machines' })).toBeInTheDocument();
    expect(screen.getByText(/Relative to the 100 machines in the synthetic Azure benchmark/)).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('ranks once, selects details and keeps all rows visible across machine switches', async () => {
    const fetch = routedFetch(); renderPage(); const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
    const table = await screen.findByRole('table');
    expect(within(table).getAllByRole('row')).toHaveLength(11);
    expect(within(table).getAllByRole('row')[1]).toHaveTextContent('02080.00%');
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

  it('uses the selected date for ranking and clears rankings and details on date change', async () => {
    const fetch = routedFetch(); renderPage(); const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
    await screen.findByRole('table');
    await user.click(screen.getByRole('button', { name: 'View prediction for machine 20' }));
    await screen.findByTestId('failure-probability');
    await user.selectOptions(screen.getByLabelText('Prediction snapshot'), '2015-09-09T06:00:00');
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    expect(screen.queryByTestId('failure-probability')).not.toBeInTheDocument();
    fetch.mockImplementationOnce(() => Promise.resolve(jsonResponse(rankingFixture({ prediction_time: '2015-09-09T06:00:00' }))));
    await user.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
    await screen.findByRole('table');
    expect(fetch.mock.calls.at(-1)?.[1].body).toBe('{"prediction_time":"2015-09-09T06:00:00"}');
  });

  it('shows pending state and blocks duplicate ranking requests', async () => {
    const pending = deferred<Response>(); const fetch = vi.fn().mockReturnValue(pending.promise);
    vi.stubGlobal('fetch', fetch); renderPage();
    await userEvent.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
    const button = screen.getByRole('button', { name: 'Comparing 100 machines…' });
    expect(button).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent('one model batch');
    fireEvent.click(button);
    expect(fetch).toHaveBeenCalledTimes(1);
    await act(async () => { pending.resolve(jsonResponse(rankingFixture())); });
  });

  it('aborts an old date and ignores its late ranking', async () => {
    const pending = deferred<Response>();
    const fetch = vi.fn().mockReturnValueOnce(pending.promise)
      .mockResolvedValueOnce(jsonResponse(rankingFixture({ prediction_time: '2015-09-09T06:00:00' })));
    vi.stubGlobal('fetch', fetch); renderPage(); const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
    const signal = (fetch.mock.calls[0][1] as RequestInit).signal;
    await user.selectOptions(screen.getByLabelText('Prediction snapshot'), '2015-09-09T06:00:00');
    expect(signal?.aborted).toBe(true);
    await user.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
    await screen.findByRole('table', { name: /9 Sep 2015/ });
    await act(async () => { pending.resolve(jsonResponse(rankingFixture())); });
    expect(screen.getByRole('table', { name: /9 Sep 2015/ })).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('aborts pending rankings on unmount', async () => {
    const pending = deferred<Response>(); const fetch = vi.fn().mockReturnValue(pending.promise);
    vi.stubGlobal('fetch', fetch); const rendered = renderPage();
    await userEvent.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
    const signal = (fetch.mock.calls[0][1] as RequestInit).signal;
    rendered.unmount(); expect(signal?.aborted).toBe(true);
    await act(async () => { pending.resolve(jsonResponse(rankingFixture())); });
  });

  it.each([500, 503])('handles HTTP %i and retries without preset results', async status => {
    const fetch = vi.fn().mockResolvedValueOnce(jsonResponse({ detail: 'Test failure' }, status))
      .mockResolvedValueOnce(jsonResponse(rankingFixture()));
    vi.stubGlobal('fetch', fetch); renderPage(); const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Ranking unavailable');
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
    expect(await screen.findByRole('table')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('rejects details from a changed model rather than associating them with the ranking', async () => {
    const fetch = routedFetch(); renderPage(); const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
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
    await user.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
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
    await userEvent.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
    const rows = within(await screen.findByRole('table')).getAllByRole('row');
    expect(rows[1]).toHaveTextContent('02080.00%');
    expect(rows[2]).toHaveTextContent('01980.00%');
  });
});
