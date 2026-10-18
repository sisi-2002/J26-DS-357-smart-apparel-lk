import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import App from '../../../app/App';
import { deferred, jsonResponse, riskFixture } from '../../../test/fixtures';
import { PREDICTION_TIMESTAMPS } from '../data/demoOptions';

function renderPage(path = '/component3/predict-risk') {
  return render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);
}
function mockResponse() {
  const fetch = vi.fn().mockResolvedValue(jsonResponse(riskFixture()));
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

describe('Component 3 historical prediction page', () => {
  it('renders at the component route without displaying a preset probability', () => {
    const fetch = mockResponse(); renderPage();
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Understand machine risk');
    expect(screen.getByLabelText('Synthetic data limitations')).toHaveTextContent('not real apparel-factory records');
    expect(screen.queryByTestId('failure-probability')).not.toBeInTheDocument();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('offers exactly 100 machines and the 13 supported naive timestamps', () => {
    renderPage();
    const machines = screen.getByLabelText('Historical machine');
    expect(within(machines).getAllByRole('option')).toHaveLength(100);
    expect(within(machines).getByRole('option', { name: 'Machine 100' })).toHaveValue('100');
    const times = within(screen.getByLabelText('Prediction snapshot')).getAllByRole('option');
    expect(times.map(option => (option as HTMLOptionElement).value)).toEqual([...PREDICTION_TIMESTAMPS]);
  });

  it('submits the selected integer ID and exact timezone-naive timestamp', async () => {
    const fetch = vi.fn().mockResolvedValue(jsonResponse(riskFixture({ machine_id: 100, prediction_time: PREDICTION_TIMESTAMPS[12] })));
    vi.stubGlobal('fetch', fetch); renderPage();
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText('Historical machine'), '100');
    await user.selectOptions(screen.getByLabelText('Prediction snapshot'), PREDICTION_TIMESTAMPS[12]);
    await user.click(screen.getByRole('button', { name: /Predict failure risk/ }));
    expect(fetch).toHaveBeenCalledWith('/api/v1/component3/predict-risk', expect.objectContaining({
      body: JSON.stringify({ machine_id: 100, prediction_time: '2015-11-25T06:00:00' }),
    }));
    expect(await screen.findByTestId('failure-probability')).toHaveTextContent('27.18%');
  });

  it('renders actual response probability, evidence, limitations and provenance', async () => {
    mockResponse(); renderPage();
    await userEvent.click(screen.getByRole('button', { name: /Predict failure risk/ }));
    expect(await screen.findByTestId('failure-probability')).toHaveTextContent('27.18%');
    expect(screen.getByTestId('evidence-H')).toHaveTextContent('42 days');
    expect(screen.getByTestId('evidence-M')).toHaveTextContent('3');
    expect(screen.getByTestId('evidence-F')).toHaveTextContent('2');
    expect(screen.getByTestId('evidence-C')).toHaveTextContent('100%');
    expect(screen.getByTestId('evidence-R')).toHaveTextContent('1.5 days');
    expect(screen.getByText(/A probability is not a reliability score/)).toBeInTheDocument();
    expect(screen.getByText(/This does not establish logging coverage/)).toBeInTheDocument();
    expect(screen.getByText(/not database freshness/)).toBeInTheDocument();
    expect(screen.getByText(/test fixture only/)).toBeInTheDocument();
    await userEvent.click(screen.getByText('Prediction provenance'));
    expect(screen.getByText('test-request')).toBeVisible();
  });

  it('displays null H/R history without substituting zero', async () => {
    const result = riskFixture();
    result.evidence_profile.H_observed_maintenance_span_days = null;
    result.evidence_profile.R_days_since_latest_maintenance = null;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(result))); renderPage();
    await userEvent.click(screen.getByRole('button', { name: /Predict failure risk/ }));
    expect(await screen.findByTestId('evidence-H')).toHaveTextContent('No prior maintenance observed');
    expect(screen.getByTestId('evidence-R')).toHaveTextContent('No prior maintenance observed');
  });

  it('shows loading status and prevents duplicate submissions', async () => {
    const pending = deferred<Response>();
    const fetch = vi.fn().mockReturnValue(pending.promise); vi.stubGlobal('fetch', fetch); renderPage();
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Predict failure risk/ }));
    expect(screen.getByRole('status')).toHaveTextContent('Reading the historical snapshot');
    expect(screen.getByRole('button', { name: 'Predicting…' })).toBeDisabled();
    fireEvent.submit(screen.getByRole('button', { name: 'Predicting…' }).closest('form')!);
    expect(fetch).toHaveBeenCalledTimes(1);
    await act(async () => { pending.resolve(jsonResponse(riskFixture())); });
    expect(await screen.findByTestId('failure-probability')).toBeVisible();
  });

  it.each([404, 422, 500, 503])('shows HTTP %i errors and allows a retry without a fallback result', async status => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'API test error' }, status))); renderPage();
    await userEvent.click(screen.getByRole('button', { name: /Predict failure risk/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Prediction unavailable');
    expect(screen.queryByTestId('failure-probability')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Predict failure risk/ })).toBeEnabled();
  });

  it('shows a network error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch'))); renderPage();
    await userEvent.click(screen.getByRole('button', { name: /Predict failure risk/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Cannot reach the prediction API');
  });

  it('clears a completed result when the selection changes', async () => {
    mockResponse(); renderPage();
    await userEvent.click(screen.getByRole('button', { name: /Predict failure risk/ }));
    await screen.findByTestId('failure-probability');
    await userEvent.selectOptions(screen.getByLabelText('Prediction snapshot'), PREDICTION_TIMESTAMPS[1]);
    expect(screen.queryByTestId('failure-probability')).not.toBeInTheDocument();
  });

  it('aborts an outdated request and ignores its late result even if transport ignores cancellation', async () => {
    const pending = deferred<Response>();
    const fetch = vi.fn().mockReturnValueOnce(pending.promise)
      .mockResolvedValueOnce(jsonResponse(riskFixture({ machine_id: 2, predicted_failure_probability: 0.13 })));
    vi.stubGlobal('fetch', fetch); renderPage();
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Predict failure risk/ }));
    const firstSignal = (fetch.mock.calls[0][1] as RequestInit).signal;
    await user.selectOptions(screen.getByLabelText('Historical machine'), '2');
    expect(firstSignal?.aborted).toBe(true);
    await user.click(screen.getByRole('button', { name: /Predict failure risk/ }));
    expect(await screen.findByTestId('failure-probability')).toHaveTextContent('13.00%');
    await act(async () => { pending.resolve(jsonResponse(riskFixture())); });
    expect(screen.getByTestId('failure-probability')).toHaveTextContent('13.00%');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('aborts pending work on unmount', async () => {
    const pending = deferred<Response>(); const fetch = vi.fn().mockReturnValue(pending.promise);
    vi.stubGlobal('fetch', fetch); const rendered = renderPage();
    await userEvent.click(screen.getByRole('button', { name: /Predict failure risk/ }));
    const signal = (fetch.mock.calls[0][1] as RequestInit).signal;
    rendered.unmount(); expect(signal?.aborted).toBe(true);
    await act(async () => { pending.resolve(jsonResponse(riskFixture())); });
  });

  it('redirects the root to the prediction page', () => {
    renderPage('/'); expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Understand machine risk');
  });

  it('has an accessible not-found route', async () => {
    renderPage('/missing');
    expect(screen.getByRole('heading', { name: 'Page not found' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('link', { name: 'Return to machine risk' }));
    await waitFor(() => expect(screen.getByLabelText('Historical machine')).toBeInTheDocument());
  });
});
