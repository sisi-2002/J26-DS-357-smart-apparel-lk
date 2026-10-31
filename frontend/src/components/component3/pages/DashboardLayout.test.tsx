import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../../../app/App';
import { deferred, jsonResponse, rankingFixture, riskFixture } from '../../../test/fixtures';

function renderDashboard() {
  return render(<MemoryRouter initialEntries={['/component3/predict-risk']}><App /></MemoryRouter>);
}

describe('machine-risk dashboard presentation and workflow', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn((url: string) => Promise.resolve(jsonResponse(
      url.endsWith('top-risk-machines') ? rankingFixture() : riskFixture()))));
  });
  it('has one shared control section and two independently labelled result panels', async () => {
    renderDashboard();
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Machine Failure-Risk Explorer');
    const controls = screen.getByRole('region', { name: 'Choose a historical snapshot' });
    expect(within(controls).getByLabelText('Prediction snapshot')).toBeInTheDocument();
    expect(screen.getAllByLabelText('Prediction snapshot')).toHaveLength(1);
    expect(within(controls).getByLabelText('Historical machine')).toBeInTheDocument();
    expect(within(controls).getByRole('button', { name: 'Predict selected machine' })).toBeEnabled();
    await screen.findByRole('table');
    expect(within(controls).getByRole('button', { name: 'Refresh ranking' })).toBeEnabled();
    expect(within(controls).getByText('7 days')).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Highest Failure-Risk Machines (Next 7 Days)' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Selected Machine Prediction' })).toBeInTheDocument();
    expect(screen.getByLabelText('Prototype information')).toBeVisible();
  });

  it('compares without changing the selected machine or requesting a prediction', async () => {
    const fetch = vi.fn().mockResolvedValue(jsonResponse(rankingFixture())); vi.stubGlobal('fetch', fetch);
    renderDashboard(); const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText('Historical machine'), '99');
    await screen.findByRole('table');
    expect(screen.getByLabelText('Historical machine')).toHaveValue('99');
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe('/api/v1/component3/top-risk-machines');
    expect(screen.queryByTestId('failure-probability')).not.toBeInTheDocument();
    expect(screen.getByText('Choose a machine to explore')).toBeVisible();
    expect(screen.queryByText('No machine is automatically chosen or predicted.')).not.toBeInTheDocument();
    expect(screen.getByText(/Comparison does not automatically request a prediction/)).toBeVisible();
  });

  it('supports direct prediction while the automatic ranking is still pending', async () => {
    const pending = deferred<Response>();
    const fetch = vi.fn((url: string) => url.endsWith('top-risk-machines') ? pending.promise
      : Promise.resolve(jsonResponse(riskFixture())));
    vi.stubGlobal('fetch', fetch);
    renderDashboard();
    await userEvent.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    expect(await screen.findByTestId('failure-probability')).toHaveTextContent('27.18%');
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls[1][0]).toBe('/api/v1/component3/predict-risk');
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    await act(async () => { pending.resolve(jsonResponse(rankingFixture())); });
    expect(await screen.findByRole('table')).toBeVisible();
    expect(screen.getByTestId('failure-probability')).toHaveTextContent('27.18%');
    expect(screen.getByLabelText('Historical machine')).toHaveValue('1');
  });

  it('supports direct prediction even if automatic ranking fails', async () => {
    vi.stubGlobal('fetch', vi.fn((url: string) => Promise.resolve(url.endsWith('top-risk-machines')
      ? jsonResponse({ detail: 'Test failure' }, 503) : jsonResponse(riskFixture({ machine_id: 99 })))));
    renderDashboard();
    expect(await screen.findByRole('alert')).toHaveTextContent('Ranking unavailable');
    await userEvent.selectOptions(screen.getByLabelText('Historical machine'), '99');
    await userEvent.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    expect(await screen.findByTestId('failure-probability')).toHaveTextContent('27.18%');
    expect(screen.getByLabelText('Historical machine')).toHaveValue('99');
  });

  it('keeps technical provenance collapsed, concise evidence labels and critical limitations visible', async () => {
    renderDashboard(); const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    await screen.findByTestId('failure-probability');
    const summary = screen.getByText('Model & Data Info');
    expect(summary.closest('details')).not.toHaveAttribute('open');
    for (const hash of screen.getAllByText('sha256:test-only')) expect(hash).not.toBeVisible();
    expect(screen.getByText('test-request')).not.toBeVisible();
    for (const key of ['H', 'M', 'F', 'C', 'R']) expect(screen.getByTestId(`evidence-${key}`)).toBeVisible();
    expect(screen.getByText('Not guaranteed log coverage.')).toBeVisible();
    expect(screen.getByText('Constant at 100% here; not factory data quality.')).toBeVisible();
    expect(screen.getByText('Event recency, not database freshness.')).toBeVisible();
    expect(screen.getByLabelText('Prototype information')).toHaveTextContent('Production lines are illustrative');
    await user.click(summary);
    expect(screen.getByText('test-request')).toBeVisible();
    for (const hash of screen.getAllByText('sha256:test-only')) expect(hash).toBeVisible();
    expect(screen.getByText(/not real apparel-factory records/)).toBeVisible();
  });

  it('highlights the user-selected ranked machine and retains the ranking', async () => {
    const fetch = vi.fn().mockResolvedValueOnce(jsonResponse(rankingFixture()))
      .mockResolvedValueOnce(jsonResponse(riskFixture({ machine_id: 20, predicted_failure_probability: 0.8 })));
    vi.stubGlobal('fetch', fetch); renderDashboard(); const user = userEvent.setup();
    await screen.findByRole('table');
    const view = screen.getByRole('button', { name: 'View prediction for machine 20' });
    expect(view).not.toHaveAttribute('aria-current');
    await user.click(view);
    await screen.findByTestId('failure-probability');
    expect(view).toHaveAttribute('aria-current', 'true');
    expect(view.closest('tr')).toHaveAttribute('data-selected', 'true');
    expect(within(view.closest('tr')!).getByText('Selected')).toBeVisible();
    expect(within(screen.getByRole('table')).getAllByRole('row')).toHaveLength(11);
    await user.selectOptions(screen.getByLabelText('Historical machine'), '19');
    expect(view).not.toHaveAttribute('aria-current');
    expect(within(view.closest('tr')!).queryByText('Selected')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'View prediction for machine 19' })).toHaveAttribute('aria-current', 'true');
  });

  it('supports keyboard submission and focusable research details', async () => {
    renderDashboard();
    const user = userEvent.setup();
    screen.getByRole('button', { name: 'Predict selected machine' }).focus();
    await user.keyboard('{Enter}');
    await screen.findByTestId('failure-probability');
    const summary = screen.getByText('Model & Data Info'); summary.focus();
    expect(summary).toHaveFocus();
    // Native summary Enter/Space behavior is verified in the real browser suite;
    // jsdom does not implement that browser default action.
  });

  it('uses the same machine name and line in the selector, ranking and prediction without sending line metadata', async () => {
    const fetch = vi.fn((url: string, options: RequestInit) => {
      const body = JSON.parse(options.body as string);
      if (url.endsWith('top-risk-machines')) return Promise.resolve(jsonResponse(rankingFixture(body)));
      return Promise.resolve(jsonResponse(riskFixture({ ...body,
        predicted_failure_probability: body.machine_id === 20 ? 0.8 : 0.25 })));
    });
    vi.stubGlobal('fetch', fetch); renderDashboard(); const user = userEvent.setup();
    const table = await screen.findByRole('table');
    const view = screen.getByRole('button', { name: 'View prediction for machine 20' });
    const cells = within(view.closest('tr')!).getAllByRole('cell');
    expect(cells[1]).toHaveTextContent('Machine 020');
    expect(cells[2]).toHaveTextContent('Line 02');
    await user.click(view);
    expect(await screen.findByTestId('selected-machine-name')).toHaveTextContent('Machine 020');
    expect(screen.getByTestId('selected-machine-line')).toHaveTextContent('Line 02');
    expect(within(screen.getByLabelText('Historical machine')).getByRole('option', { selected: true }))
      .toHaveTextContent('Machine 020 · Line 02');
    await user.selectOptions(screen.getByLabelText('Historical machine'), '64');
    await user.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    expect(await screen.findByTestId('selected-machine-name')).toHaveTextContent('Machine 064');
    expect(screen.getByTestId('selected-machine-line')).toHaveTextContent('Line 07');
    expect(screen.getByRole('table')).toBe(table);
    const predictionBodies = fetch.mock.calls.filter(([url]) => url.endsWith('predict-risk'))
      .map(([, options]) => JSON.parse(options.body as string));
    expect(predictionBodies).toEqual([
      { machine_id: 20, prediction_time: '2015-09-02T06:00:00' },
      { machine_id: 64, prediction_time: '2015-09-02T06:00:00' },
    ]);
  });
});
