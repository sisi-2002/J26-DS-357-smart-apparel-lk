import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import App from '../../../app/App';
import { jsonResponse, rankingFixture, riskFixture } from '../../../test/fixtures';

function renderDashboard() {
  return render(<MemoryRouter initialEntries={['/component3/predict-risk']}><App /></MemoryRouter>);
}

describe('machine-risk dashboard presentation and workflow', () => {
  it('has one shared control section and two independently labelled result panels', () => {
    renderDashboard();
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Machine Failure-Risk Explorer');
    const controls = screen.getByRole('region', { name: 'Choose a historical snapshot' });
    expect(within(controls).getByLabelText('Prediction snapshot')).toBeInTheDocument();
    expect(screen.getAllByLabelText('Prediction snapshot')).toHaveLength(1);
    expect(within(controls).getByLabelText('Historical machine')).toBeInTheDocument();
    expect(within(controls).getByRole('button', { name: 'Predict selected machine' })).toBeEnabled();
    expect(within(controls).getByRole('button', { name: 'Compare all 100 machines' })).toBeEnabled();
    expect(within(controls).getByText('7 days')).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Top 10 Machines' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Selected Machine Prediction' })).toBeInTheDocument();
    expect(screen.getByLabelText('Synthetic data limitations')).toBeVisible();
  });

  it('compares without changing the selected machine or requesting a prediction', async () => {
    const fetch = vi.fn().mockResolvedValue(jsonResponse(rankingFixture())); vi.stubGlobal('fetch', fetch);
    renderDashboard(); const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText('Historical machine'), '99');
    await user.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
    await screen.findByRole('table');
    expect(screen.getByLabelText('Historical machine')).toHaveValue('99');
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe('/api/v1/component3/top-risk-machines');
    expect(screen.queryByTestId('failure-probability')).not.toBeInTheDocument();
    expect(screen.getByText('Choose a machine to explore')).toBeVisible();
    expect(screen.queryByText('No machine is automatically chosen or predicted.')).not.toBeInTheDocument();
    expect(screen.getByText(/Comparison does not automatically request a prediction/)).toBeVisible();
  });

  it('supports direct prediction without requesting or populating the ranking', async () => {
    const fetch = vi.fn().mockResolvedValue(jsonResponse(riskFixture())); vi.stubGlobal('fetch', fetch);
    renderDashboard();
    await userEvent.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    expect(await screen.findByTestId('failure-probability')).toHaveTextContent('27.18%');
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe('/api/v1/component3/predict-risk');
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('keeps technical provenance collapsed and all five evidence explanations visible', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(riskFixture())));
    renderDashboard(); const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Predict selected machine' }));
    await screen.findByTestId('failure-probability');
    const summary = screen.getByText('Research details');
    expect(summary.closest('details')).not.toHaveAttribute('open');
    expect(screen.getByText('sha256:test-only')).not.toBeVisible();
    expect(screen.getByText('test-request')).not.toBeVisible();
    for (const key of ['H', 'M', 'F', 'C', 'R']) expect(screen.getByTestId(`evidence-${key}`)).toBeVisible();
    expect(screen.getByText(/This does not establish logging coverage/)).toBeVisible();
    expect(screen.getByText(/not factory CMMS quality or prediction reliability/)).toBeVisible();
    expect(screen.getByText(/not database freshness/)).toBeVisible();
    await user.click(summary);
    expect(screen.getByText('test-request')).toBeVisible();
    expect(screen.getByText('sha256:test-only')).toBeVisible();
  });

  it('highlights the user-selected ranked machine and retains the ranking', async () => {
    const fetch = vi.fn().mockResolvedValueOnce(jsonResponse(rankingFixture()))
      .mockResolvedValueOnce(jsonResponse(riskFixture({ machine_id: 20, predicted_failure_probability: 0.8 })));
    vi.stubGlobal('fetch', fetch); renderDashboard(); const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Compare all 100 machines' }));
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
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(riskFixture()))); renderDashboard();
    const user = userEvent.setup();
    screen.getByRole('button', { name: 'Predict selected machine' }).focus();
    await user.keyboard('{Enter}');
    await screen.findByTestId('failure-probability');
    const summary = screen.getByText('Research details'); summary.focus();
    expect(summary).toHaveFocus();
    // Native summary Enter/Space behavior is verified in the real browser suite;
    // jsdom does not implement that browser default action.
  });
});
