import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import App from '../../../app/App';
import snapshot from '../data/n2Study.snapshot.json';
import { readEvidenceStudy } from '../api/evidenceStudyData';
import EvidenceStudyPage from './EvidenceStudyPage';

describe('N2 bundled research study', () => {
  it('places three preliminary findings after the disclaimer and derives their results from the snapshot', () => {
    render(<EvidenceStudyPage />);
    const findings = screen.getByRole('region', { name: 'Key Research Findings — Preliminary' });
    const notice = screen.getByLabelText('Synthetic data limitations');
    const experiment = screen.getByRole('region', { name: 'How did held-out prediction error change?' });
    expect(notice.compareDocumentPosition(findings) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(findings.compareDocumentPosition(experiment) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    const articles = within(findings).getAllByRole('article');
    expect(articles).toHaveLength(3);
    const lowest = snapshot.conditions.reduce((a, b) => a.brier < b.brier ? a : b);
    expect(articles[0]).toHaveTextContent(`${lowest.lookback_days}-day condition achieved the lowest observed Brier score: ${lowest.brier.toFixed(6)}`);
    expect(articles[0]).toHaveTextContent('Models were separately refitted');
    expect(articles[0]).toHaveTextContent('does not show that shorter history is universally better');
    expect(articles[1]).toHaveTextContent('observational and potentially confounded');
    expect(articles[1]).toHaveTextContent('no causal conclusion or validated sufficiency threshold');
    expect(snapshot.conditions.every(condition => condition.incremental[4].mae > condition.incremental[1].mae)).toBe(true);
    expect(articles[2]).toHaveTextContent('Negative preliminary finding:');
    expect(articles[2]).toHaveTextContent('slightly worsened loss-estimation MAE in all three history conditions');
  });

  it('starts with technical details collapsed while main charts, tables and critical limitations remain visible', () => {
    render(<EvidenceStudyPage />);
    for (const title of ['Evidence-only loss regression versus a constant-mean baseline',
      'All five saved alternatives, including RMSE and R²', 'Research methodology and provenance']) {
      expect(screen.getByText(title, { exact: true }).closest('details')).not.toHaveAttribute('open');
    }
    expect(screen.getByText(/Risk-only uses p and p²/)).not.toBeVisible();
    expect(screen.getByText(snapshot.provenance[0].path, { exact: true })).not.toBeVisible();
    expect(screen.getAllByRole('img')).toHaveLength(3);
    for (const name of ['Original failure predictions · 7-day Brier score', 'Full observed history · H quantile-group results',
      'Second-stage loss estimation · out-of-fold MAE']) expect(screen.getByRole('table', { name })).toBeVisible();
    expect(screen.getByRole('table', { name: /All predeclared incremental alternatives/ })).not.toBeVisible();
    expect(screen.getByRole('table', { name: /Second-stage evidence-only regression/ })).not.toBeVisible();
    expect(screen.getByText(/not verified logging coverage/)).toBeVisible();
    expect(screen.getByText(/not database freshness/)).toBeVisible();
    expect(screen.getByText('C · Constant at 100%')).toBeVisible();
    const h = snapshot.conditions[0].evidence.H.groups;
    const min = Math.min(...h.map(group => group.evidence_min));
    const max = Math.max(...h.map(group => group.evidence_max));
    expect(screen.getByText(`Loss axis starts at 0. Evidence axis: ${min}–${max} (starts above zero).`)).toBeVisible();
  });

  it('opens from navigation with no backend calls and separates the uncalibrated experiments', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    render(<MemoryRouter initialEntries={['/component3/predict-risk']}><App /></MemoryRouter>);
    await userEvent.click(screen.getByRole('link', { name: 'Evidence study' }));
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Maintenance Evidence Sufficiency Study');
    expect(screen.getByText(/Original uncalibrated 7-day N2 pilot models/)).toBeVisible();
    expect(screen.getByText(/Function 1 uses a separate calibrated model/)).toBeVisible();
    expect(screen.getByLabelText('Synthetic data limitations')).toBeVisible();
    expect(fetch).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('link', { name: /Machine risk/ }));
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Machine Failure-Risk Explorer');
  });

  it('shows verified saved metrics, paired sample counts and the negative finding', () => {
    render(<EvidenceStudyPage />);
    const brier = screen.getByRole('table', { name: 'Original failure predictions · 7-day Brier score' });
    expect(within(brier).getAllByRole('row').slice(1).map(row => row.textContent)).toEqual([
      'Full1,3000.092460', '180 days1,3000.091828', '90 days1,3000.090315',
    ]);
    const comparison = screen.getByRole('table', { name: 'Second-stage loss estimation · out-of-fold MAE' });
    expect(within(comparison).getAllByRole('row').slice(1).map(row => row.textContent)).toEqual([
      'Full0.1071210.107742-0.000621', '180 days0.1282030.129435-0.001232', '90 days0.0973570.098818-0.001461',
    ]);
    expect(screen.getByText(/Negative finding: risk-only slightly outperformed/)).toBeVisible();
    expect(screen.getByText(/Shorter history is not universally better/)).toBeVisible();
    expect(screen.getByText(/MAE measures error when estimating that realized squared loss/)).toBeVisible();
    expect(screen.getByText('146')).toBeVisible();
  });

  it('selects every saved evidence series with all numerical rows and preserves four-group 90-day F', async () => {
    render(<EvidenceStudyPage />); const user = userEvent.setup();
    for (const condition of snapshot.conditions) {
      await user.selectOptions(screen.getByLabelText('History condition'), condition.id);
      for (const symbol of ['H', 'M', 'F', 'R'] as const) {
        await user.selectOptions(screen.getByLabelText('Evidence characteristic'), symbol);
        const series = condition.evidence[symbol];
        expect(screen.getByRole('status')).toHaveTextContent(series.rho.toFixed(4));
        const table = screen.getByRole('table', { name: `${condition.label} · ${symbol} quantile-group results` });
        const rows = within(table).getAllByRole('row').slice(1);
        expect(rows).toHaveLength(series.groups.length);
        series.groups.forEach((group, i) => {
          const cells = within(rows[i]).getAllByRole('cell');
          expect(cells[0]).toHaveTextContent(`${group.evidence_min}–${group.evidence_max}`);
          expect(cells[1]).toHaveTextContent(group.evidence_mean.toFixed(3));
          expect(cells[2]).toHaveTextContent(String(group.n_snapshots));
          expect(cells[3]).toHaveTextContent(String(group.n_machines));
          expect(cells[4]).toHaveTextContent(group.mean_squared_prediction_error_brier.toFixed(6));
          expect(cells[5]).toHaveTextContent(`${(group.observed_failure_prevalence * 100).toFixed(2)}%`);
        });
      }
    }
    await user.selectOptions(screen.getByLabelText('Evidence characteristic'), 'F');
    expect(screen.getByRole('status')).toHaveTextContent('4 observed quantile groups');
    expect(screen.getByRole('img', { name: /F: Prior failure events versus/ })).toBeVisible();
  }, 15_000); // Exhaustively checks all 12 selections and every saved group row.

  it('explains H/R/C and exposes complete methodology, all five alternatives and source hashes', async () => {
    render(<EvidenceStudyPage />); const user = userEvent.setup();
    expect(screen.getByText(/not verified logging coverage/)).toBeVisible();
    expect(screen.getByText(/not database freshness/)).toBeVisible();
    expect(screen.getByText('C · Constant at 100%')).toBeVisible();
    const disclosure = screen.getByText('Research methodology and provenance');
    expect(disclosure.closest('details')).not.toHaveAttribute('open');
    await user.click(disclosure);
    for (const source of snapshot.provenance) {
      expect(screen.getByText(source.path, { exact: true })).toBeVisible();
      expect(screen.getByText(`SHA-256 ${source.sha256}`)).toBeVisible();
    }
    expect(screen.getByText(/Test data inspected during earlier model development/)).toBeVisible();
    await user.click(screen.getByText('All five saved alternatives, including RMSE and R²'));
    expect(within(screen.getByRole('table', { name: /All predeclared incremental alternatives/ })).getAllByRole('row')).toHaveLength(16);
    expect(screen.getByText('-0.000669')).toBeVisible(); // Negative baseline R² is preserved.
    await user.click(screen.getByText('Evidence-only loss regression versus a constant-mean baseline'));
    expect(screen.getByRole('table', { name: /Second-stage evidence-only regression/ })).toBeVisible();
  });

  it.each([null, {}, { ...snapshot, schema_version: 'unknown' }])('fails explicitly for an unavailable or invalid bundle', value => {
    render(<EvidenceStudyPage snapshot={value} />);
    expect(screen.getByRole('alert')).toHaveTextContent('Research results unavailable');
    expect(screen.getByRole('alert')).toHaveTextContent('No substitute experimental results');
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Synthetic data limitations')).toBeVisible();
  });

  it('rejects internally inconsistent aggregates and provenance rather than displaying fallback numbers', () => {
    const badCounts = structuredClone(snapshot); badCounts.conditions[0].evidence.H.groups[0].n_snapshots++;
    const badDelta = structuredClone(snapshot); badDelta.conditions[1].delta_mae_risk_minus_evidence = 0.9;
    const badHash = structuredClone(snapshot); badHash.provenance[0].sha256 = 'invalid';
    const badC = structuredClone(snapshot); badC.C_constant_pct = 99;
    const badGroups = structuredClone(snapshot); badGroups.conditions[2].evidence.F.groups.push(badGroups.conditions[2].evidence.F.groups[0]);
    for (const invalid of [badCounts, badDelta, badHash, badC, badGroups]) expect(() => readEvidenceStudy(invalid)).toThrow(/invalid or inconsistent/);
  });
});
