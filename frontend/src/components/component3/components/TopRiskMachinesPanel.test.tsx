import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { rankingFixture } from '../../../test/fixtures';
import TopRiskMachinesPanel from './TopRiskMachinesPanel';

function show(probabilities: number[], onSelect = vi.fn()) {
  const result = rankingFixture();
  result.machines = result.machines.map((machine, index) => ({
    ...machine, predicted_failure_probability: probabilities[index],
  }));
  render(<TopRiskMachinesPanel predictionTime={result.prediction_time} selectedMachineId={99}
    loading={false} error={null} result={result} onSelect={onSelect} />);
  return { result, rows: within(screen.getByRole('table')).getAllByRole('row').slice(1) };
}

describe('comparison probability presentation', () => {
  it('rounds only percentage text to two decimals, preserving absolute bars and returned order', () => {
    const values = [1, 0.523767, 0.500002, 0.500001, 0.1, 0.01, 0.001, 0.000001, 0, 0];
    const expected = ['100.00%', '52.38%', '50.00%', '50.00%', '10.00%', '1.00%', '0.10%', '0.00%', '0.00%', '0.00%'];
    const { result, rows } = show(values);
    rows.forEach((row, index) => {
      const cells = within(row).getAllByRole('cell');
      expect(cells[1]).toHaveTextContent(String(result.machines[index].machine_id).padStart(3, '0'));
      expect(cells[1]).toHaveTextContent('Machine');
      expect(cells[2]).toHaveTextContent('Line 02');
      expect(cells[3]).toHaveTextContent(expected[index]);
      expect(cells[3]).toHaveAttribute('title', `Returned probability: ${values[index]}`);
      expect(row.querySelector('.probability-fill')).toHaveStyle({ width: `${values[index] * 100}%` });
      expect(row.querySelector('.probability-track')).toHaveAttribute('aria-hidden', 'true');
    });
    expect(screen.getByText('0–100% scale')).toBeVisible();
  });

  it('passes the unchanged API probability to individual inference selection', async () => {
    const onSelect = vi.fn();
    const { result } = show([0.523767, ...Array<number>(9).fill(0.1)], onSelect);
    await userEvent.click(screen.getByRole('button', { name: 'View prediction for machine 20' }));
    expect(onSelect).toHaveBeenCalledWith(result.machines[0]);
    expect(onSelect.mock.calls[0][0].predicted_failure_probability).toBe(0.523767);
  });

  it.each([0.01, 0.9])('marks positions 1–3 independently of the probability (%s)', value => {
    const { rows } = show(Array<number>(10).fill(value));
    rows.forEach((row, index) => {
      expect(row.querySelector('.rank-position')).toHaveAttribute('data-leading', String(index < 3));
      // A 90% maximum must still occupy 90% of the track, never 100%.
      expect(row.querySelector('.probability-fill')).toHaveStyle({ width: `${value * 100}%` });
    });
    expect(screen.getByText('Coral marks relative positions 1–3 only, not validated danger or urgency levels.')).toBeVisible();
  });
});
