import { render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { rankingFixture } from '../../../test/fixtures';
import TopRiskMachinesPanel from './TopRiskMachinesPanel';

function show(probabilities: number[]) {
  const result = rankingFixture();
  result.machines = result.machines.map((machine, index) => ({
    ...machine, predicted_failure_probability: probabilities[index],
  }));
  render(<TopRiskMachinesPanel predictionTime={result.prediction_time} selectedMachineId={99}
    loading={false} error={null} result={result} onSelect={vi.fn()} />);
  return { result, rows: within(screen.getByRole('table')).getAllByRole('row').slice(1) };
}

describe('comparison probability presentation', () => {
  it('keeps an absolute scale, API percentage precision and returned order, including very close values', () => {
    const values = [1, 0.523767, 0.500002, 0.500001, 0.1, 0.01, 0.001, 0.000001, 0, 0];
    const expected = ['100.00%', '52.3767%', '50.0002%', '50.0001%', '10.00%', '1.00%', '0.10%', '0.0001%', '0.00%', '0.00%'];
    const { result, rows } = show(values);
    rows.forEach((row, index) => {
      const cells = within(row).getAllByRole('cell');
      expect(cells[1]).toHaveTextContent(String(result.machines[index].machine_id).padStart(3, '0'));
      expect(cells[2]).toHaveTextContent(expected[index]);
      expect(row.querySelector('.probability-fill')).toHaveStyle({ width: `${values[index] * 100}%` });
      expect(row.querySelector('.probability-track')).toHaveAttribute('aria-hidden', 'true');
    });
    expect(screen.getByText('0–100% scale')).toBeVisible();
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
