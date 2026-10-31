import { expect, test } from '@playwright/test';
import type { FailureRiskResponse, TopRiskMachinesResponse } from '../src/components/component3/types/component3.types';

const time = '2015-09-02T06:00:00';
for (const viewport of ['desktop', 'mobile'] as const) {
  test(`real Top 10 ranking and selected evidence (${viewport})`, async ({ page }, testInfo) => {
    test.setTimeout(60_000);
    if (viewport === 'mobile') await page.setViewportSize({ width: 390, height: 844 });
    let rankingResponses = 0;
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    // Development StrictMode can start and abort the initial effect request.
    // Count completed batches, and ensure machine selection never reloads one.
    page.on('response', response => {
      if (response.url().endsWith('/top-risk-machines') && response.ok()) rankingResponses++;
    });
    const pending = page.waitForResponse(response => response.url().endsWith('/top-risk-machines'));
    await page.goto('/component3/predict-risk');
    const response = await pending;
    expect(response.status()).toBe(200);
    expect(response.request().postDataJSON()).toEqual({ prediction_time: time });
    const ranking: TopRiskMachinesResponse = await response.json();
    expect(ranking.scored_machine_count).toBe(100);
    expect(ranking.machines).toHaveLength(10);
    expect(ranking.research_only).toBe(true);
    expect(ranking.data_status).toBe('synthetic_azure_historical_demo');
    expect(new Set(ranking.machines.map(machine => machine.machine_id)).size).toBe(10);
    const table = page.getByRole('table');
    await expect(table.getByRole('row')).toHaveCount(11);
    const selectorLabels = await page.getByLabel('Historical machine').locator('option').allTextContents();
    expect(selectorLabels).toHaveLength(100);
    expect(selectorLabels[63]).toBe('Machine 064 · Line 07');
    expect(selectorLabels[70]).toBe('Machine 071 · Line 08');
    expect(selectorLabels[21]).toBe('Machine 022 · Line 03');
    for (const [index, machine] of ranking.machines.entries()) {
      const row = table.getByRole('row').nth(index + 1);
      await expect(row.getByRole('cell').nth(0)).toHaveText(String(machine.rank));
      await expect(row.getByRole('cell').nth(1)).toHaveText(`Machine ${String(machine.machine_id).padStart(3, '0')}`);
      await expect(row.getByRole('cell').nth(2)).toHaveText(`Line ${String(Math.floor((machine.machine_id - 1) / 10) + 1).padStart(2, '0')}`);
      const percentage = (machine.predicted_failure_probability * 100).toLocaleString('en-GB', {
        minimumFractionDigits: 2, maximumFractionDigits: 2,
      });
      await expect(row.getByRole('cell').nth(3)).toHaveText(`${percentage}%`);
      // Check the rendered track ratio against the actual API probability,
      // including the top row: the longest bar must not be normalized to 100%.
      const track = (await row.locator('.probability-track').boundingBox())!;
      const fill = (await row.locator('.probability-fill').boundingBox())!;
      expect(Math.abs(fill.width - track.width * machine.predicted_failure_probability)).toBeLessThan(0.1);
    }

    // Desktop verifies details for every ranked row; mobile verifies switching
    // between two ranked machines while retaining the complete list.
    const selected = viewport === 'desktop' ? ranking.machines : ranking.machines.slice(0, 2);
    for (const machine of selected) {
      const detailPending = page.waitForResponse(response => response.url().endsWith('/predict-risk')
        && response.request().postDataJSON()?.machine_id === machine.machine_id);
      await page.getByRole('button', { name: `View prediction for machine ${machine.machine_id}`, exact: true }).click();
      const detailResponse = await detailPending;
      expect(detailResponse.status()).toBe(200);
      expect(detailResponse.request().postDataJSON()).toEqual({ machine_id: machine.machine_id, prediction_time: time });
      const detail: FailureRiskResponse = await detailResponse.json();
      expect(detail.machine_id).toBe(machine.machine_id);
      expect(detail.prediction_time).toBe(time);
      expect(detail.predicted_failure_probability).toBe(machine.predicted_failure_probability);
      expect(detail.model_version).toBe(ranking.model_version);
      await expect(page.getByTestId('failure-probability')).toHaveText(`${(detail.predicted_failure_probability * 100).toFixed(2)}%`);
      await expect(page.getByLabel('Historical machine')).toHaveValue(String(machine.machine_id));
      await expect(page.getByTestId('selected-machine-name')).toHaveText(`Machine ${String(machine.machine_id).padStart(3, '0')}`);
      await expect(page.getByTestId('selected-machine-line')).toHaveText(`Line ${String(Math.floor((machine.machine_id - 1) / 10) + 1).padStart(2, '0')}`);
      const evidence = detail.evidence_profile;
      const days = (value: number | null) => value === null ? 'No prior maintenance observed'
        : `${value.toLocaleString('en-GB', { maximumFractionDigits: 2 })} days`;
      for (const [key, value] of Object.entries({
        H: days(evidence.H_observed_maintenance_span_days), R: days(evidence.R_days_since_latest_maintenance),
        M: evidence.M_prior_maintenance_events.toLocaleString('en-GB'),
        F: evidence.F_prior_failure_events.toLocaleString('en-GB'), C: `${evidence.C_published_field_completeness_pct}%`,
      })) await expect(page.getByTestId(`evidence-${key}`)).toHaveText(value);
      await expect(table.getByRole('row')).toHaveCount(11);
    }
    expect(rankingResponses).toBe(1);
    await expect(page.getByRole('alert')).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath('top-risk-real-prediction.png'), fullPage: true });
    console.log(JSON.stringify({ viewport, prediction_time: ranking.prediction_time, machines: ranking.machines,
      selected_details_verified: selected.map(machine => machine.machine_id), model_version: ranking.model_version }));
    const selectedMachine = await page.getByLabel('Historical machine').inputValue();
    const nextRanking = page.waitForResponse(response => response.url().endsWith('/top-risk-machines')
      && response.request().postDataJSON()?.prediction_time === '2015-09-09T06:00:00');
    await page.getByLabel('Prediction snapshot').selectOption('2015-09-09T06:00:00');
    expect((await nextRanking).status()).toBe(200);
    await expect(page.getByRole('table', { name: /9 Sep 2015/ }).getByRole('row')).toHaveCount(11);
    await expect(page.getByTestId('failure-probability')).toHaveCount(0);
    await expect(page.getByLabel('Historical machine')).toHaveValue(selectedMachine);
    expect(await page.getByLabel('Historical machine').locator('option').allTextContents()).toEqual(selectorLabels);
    expect(rankingResponses).toBe(2);
    const reloadResponse = page.waitForResponse(response => response.url().endsWith('/top-risk-machines'));
    await page.reload();
    const reloaded: TopRiskMachinesResponse = await (await reloadResponse).json();
    expect(reloaded.machines).toEqual(ranking.machines);
    expect(reloaded.model_version).toBe(ranking.model_version);
    expect(await page.getByLabel('Historical machine').locator('option').allTextContents()).toEqual(selectorLabels);
    await expect(page.getByRole('table').getByRole('row')).toHaveCount(11);
    expect(errors).toEqual([]);
  });
}

// The LR regression workflow runs against the flag-off server.
test.skip(process.env.E2E_EXPERIMENTAL_ENABLED === 'true', 'Stable LR workflow; exercised with flags off');
