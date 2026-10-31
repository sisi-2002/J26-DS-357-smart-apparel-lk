import { expect, test } from '@playwright/test';
import type { FailureRiskResponse } from '../src/components/component3/types/component3.types';

for (const mode of ['desktop', 'mobile'] as const) {
  test(`actual React → proxy → FastAPI → saved model prediction (${mode})`, async ({ page }, testInfo) => {
    if (mode === 'mobile') await page.setViewportSize({ width: 390, height: 844 });
    const consoleErrors: string[] = [];
    page.on('pageerror', error => consoleErrors.push(error.message));
    await page.goto('/component3/predict-risk');
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('Machine Failure-Risk Explorer');
    await expect(page.getByTestId('failure-probability')).toHaveCount(0);
    await page.getByLabel('Historical machine').selectOption('1');
    await page.getByLabel('Prediction snapshot').selectOption('2015-09-02T06:00:00');

    const responsePromise = page.waitForResponse(response =>
      response.url().endsWith('/api/v1/component3/predict-risk') && response.request().method() === 'POST');
    await page.getByRole('button', { name: /Predict selected machine/ }).click();
    const response = await responsePromise;
    expect(response.status()).toBe(200);
    expect(response.request().postDataJSON()).toEqual({ machine_id: 1, prediction_time: '2015-09-02T06:00:00' });
    const actual: FailureRiskResponse = await response.json();
    expect(actual.research_only).toBe(true);
    expect(actual.data_status).toBe('synthetic_azure_historical_demo');
    expect(actual.model_version).toMatch(/^sha256:[a-f0-9]{64}$/);
    expect(actual.horizon_days).toBe(7);
    expect(actual.predicted_failure_probability).toBeGreaterThanOrEqual(0);
    expect(actual.predicted_failure_probability).toBeLessThanOrEqual(1);
    await expect(page.getByTestId('failure-probability')).toHaveText(`${(actual.predicted_failure_probability * 100).toFixed(2)}%`);
    const evidence = actual.evidence_profile;
    for (const [key, value] of Object.entries({
      M: evidence.M_prior_maintenance_events.toLocaleString('en-GB'),
      F: evidence.F_prior_failure_events.toLocaleString('en-GB'),
      C: `${evidence.C_published_field_completeness_pct}%`,
      H: evidence.H_observed_maintenance_span_days === null ? 'No prior maintenance observed'
        : `${evidence.H_observed_maintenance_span_days.toLocaleString('en-GB', { maximumFractionDigits: 2 })} days`,
      R: evidence.R_days_since_latest_maintenance === null ? 'No prior maintenance observed'
        : `${evidence.R_days_since_latest_maintenance.toLocaleString('en-GB', { maximumFractionDigits: 2 })} days`,
    })) await expect(page.getByTestId(`evidence-${key}`)).toHaveText(value);
    await expect(page.getByRole('alert')).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath('actual-prediction.png'), fullPage: true });
    console.log(JSON.stringify({ viewport: mode, probability: actual.predicted_failure_probability,
      evidence: actual.evidence_profile, model_version: actual.model_version, request_id: actual.request_id }));

    // A changed input must clear the completed result immediately.
    await page.getByLabel('Historical machine').selectOption('2');
    await expect(page.getByTestId('failure-probability')).toHaveCount(0);
    expect(consoleErrors).toEqual([]);
  });
}

// The LR regression workflow runs against the flag-off server.
test.skip(process.env.E2E_EXPERIMENTAL_ENABLED === 'true', 'Stable LR workflow; exercised with flags off');
