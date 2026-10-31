import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';
import { checkIndustrialTheme, checkNavigationFocus, checkTextContrast } from './helpers/theme';
import type { ExperimentalPrediction, ExperimentalRanking, Horizon } from '../src/components/component3/experimental/experimentalApi';

const enabled = process.env.E2E_EXPERIMENTAL_ENABLED === 'true';
const endpoint = '/api/v1/component3/experimental/predict-risk';
const rankEndpoint = '/api/v1/component3/experimental/top-risk-machines';

test('disabled unified feature preserves LR dashboard and APIs', async ({ page }) => {
  test.skip(enabled, 'Requires both flags off');
  const requests: string[] = []; page.on('request', r => requests.push(r.url()));
  await page.goto('/component3/experimental-risk');
  await expect(page).toHaveURL(/\/component3\/predict-risk$/);
  await expect(page.getByLabel('Rank machines by')).toHaveCount(0);
  await expect(page.locator('.ranking-table tbody tr')).toHaveCount(10);
  await page.getByRole('button', { name: 'Predict selected machine' }).click();
  await expect(page.getByTestId('failure-probability')).toBeVisible();
  expect(requests.some(url => url.includes('/api/v1/component3/experimental/'))).toBe(false);
  for (const path of [endpoint, rankEndpoint]) expect((await page.request.post(path, { data: { machine_id: 1, prediction_time: '2015-09-02T06:00:00' } })).status()).toBe(404);
});

for (const [mode, width, height] of [['desktop', 1440, 1100], ['tablet', 1024, 1180], ['mobile', 390, 844]] as const) {
  test(`unified real v3 cards, ranking, evidence and isolation (${mode})`, async ({ page }, testInfo) => {
    test.skip(!enabled, 'Requires both experimental flags'); test.setTimeout(60_000);
    await page.setViewportSize({ width, height });
    const errors: string[] = []; const requests: string[] = [];
    page.on('pageerror', e => errors.push(e.message)); page.on('request', r => requests.push(r.url()));
    const lines = readFileSync('../research/component3/experimental_multihorizon_v3/run_003/heldout_predictions.csv', 'utf8').trim().split(/\r?\n/);
    const headers = lines.shift()!.split(',');
    const rows = lines.map(line => Object.fromEntries(line.split(',').map((value, i) => [headers[i], value]))).filter(r => r.method === 'unpooled_hgb');
    const saved = (mid: number, time: string, h: Horizon) => Number(rows.find(r => Number(r.machineID) === mid && r.prediction_time.replace(' ', 'T') === time && Number(r.horizon) === h)!.p);
    async function verifyRanking(response: Awaited<ReturnType<Page['waitForResponse']>>, time: string, h: Horizon) {
      expect(response.status()).toBe(200); const actual: ExperimentalRanking = await response.json();
      const expected = rows.filter(r => r.prediction_time.replace(' ', 'T') === time && Number(r.horizon) === h)
        .sort((a, b) => Number(b.p) - Number(a.p) || Number(a.machineID) - Number(b.machineID)).slice(0, 10);
      expect(actual.horizon_days).toBe(h);
      expect(actual.machines.map(m => m.machine_id)).toEqual(expected.map(r => Number(r.machineID)));
      for (const row of actual.machines) for (const days of [7, 14, 30] as const) expect(row[`p${days}`]).toBe(saved(row.machine_id, time, days));
      await expect(page.locator('.ranking-table tbody tr')).toHaveCount(10); return actual;
    }
    async function verifyPrediction(response: Awaited<ReturnType<Page['waitForResponse']>>, mid: number, time: string) {
      expect(response.status()).toBe(200); const actual: ExperimentalPrediction = await response.json();
      expect(actual.machine_id).toBe(mid); expect(actual.prediction_time).toBe(time);
      expect(actual.experimental && actual.research_only).toBe(true);
      expect(actual.p7 <= actual.p14 && actual.p14 <= actual.p30).toBe(true);
      for (const days of [7, 14, 30] as const) {
        expect(actual[`p${days}`]).toBe(saved(mid, time, days));
        await expect(page.getByTestId(`experimental-p${days}`)).toHaveText(`${(actual[`p${days}`] * 100).toFixed(2)}%`);
      }
      const e = actual.evidence_profile;
      await expect(page.getByTestId('evidence-M')).toHaveText(e.M_prior_maintenance_events.toLocaleString('en-GB'));
      await expect(page.getByTestId('evidence-F')).toHaveText(e.F_prior_failure_events.toLocaleString('en-GB'));
      await expect(page.getByTestId('evidence-errors')).toHaveText(String(e.error_events_prior));
      await expect(page.getByTestId('error-windows')).toHaveText(`${e.error_events_7d} / ${e.error_events_30d} / ${e.error_events_90d}`);
      await expect(page.getByTestId('evidence-error-recency')).toHaveText(e.days_since_last_error === null ? 'No prior error observed' : `${e.days_since_last_error.toLocaleString('en-GB', { maximumFractionDigits: 2 })} ${e.days_since_last_error === 1 ? 'day' : 'days'}`);
      await expect(page.getByRole('heading', { name: `Machine ${String(mid).padStart(3, '0')} · Line ${String(Math.floor((mid - 1) / 10) + 1).padStart(2, '0')}` })).toBeVisible(); return actual;
    }
    let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
    await page.route(`**${rankEndpoint}`, async route => { await gate; await route.continue(); });
    const initial = page.waitForResponse(r => r.url().endsWith(rankEndpoint));
    await page.goto('/component3/experimental-risk');
    await expect(page).toHaveURL(/\/component3\/predict-risk$/);
    await page.getByLabel('Historical machine').selectOption('64'); release();
    const time = '2015-09-02T06:00:00'; let ranking = await verifyRanking(await initial, time, 7);
    await page.unroute(`**${rankEndpoint}`);
    await expect(page.getByLabel('Historical machine')).toHaveValue('64');
    await expect(page.getByTestId('experimental-p7')).toHaveCount(0);
    await expect(page.locator('.research-details')).not.toHaveAttribute('open');
    await checkIndustrialTheme(page); await checkNavigationFocus(page); await checkTextContrast(page);
    await page.screenshot({ path: testInfo.outputPath(`unified-${mode}-ranking-only.png`), fullPage: true });
    await page.getByLabel('Historical machine').selectOption('1');
    let waiting = page.waitForResponse(r => r.url().endsWith(endpoint));
    await page.getByRole('button', { name: 'Predict selected machine' }).click();
    const first = await verifyPrediction(await waiting, 1, time); await checkTextContrast(page);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`unified-${mode}-populated.png`), fullPage: true });
    console.log(JSON.stringify({ mode, input: [1, time], p7: first.p7, p14: first.p14, p30: first.p30, evidence: first.evidence_profile }));
    for (const h of [7, 14, 30] as const) {
      if (h !== 7) {
        const rankWait = page.waitForResponse(r => r.url().endsWith(rankEndpoint));
        await page.getByLabel('Rank machines by').selectOption(String(h));
        ranking = await verifyRanking(await rankWait, time, h); await expect(page.getByTestId('experimental-p7')).toBeVisible();
      }
      const top = ranking.machines[0]; waiting = page.waitForResponse(r => r.url().endsWith(endpoint));
      await page.getByRole('button', { name: `View prediction for machine ${top.machine_id}`, exact: true }).click();
      const result = await verifyPrediction(await waiting, top.machine_id, time);
      expect([result.p7, result.p14, result.p30]).toEqual([top.p7, top.p14, top.p30]);
      expect(result.model_version).toBe(ranking.model_version);
      await expect(page.locator('tr[data-selected="true"] .ranking-probability')).toHaveText(`${(top[`p${h}`] * 100).toFixed(2)}%`);
    }
    await page.getByLabel('Historical machine').selectOption('100');
    const nextTime = '2015-11-25T06:00:00'; const changedRank = page.waitForResponse(r => r.url().endsWith(rankEndpoint));
    await page.getByLabel('Prediction snapshot').selectOption(nextTime);
    await expect(page.getByTestId('experimental-p7')).toHaveCount(0); await verifyRanking(await changedRank, nextTime, 30);
    await expect(page.getByLabel('Historical machine')).toHaveValue('100');
    waiting = page.waitForResponse(r => r.url().endsWith(endpoint));
    await page.getByRole('button', { name: 'Predict selected machine' }).click(); await verifyPrediction(await waiting, 100, nextTime);
    await page.locator('.research-details > summary').click();
    await expect(page.getByText(/not real apparel-factory records/)).toBeVisible();
    await expect(page.getByText(/not validated evidence-loss relationships for v3/)).toBeVisible(); await checkTextContrast(page);
    expect(requests.some(url => /\/api\/v1\/component3\/(predict-risk|top-risk-machines)$/.test(url))).toBe(false);
    await page.screenshot({ path: testInfo.outputPath(`unified-${mode}-audit.png`), fullPage: true });
    await page.route(`**${endpoint}`, route => route.abort('failed'));
    await page.getByRole('button', { name: 'Predict selected machine' }).click();
    await expect(page.getByRole('alert')).toContainText('No substitute probabilities'); await expect(page.getByTestId('experimental-p7')).toHaveCount(0);
    await page.route(`**${rankEndpoint}`, route => route.abort('failed'));
    await page.getByRole('button', { name: 'Refresh ranking' }).click();
    await expect(page.getByText('Ranking unavailable')).toBeVisible(); await expect(page.locator('.ranking-table')).toHaveCount(0);
    await page.screenshot({ path: testInfo.outputPath(`unified-${mode}-error.png`), fullPage: true });
    await page.unroute(`**${endpoint}`); await page.unroute(`**${rankEndpoint}`);
    const reloadRank = page.waitForResponse(r => r.url().endsWith(rankEndpoint)); await page.reload();
    await verifyRanking(await reloadRank, time, 7); await expect(page.getByLabel('Historical machine')).toHaveValue('1');
    await expect(page.getByTestId('experimental-p7')).toHaveCount(0); expect(errors).toEqual([]);
  });
}
