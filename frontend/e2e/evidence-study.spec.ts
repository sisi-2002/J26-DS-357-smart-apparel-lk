import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { expect, test } from '@playwright/test';
import snapshot from '../src/components/component3/data/n2Study.snapshot.json' with { type: 'json' };

for (const [mode, viewport] of Object.entries({ desktop: { width: 1440, height: 1100 },
  tablet: { width: 1024, height: 1024 }, mobile: { width: 390, height: 844 } })) {
  test(`N2 bundled aggregates, selectors, provenance and responsive layout (${mode})`, async ({ page }) => {
    const captures = path.resolve('test-results/evidence-study');
    await mkdir(captures, { recursive: true });
    await page.setViewportSize(viewport);
    const errors: string[] = []; const apiRequests: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    // Even with all API access denied, this research view must work. It uses
    // the source-controlled snapshot, not local CSVs or live inference calls.
    await page.route(url => url.pathname.startsWith('/api/'), route => {
      apiRequests.push(route.request().url()); return route.abort();
    });
    await page.goto('/component3/evidence-study');
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('Maintenance Evidence Sufficiency Study');
    await expect(page.getByRole('link', { name: 'Evidence study' })).toHaveAttribute('aria-current', 'page');
    await expect(page.getByText(/Original uncalibrated 7-day N2 pilot models/)).toBeVisible();
    await expect(page.getByText(/Negative finding: risk-only slightly outperformed/)).toBeVisible();
    const findings = page.getByRole('region', { name: 'Key Research Findings — Preliminary' });
    await expect(findings.getByRole('article')).toHaveCount(3);
    const lowest = snapshot.conditions.reduce((a, b) => a.brier < b.brier ? a : b);
    await expect(findings.getByRole('article').nth(0)).toContainText(lowest.brier.toFixed(6));
    await expect(findings.getByRole('article').nth(1)).toContainText('observational and potentially confounded');
    await expect(findings.getByRole('article').nth(2)).toContainText('Negative preliminary finding');
    for (const disclosure of await page.locator('.evidence-study details').all()) await expect(disclosure).not.toHaveAttribute('open');
    await expect(page.getByText(/Risk-only uses p and p²/)).not.toBeVisible();
    const brier = page.getByRole('table', { name: 'Original failure predictions · 7-day Brier score' });
    for (const [index, condition] of snapshot.conditions.entries()) {
      await expect(brier.getByRole('row').nth(index + 1).getByRole('cell').last()).toHaveText(condition.brier.toFixed(6));
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: path.join(captures, `${mode}-overview.png`), fullPage: true });
    await findings.screenshot({ path: path.join(captures, `${mode}-key-findings.png`) });

    for (const condition of snapshot.conditions) {
      await page.getByLabel('History condition', { exact: true }).selectOption(condition.id);
      for (const symbol of ['H', 'M', 'F', 'R'] as const) {
        await page.getByLabel('Evidence characteristic', { exact: true }).selectOption(symbol);
        const series = condition.evidence[symbol];
        await expect(page.getByRole('status')).toContainText(series.rho.toFixed(4));
        const table = page.getByRole('table', { name: `${condition.label} · ${symbol} quantile-group results` });
        await expect(table.getByRole('row')).toHaveCount(series.groups.length + 1);
        for (const [index, group] of series.groups.entries()) {
          const cells = table.getByRole('row').nth(index + 1).getByRole('cell');
          await expect(cells.nth(4)).toHaveText(group.mean_squared_prediction_error_brier.toFixed(6));
          await expect(cells.nth(2)).toHaveText(String(group.n_snapshots));
        }
      }
    }
    await page.getByLabel('Evidence characteristic', { exact: true }).selectOption('F');
    await expect(page.getByRole('status')).toContainText('4 observed quantile groups');
    const plot = page.getByRole('img', { name: /F: Prior failure events versus/ });
    await plot.scrollIntoViewIfNeeded();
    await expect(plot.locator('circle')).toHaveCount(4);
    expect(await plot.locator('text').first().evaluate(element => parseFloat(getComputedStyle(element).fontSize))).toBeGreaterThanOrEqual(14);
    await expect(page.getByText('Loss axis starts at 0. Evidence axis: 0–6 (starts at zero).')).toBeVisible();
    const chartRegion = page.getByRole('region', { name: 'Evidence-loss chart, scroll horizontally on small screens' });
    await chartRegion.screenshot({ path: path.join(captures, `${mode}-chart.png`) });
    if (mode === 'mobile') {
      await expect(page.getByText(/On smaller screens, scroll the chart/)).toBeVisible();
      await chartRegion.focus(); await page.keyboard.press('ArrowRight');
      await expect.poll(() => chartRegion.evaluate(element => element.scrollLeft)).toBeGreaterThan(0);
      await chartRegion.evaluate(element => { element.scrollLeft = 0; });
    }
    const notice = page.getByLabel('Synthetic data limitations');
    await expect(notice).toBeVisible();
    expect((await notice.boundingBox())!.y).toBeGreaterThanOrEqual(0);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(captures, `${mode}-90day-f.png`), fullPage: true });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);

    const methodology = page.getByText('Research methodology and provenance', { exact: true });
    await methodology.focus(); await page.keyboard.press('Enter');
    await expect(page.getByText(snapshot.provenance[0].path, { exact: true })).toBeVisible();
    for (const source of snapshot.provenance) await expect(page.getByText(`SHA-256 ${source.sha256}`, { exact: true })).toBeVisible();
    await page.getByText('All five saved alternatives, including RMSE and R²', { exact: true }).click();
    const complete = page.getByRole('table', { name: /All predeclared incremental alternatives/ });
    await expect(complete.getByRole('row')).toHaveCount(16);
    await expect(complete.getByRole('cell', { name: '-0.000669', exact: true })).toBeVisible();
    await page.getByText('Evidence-only loss regression versus a constant-mean baseline', { exact: true }).click();
    await expect(page.getByRole('table', { name: /Second-stage evidence-only regression/ })).toBeVisible();
    await page.keyboard.press('Tab');
    for (const control of [page.getByLabel('History condition', { exact: true }), page.getByLabel('Evidence characteristic', { exact: true }), methodology]) {
      await control.focus();
      await expect(control).toBeFocused();
      expect(await control.evaluate(element => getComputedStyle(element).outlineStyle)).toBe('solid');
      expect(await control.evaluate(element => getComputedStyle(element).outlineWidth)).toBe('3px');
    }
    expect(apiRequests).toEqual([]); expect(errors).toEqual([]);
    console.log(JSON.stringify({ study: mode, source: 'bundled aggregate JSON', apiRequests: apiRequests.length,
      brier: snapshot.conditions.map(condition => condition.brier), selected: '90-day F: four groups' }));
  });
}
