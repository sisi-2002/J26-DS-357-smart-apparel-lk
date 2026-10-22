import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { expect, test, type Page } from '@playwright/test';
import type { FailureRiskResponse, TopRiskMachinesResponse } from '../src/components/component3/types/component3.types';

const captures = path.resolve('test-results/dashboard');
const sizes = { desktop: { width: 1440, height: 1100 }, tablet: { width: 1024, height: 1024 }, mobile: { width: 390, height: 844 } };

async function checkTextContrast(page: Page) {
  const failures = await page.locator('.machine-risk-explorer').evaluate(root => {
    const rgb = (value: string) => {
      const parts = value.match(/[\d.]+/g)!.map(Number);
      return [...parts.slice(0, 3), parts[3] ?? 1];
    };
    const luminance = (color: number[]) => color.slice(0, 3).map(value => {
      const s = value / 255;
      return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    }).reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
    const failures: { text: string; ratio: number }[] = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const element = node.parentElement!;
      if (!node.textContent?.trim() || element.closest('[aria-hidden="true"]')) continue;
      const range = document.createRange(); range.selectNodeContents(node);
      if (![...range.getClientRects()].some(rect => rect.width > 0 && rect.height > 0)) continue;
      const style = getComputedStyle(element);
      if (style.visibility !== 'visible') continue;
      const ancestors: Element[] = [];
      for (let ancestor: Element | null = element; ancestor; ancestor = ancestor.parentElement) ancestors.unshift(ancestor);
      let background = [255, 255, 255];
      for (const ancestor of ancestors) {
        const color = rgb(getComputedStyle(ancestor).backgroundColor);
        background = background.map((value, index) => color[index] * color[3] + value * (1 - color[3]));
      }
      const color = rgb(style.color);
      const foreground = background.map((value, index) => color[index] * color[3] + value * (1 - color[3]));
      const a = luminance(foreground); const b = luminance(background);
      const ratio = (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
      // Use the normal-text requirement even for large headings/numbers.
      if (ratio < 4.5) failures.push({ text: node.textContent.trim().slice(0, 100), ratio });
    }
    return failures;
  });
  expect(failures, 'visible Component 3 text meets WCAG AA contrast').toEqual([]);
}

for (const [size, viewport] of Object.entries(sizes)) {
  test(`dashboard empty, ranking-only and selected states (${size})`, async ({ page }) => {
    test.setTimeout(60_000);
    await page.setViewportSize(viewport);
    await mkdir(captures, { recursive: true });
    let predictionRequests = 0;
    page.on('request', request => {
      if (request.method() === 'POST' && request.url().endsWith('/predict-risk')) predictionRequests++;
    });
    // Hold, then continue the actual batch request to inspect the loading state
    // and a deliberate user selection. All response data still comes from FastAPI.
    let releaseRanking!: () => void;
    const rankingGate = new Promise<void>(resolve => { releaseRanking = resolve; });
    await page.route('**/top-risk-machines', async route => { await rankingGate; await route.continue(); });
    const rankedResponse = page.waitForResponse(response => response.url().endsWith('/top-risk-machines'));
    await page.goto('/component3/predict-risk');
    const controls = page.getByRole('region', { name: 'Choose a historical snapshot' });
    const rankingPanel = page.getByRole('region', { name: 'Highest Failure-Risk Machines (Next 7 Days)' });
    const predictionPanel = page.getByRole('region', { name: 'Selected Machine Prediction' });
    const notice = page.getByLabel('Synthetic data limitations');
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('Machine Failure-Risk Explorer');
    await expect(controls.getByLabel('Prediction snapshot')).toHaveCount(1);
    await expect(page.getByTestId('failure-probability')).toHaveCount(0);
    await expect(page.getByRole('table')).toHaveCount(0);
    await expect(notice).toBeVisible();
    await expect(notice).toContainText('Synthetic Demo · Azure 2015 Data');
    await expect(notice).toContainText('not validated maintenance alerts');
    await expect(controls.getByRole('button', { name: 'Comparing 100 machines…' })).toBeDisabled();
    const inputBox = (await controls.boundingBox())!;
    const left = (await rankingPanel.boundingBox())!;
    const right = (await predictionPanel.boundingBox())!;
    expect(inputBox.y + inputBox.height).toBeLessThanOrEqual(left.y);
    if (size === 'desktop') {
      expect(Math.abs(left.y - right.y)).toBeLessThan(2);
      expect(left.width / (left.width + right.width)).toBeCloseTo(0.4, 1);
      expect(left.x + left.width).toBeLessThan(right.x);
    } else {
      expect(right.y).toBeGreaterThanOrEqual(left.y + left.height);
    }
    expect(right.height).toBeLessThan(360);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await checkTextContrast(page);
    await page.screenshot({ path: path.join(captures, `${size}-empty.png`), fullPage: true });

    // The user explicitly chooses a machine that is not assumed to be ranked.
    await page.getByLabel('Historical machine').selectOption('99');
    releaseRanking();
    const response = await rankedResponse; expect(response.status()).toBe(200);
    const ranking: TopRiskMachinesResponse = await response.json();
    await expect(page.getByRole('table').getByRole('row')).toHaveCount(11);
    await expect(page.getByLabel('Historical machine')).toHaveValue('99');
    await expect(page.getByTestId('failure-probability')).toHaveCount(0);
    expect(predictionRequests).toBe(0);
    await checkTextContrast(page);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(captures, `${size}-ranking-only.png`), fullPage: true });

    const machine = ranking.machines[0];
    const detailsResponse = page.waitForResponse(response => response.url().endsWith('/predict-risk'));
    const view = page.getByRole('button', { name: `View prediction for machine ${machine.machine_id}`, exact: true });
    await view.click();
    const detailHttp = await detailsResponse; expect(detailHttp.status()).toBe(200);
    const detail: FailureRiskResponse = await detailHttp.json();
    expect(detail.predicted_failure_probability).toBe(machine.predicted_failure_probability);
    await expect(page.getByTestId('failure-probability')).toHaveText(`${(detail.predicted_failure_probability * 100).toFixed(2)}%`);
    await expect(view).toHaveAttribute('aria-current', 'true');
    await expect(view.locator('xpath=ancestor::tr')).toHaveAttribute('data-selected', 'true');
    await expect(view.locator('xpath=ancestor::tr').getByText('Selected', { exact: true })).toBeVisible();
    await expect(page.getByRole('table').getByRole('row')).toHaveCount(11);
    await expect(notice).toBeVisible();
    if (size !== 'desktop') {
      await expect(page.getByRole('heading', { name: 'Selected Machine Prediction' })).toBeFocused();
      const sticky = (await notice.boundingBox())!;
      const selectedHeading = (await page.getByRole('heading', { name: 'Selected Machine Prediction' }).boundingBox())!;
      expect(sticky.y).toBeGreaterThanOrEqual(0);
      expect(selectedHeading.y).toBeGreaterThan(sticky.y + sticky.height);
    }
    const research = page.locator('details.research-details');
    await expect(research.locator('summary')).toContainText('Model & Data Info');
    await expect(research).not.toHaveAttribute('open');
    await expect(research.getByText(detail.request_id, { exact: true })).not.toBeVisible();
    await checkTextContrast(page);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(captures, `${size}-selected-prediction.png`), fullPage: true });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await research.locator('summary').focus();
    await page.keyboard.press('Enter');
    await expect(research.getByText(detail.request_id, { exact: true })).toBeVisible();
    await checkTextContrast(page);
    // After keyboard interaction, every interactive control retains a visible
    // focus outline. The existing mobile heading-focus assertion stays intact.
    for (const control of [controls.getByLabel('Prediction snapshot'), controls.getByLabel('Historical machine'),
      controls.getByRole('button', { name: 'Predict selected machine' }),
      controls.getByRole('button', { name: 'Refresh ranking' }), view, research.locator('summary')]) {
      await control.focus();
      await expect(control).toBeFocused();
      expect(await control.evaluate(element => {
        const style = getComputedStyle(element);
        return element.matches(':focus-visible') && style.outlineStyle === 'solid' && parseFloat(style.outlineWidth) >= 3;
      })).toBe(true);
    }
  });
}
