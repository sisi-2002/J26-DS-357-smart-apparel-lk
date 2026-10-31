import { expect, type Page } from '@playwright/test';

export async function checkIndustrialTheme(page: Page) {
  // Verify the resting palette rather than the darker primary-button hover state.
  await page.mouse.move(0, 0);
  const brand = await page.locator('html').evaluate(root => {
    const style = getComputedStyle(root);
    return {
      colors: ['deep-navy', 'textile-teal', 'soft-white', 'sage-mist', 'warm-amber']
        .map(name => style.getPropertyValue(`--color-${name}`).trim()),
      scheme: style.colorScheme,
    };
  });
  expect(brand.colors).toEqual(['#142D4E', '#087F8C', '#F5F7F8', '#D9E9E8', '#E4A75A']);
  expect(brand.scheme).toBe('light');
  await expect(page.locator('html')).toHaveCSS('background-color', 'rgb(245, 247, 248)');
  for (const shell of ['.sidebar', '.topbar']) {
    await expect(page.locator(shell)).toHaveCSS('background-color', 'rgb(20, 45, 78)');
  }
  await expect(page.getByRole('heading', { level: 1 })).toHaveCSS('color', 'rgb(20, 45, 78)');
  await expect(page.locator('.panel').first()).toHaveCSS('background-color', 'rgb(255, 255, 255)');
  const active = page.getByRole('navigation', { name: 'Main navigation' }).locator('a[aria-current="page"]');
  await expect(active).toHaveCSS('background-color', 'rgb(8, 127, 140)');
  await expect(active).toHaveCSS('box-shadow', 'rgb(217, 233, 232) 3px 0px 0px 0px inset');
  if (await page.locator('.evidence-study').count()) {
    await expect(page.getByLabel('Synthetic data limitations')).toHaveCSS('border-top-color', 'rgb(228, 167, 90)');
    await expect(page.locator('.notice-icon')).toHaveCSS('background-color', 'rgb(228, 167, 90)');
  } else {
    await expect(page.getByLabel('Synthetic data limitations')).toHaveCount(0);
    await expect(page.getByLabel('Prototype information')).toContainText('Production lines are illustrative');
  }
  const primary = page.getByRole('button', { name: 'Predict selected machine' });
  if (await primary.count()) await expect(primary).toHaveCSS('background-color', 'rgb(8, 127, 140)');
}

export async function checkTextContrast(page: Page) {
  const failures = await page.locator('body').evaluate(root => {
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
      const color = rgb(element instanceof SVGElement ? style.fill : style.color);
      const foreground = background.map((value, index) => color[index] * color[3] + value * (1 - color[3]));
      const a = luminance(foreground); const b = luminance(background);
      const ratio = (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
      // Use the normal-text requirement even for large headings/numbers.
      if (ratio < 4.5) failures.push({ text: node.textContent.trim().slice(0, 100), ratio });
    }
    return failures;
  });
  expect(failures, 'visible shell and feature text meet WCAG AA contrast').toEqual([]);
}

export async function checkNavigationFocus(page: Page) {
  await page.keyboard.press('Tab');
  const active = page.getByRole('navigation', { name: 'Main navigation' }).locator('a[aria-current="page"]');
  await active.focus();
  await expect(active).toBeFocused();
  await expect(active).toHaveCSS('outline-color', 'rgb(228, 167, 90)');
  await expect(active).toHaveCSS('outline-width', '3px');
  expect(await active.evaluate(element => element.matches(':focus-visible'))).toBe(true);
}
