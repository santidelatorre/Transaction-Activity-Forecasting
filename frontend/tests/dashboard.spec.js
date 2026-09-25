import { test, expect } from '@playwright/test';

test('real metrics, explanation and experiment pagination agree with the API', async ({ page, request }) => {
  const source = await (await request.get('/api/v1/dashboard')).json();
  test.skip(!source.available, 'Requires the real V3 evaluation artifacts.');
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  await expect(page.getByRole('heading', { name: /Every transaction matters/ })).toBeVisible();
  await expect(page.locator('#quality')).toContainText(source.metrics.macro_f1.toLocaleString('en-US', { minimumFractionDigits: 4, maximumFractionDigits: 4 }));
  await expect(page.locator('#quality')).toContainText((source.metrics.accuracy * 100).toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 }));
  await expect(page.locator('#progress svg')).toBeVisible();
  await expect(page.locator('#explainability [role="img"]')).toHaveCount(Math.min(6, source.importance.length));
  await page.screenshot({ path: 'test-results/dashboard-desktop.png', fullPage: true });
  await page.getByRole('button', { name: 'Next experiment page' }).click();
  await expect(page.locator('#experiments')).toContainText('Page 2');
  await expect(page.locator('#experiments')).toContainText('Targeted correction');
  await page.getByRole('button', { name: 'Previous experiment page' }).click();
  await expect(page.locator('#experiments')).toContainText('V3-A · family identity');
  await page.getByRole('button', { name: /V3-A · family identity.*E3/ }).click();
  await expect(page.locator('#experiments')).toContainText('Evaluation commit:');
  expect(errors).toEqual([]);
});

test('mobile layout keeps tables inside their scroll container', async ({ page, request }) => {
  const source = await (await request.get('/api/v1/dashboard')).json();
  test.skip(!source.available, 'Requires the real V3 evaluation artifacts.');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.locator('#progress svg')).toBeVisible();
  await expect(page.locator('#experiments tbody tr')).not.toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: 'test-results/dashboard-mobile.png', fullPage: true });
});

test('missing artifacts never become made-up metrics or chart points', async ({ page }) => {
  await page.route('**/api/v1/dashboard', (route) => route.fulfill({ json: { available: false, metrics: null, experiments: [], importance: [] } }));
  await page.route('**/api/v1/experiments?*', (route) => route.fulfill({ json: { experiments: [] } }));
  await page.goto('/');
  await expect(page.getByText(/V3-A results have not been loaded/)).toBeVisible();
  await expect(page.locator('#quality article')).toHaveCount(3);
  await expect(page.locator('#quality article').first()).toContainText('—');
  await expect(page.locator('#progress svg')).toHaveCount(0);
  await expect(page.locator('#explainability [role="img"]')).toHaveCount(0);
});

test('API failure offers a working retry', async ({ page }) => {
  let failed = true;
  await page.route('**/api/v1/dashboard', (route) => failed
    ? route.fulfill({ status: 503, json: { detail: 'Unavailable' } })
    : route.fulfill({ json: { available: false, metrics: null, experiments: [], importance: [] } }));
  await page.goto('/');
  await expect(page.locator('#quality [role="alert"]')).toContainText('503');
  failed = false;
  await page.getByRole('button', { name: 'Refresh results' }).click();
  await expect(page.getByText(/V3-A results have not been loaded/)).toBeVisible();
  await expect(page.locator('#quality [role="alert"]')).toHaveCount(0);
});
