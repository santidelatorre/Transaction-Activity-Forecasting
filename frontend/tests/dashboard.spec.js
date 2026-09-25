import { test, expect } from '@playwright/test';

const score = (value) => value.toLocaleString('en-US', { minimumFractionDigits: 4, maximumFractionDigits: 4 });

test('version and V4 experiment visuals agree with sourced API values', async ({ page, request }) => {
  const source = await (await request.get('/api/v1/dashboard')).json();
  test.skip(!source.available, 'Requires the measured V3 evaluation artifact.');
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  await expect(page).toHaveTitle('Transaction Activity Forecasting · Recurring Insights');
  await expect(page.getByRole('heading', { level: 1, name: 'Transaction Activity Forecasting' })).toBeVisible();
  const logo = page.getByRole('img', { name: 'UBS logo' });
  await expect(logo).toBeVisible();
  expect(await logo.evaluate((image) => image.naturalWidth)).toBeGreaterThan(0);
  await expect(page.locator('#quality')).toContainText(score(source.metrics.macro_f1));
  await expect(page.locator('#progress svg')).toBeVisible();
  for (const row of source.version_history.filter((item) => item.macro_f1 != null)) {
    await expect(page.locator('#progress svg')).toContainText(score(row.macro_f1));
  }
  await expect(page.locator('#progress')).toContainText('No new score');
  await expect(page.locator('#progress')).toContainText('VALID was reused');
  await expect(page.locator('#progress svg line[stroke="#686868"]')).toHaveCount(5);
  const progressBox = await page.locator('#progress').boundingBox();
  const signalsBox = await page.locator('#explainability').boundingBox();
  const comparisonBox = await page.locator('#comparison').boundingBox();
  expect(Math.abs(progressBox.width - comparisonBox.width)).toBeLessThan(2);
  expect(signalsBox.y).toBeGreaterThanOrEqual(progressBox.y + progressBox.height);
  await expect(page.locator('#comparison > div [role="img"]')).toHaveCount(
    source.v4_experiments.reduce((total, row) => total + Number(row.train_oof != null) + Number(row.valid != null), 0),
  );
  for (const row of source.v4_experiments) {
    const visual = page.locator('#comparison').getByText(row.name, { exact: true });
    await expect(visual).toBeVisible();
    await expect(page.locator('#comparison')).toContainText(score(row.train_oof));
    if (row.valid == null) {
      await expect(visual.locator('..').locator('..')).toContainText('Not evaluated');
    } else {
      await expect(visual.locator('..').locator('..')).toContainText(score(row.valid));
    }
  }
  await expect(page.locator('#explainability [role="img"]')).toHaveCount(Math.min(6, source.importance.length));
  await page.getByRole('button', { name: 'Next experiment page' }).click();
  await expect(page.locator('#experiments')).toContainText('Page 2');
  await expect(page.locator('#experiments')).toContainText('Focused correction');
  await page.getByRole('button', { name: 'Previous experiment page' }).click();
  await page.getByRole('button', { name: /V3-A · family identity.*E3/ }).click();
  await expect(page.locator('#experiments')).toContainText('Evaluation commit:');
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/dashboard-desktop.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('mobile layout keeps charts and audit table inside scroll containers', async ({ page, request }) => {
  const source = await (await request.get('/api/v1/dashboard')).json();
  test.skip(!source.available, 'Requires the measured V3 evaluation artifact.');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.getByRole('img', { name: 'UBS logo' })).toBeVisible();
  await expect(page.locator('#progress svg')).toBeVisible();
  await expect(page.locator('#comparison [role="group"]')).toBeVisible();
  await expect(page.locator('#experiments tbody tr')).not.toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: 'test-results/dashboard-mobile.png', fullPage: true });
});

test('unavailable V3 measurements and V4 decision do not become invented scores', async ({ page }) => {
  await page.route('**/api/v1/dashboard', (route) => route.fulfill({ json: {
    available: false, metrics: null, importance: [], experiments: [], v4_experiments: [],
    version_history: [
      { id: 'v1', label: 'V1 baseline', macro_f1: 0.2710242658492452, scope: 'VALID', protocol: 'official' },
      { id: 'v2', label: 'V2 final', macro_f1: 0.391549456, scope: 'VALID', protocol: 'official' },
      { id: 'v3-a', label: 'V3-A', macro_f1: null, scope: 'VALID', kind: 'unavailable' },
      { id: 'v4', label: 'V4 decision', macro_f1: null, kind: 'decision', note: 'V3-A retained' },
    ],
  } }));
  await page.route('**/api/v1/experiments?*', (route) => route.fulfill({ json: { experiments: [] } }));
  await page.goto('/');
  await expect(page.getByText(/local V3-A evaluation artifact is unavailable/)).toBeVisible();
  await expect(page.locator('#quality article').first()).toContainText('—');
  await expect(page.locator('#progress svg')).toContainText('0.2710');
  await expect(page.locator('#progress svg')).toContainText('0.3915');
  await expect(page.locator('#progress svg')).toContainText('No new score');
  await expect(page.locator('#progress svg line[stroke="#686868"]')).toHaveCount(1);
  await expect(page.locator('#comparison [role="img"]')).toHaveCount(0);
});

test('incomparable protocols break the version line', async ({ page }) => {
  await page.route('**/api/v1/dashboard', (route) => route.fulfill({ json: {
    available: false, metrics: null, importance: [], experiments: [], v4_experiments: [],
    version_history: [
      { id: 'v2', label: 'V2', macro_f1: 0.39, scope: 'VALID', protocol: 'official' },
      { id: 'v3-a', label: 'V3-A', macro_f1: 0.42, scope: 'TRAIN OOF', protocol: 'different' },
      { id: 'v4', label: 'V4 decision', macro_f1: null, kind: 'decision' },
    ],
  } }));
  await page.route('**/api/v1/experiments?*', (route) => route.fulfill({ json: { experiments: [] } }));
  await page.goto('/');
  await expect(page.locator('#progress svg')).toContainText('TRAIN OOF');
  await expect(page.locator('#progress svg line[stroke="#686868"]')).toHaveCount(0);
  await expect(page.locator('#progress svg')).toContainText('No new score');
});

test('API failure offers a working retry', async ({ page }) => {
  let failed = true;
  await page.route('**/api/v1/dashboard', (route) => failed
    ? route.fulfill({ status: 503, json: { detail: 'Unavailable' } })
    : route.fulfill({ json: { available: false, metrics: null, experiments: [], importance: [], version_history: [], v4_experiments: [] } }));
  await page.goto('/');
  await expect(page.locator('#quality [role="alert"]')).toContainText('503');
  failed = false;
  await page.getByRole('button', { name: 'Refresh results' }).click();
  await expect(page.getByText(/local V3-A evaluation artifact is unavailable/)).toBeVisible();
  await expect(page.locator('#quality [role="alert"]')).toHaveCount(0);
});
