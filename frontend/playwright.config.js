import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  workers: 1,
  use: {
    baseURL: process.env.DASHBOARD_URL || 'http://127.0.0.1:5173',
    channel: process.env.DASHBOARD_BROWSER || 'chrome',
    screenshot: 'only-on-failure',
    viewport: { width: 1440, height: 1000 },
  },
});
