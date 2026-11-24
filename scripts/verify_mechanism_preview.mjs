import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const screenshotDir = '/Users/sourav/.gemini/antigravity/brain/81cca49b-0f27-4067-9d35-1aecd85ee43e/screenshots';
if (!fs.existsSync(screenshotDir)) {
  fs.mkdirSync(screenshotDir, { recursive: true });
}

async function run() {
  console.log('🚀 Starting Verification of Auto-Fit & Reactive Data-Only Flow...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  console.log('Navigating to http://localhost:3000...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 800));

  // 1. Switch to Mechanism Tab
  const mechanismBtn = await page.waitForSelector('[data-testid="tab-visualizer-mechanism"]');
  await mechanismBtn.click();
  await new Promise(r => setTimeout(r, 600));

  // 2. Capture all blocks fitted on desktop
  await page.screenshot({ path: path.join(screenshotDir, '15_all_blocks_fitted_desktop.png'), fullPage: true });
  console.log('✓ Captured 15_all_blocks_fitted_desktop.png');

  // 3. Trigger manual flow replay (simulating training event)
  const replayBtn = await page.waitForSelector('[data-testid="btn-workflow-replay"]');
  await replayBtn.click();
  await new Promise(r => setTimeout(r, 200)); // middle of flow
  await page.screenshot({ path: path.join(screenshotDir, '16_active_training_flow_packet.png'), fullPage: true });
  console.log('✓ Captured 16_active_training_flow_packet.png');

  // Wait for flow to finish and return to idle
  await new Promise(r => setTimeout(r, 1200));
  await page.screenshot({ path: path.join(screenshotDir, '17_idle_after_flow_completed.png'), fullPage: true });
  console.log('✓ Captured 17_idle_after_flow_completed.png');

  // 4. Test Mobile Screen (375x812) to verify all blocks fit on mobile
  console.log('Testing Mobile Screen (375x812)...');
  await page.setViewport({ width: 375, height: 812 });
  await new Promise(r => setTimeout(r, 600));
  await page.screenshot({ path: path.join(screenshotDir, '18_all_blocks_fitted_mobile_375.png'), fullPage: true });
  console.log('✓ Captured 18_all_blocks_fitted_mobile_375.png');

  // 5. Test Tablet / Medium Screen (650x900)
  console.log('Testing Tablet Screen (650x900)...');
  await page.setViewport({ width: 650, height: 900 });
  await new Promise(r => setTimeout(r, 500));
  await page.screenshot({ path: path.join(screenshotDir, '19_all_blocks_fitted_tablet_650.png'), fullPage: true });
  console.log('✓ Captured 19_all_blocks_fitted_tablet_650.png');

  await browser.close();
  console.log('🎉 Verification completed successfully!');
}

run().catch(err => {
  console.error('❌ Verification failed:', err);
  process.exit(1);
});
