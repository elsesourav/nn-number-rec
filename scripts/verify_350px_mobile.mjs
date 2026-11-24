import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const screenshotDir = path.resolve('screenshots');

if (!fs.existsSync(screenshotDir)) {
  fs.mkdirSync(screenshotDir, { recursive: true });
}

async function run() {
  console.log('🚀 Starting 350px Mobile Viewport Verification...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  // EXACT 350px screen width requested by user
  await page.setViewport({ width: 350, height: 780, deviceScaleFactor: 2 });

  console.log('Navigating to http://localhost:3000...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 800));

  // Check for horizontal overflow
  const pageMetrics = await page.evaluate(() => {
    return {
      windowWidth: window.innerWidth,
      bodyScrollWidth: document.body.scrollWidth,
      htmlScrollWidth: document.documentElement.scrollWidth,
      hasHorizontalOverflow: document.documentElement.scrollWidth > window.innerWidth,
    };
  });
  console.log('Page Width Metrics:', pageMetrics);

  // 1. Initial 350px Mobile View (Header + Graph Mode)
  await page.screenshot({ path: path.join(screenshotDir, '30_mobile_350px_initial_graph.png') });
  console.log('✓ Captured 30_mobile_350px_initial_graph.png');

  // Verify Default Stroke Width is 18px
  const strokeSlider = await page.waitForSelector('[aria-label="Stroke Width"]');
  const strokeVal = await page.evaluate(el => el.getAttribute('aria-valuenow'), strokeSlider);
  console.log('Default Stroke Width Value:', strokeVal);

  // 2. Capture Header Close-Up
  const header = await page.waitForSelector('header');
  await header.screenshot({ path: path.join(screenshotDir, '31_header_350px_compact.png') });
  console.log('✓ Captured 31_header_350px_compact.png');

  // 3. Test Mode Switcher: Click Train (capture sliding pill animation)
  const trainTabBtn = await page.waitForSelector('[data-testid="tab-train"]');
  await trainTabBtn.click();
  await new Promise(r => setTimeout(r, 100)); // capture during transition
  await page.screenshot({ path: path.join(screenshotDir, '32_train_switch_mid_transition_350px.png') });

  await new Promise(r => setTimeout(r, 350)); // settled in train mode
  await page.screenshot({ path: path.join(screenshotDir, '33_train_mode_settled_350px.png') });
  console.log('✓ Captured Train Mode on 350px');

  // Switch back to Recognize mode
  const recTabBtn = await page.waitForSelector('[data-testid="tab-recognize"]');
  await recTabBtn.click();
  await new Promise(r => setTimeout(r, 350));

  // 4. Test Visualizer Switcher: Click Mechanism (capture sliding pill animation)
  const mechanismTabBtn = await page.waitForSelector('[data-testid="tab-visualizer-mechanism"]');
  await mechanismTabBtn.click();
  await new Promise(r => setTimeout(r, 100)); // mid-transition
  await page.screenshot({ path: path.join(screenshotDir, '34_vis_switch_mid_transition_350px.png') });

  await new Promise(r => setTimeout(r, 400)); // settled in mechanism mode
  await page.screenshot({ path: path.join(screenshotDir, '35_mechanism_settled_350px.png') });
  console.log('✓ Captured Mechanism on 350px (35_mechanism_settled_350px.png)');

  // Switch back to Graph mode
  const graphTabBtn = await page.waitForSelector('[data-testid="tab-visualizer-graph"]');
  await graphTabBtn.click();
  await new Promise(r => setTimeout(r, 400));

  // Check Graph Canvas Nodes Clearance on 350px
  const graphClearance = await page.evaluate(() => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return 'no canvas';
    const rect = canvas.getBoundingClientRect();
    const rightControls = document.querySelector('[data-testid="tab-visualizer-graph"]')?.parentElement?.getBoundingClientRect();
    return {
      canvasRect: { x: rect.left, y: rect.top, w: rect.width, h: rect.height },
      topControlsBottom: rightControls ? rightControls.bottom : null,
      topControlsRight: rightControls ? rightControls.right : null,
      windowWidth: window.innerWidth
    };
  });
  console.log('Graph Layout Clearance Metrics:', graphClearance);

  await page.screenshot({ path: path.join(screenshotDir, '36_final_graph_fitted_350px.png') });
  console.log('✓ Captured 36_final_graph_fitted_350px.png');

  await browser.close();
  console.log('🎉 350px Mobile verification completed!');
}

run().catch((err) => {
  console.error('Verification failed:', err);
  process.exit(1);
});
