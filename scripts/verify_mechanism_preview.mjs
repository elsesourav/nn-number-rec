import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const screenshotDir = '/Users/sourav/.gemini/antigravity/brain/81cca49b-0f27-4067-9d35-1aecd85ee43e/screenshots';
if (!fs.existsSync(screenshotDir)) {
  fs.mkdirSync(screenshotDir, { recursive: true });
}

async function run() {
  console.log('🚀 Starting Mechanism Preview Verification...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  console.log('Navigating to http://localhost:3000...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await new Promise(r => setTimeout(r, 1000));

  // 1. Initial State: Graph Tab is active
  await page.screenshot({ path: path.join(screenshotDir, '01_visualizer_graph_mode.png'), fullPage: true });
  console.log('✓ Captured 01_visualizer_graph_mode.png');

  // 2. Switch to Mechanism Tab
  const mechanismBtn = await page.waitForSelector('[data-testid="tab-visualizer-mechanism"]');
  await mechanismBtn.click();
  await new Promise(r => setTimeout(r, 600));

  await page.screenshot({ path: path.join(screenshotDir, '02_mechanism_step1_input.png'), fullPage: true });
  console.log('✓ Switched to Mechanism Tab. Captured 02_mechanism_step1_input.png');

  // 3. Step through to Step 2 (Synapses)
  const nextBtn = await page.waitForSelector('[data-testid="btn-mechanism-next"]');
  await nextBtn.click();
  await new Promise(r => setTimeout(r, 400));
  await page.screenshot({ path: path.join(screenshotDir, '03_mechanism_step2_synapses.png'), fullPage: true });
  console.log('✓ Advanced to Step 2. Captured 03_mechanism_step2_synapses.png');

  // 4. Step through to Step 3 (Activation Function Curve)
  await nextBtn.click();
  await new Promise(r => setTimeout(r, 400));
  await page.screenshot({ path: path.join(screenshotDir, '04_mechanism_step3_activation.png'), fullPage: true });
  console.log('✓ Advanced to Step 3 (Activation). Captured 04_mechanism_step3_activation.png');

  // 5. Step through to Step 4 (Deep Propagation)
  await nextBtn.click();
  await new Promise(r => setTimeout(r, 400));
  await page.screenshot({ path: path.join(screenshotDir, '05_mechanism_step4_deep_layers.png'), fullPage: true });
  console.log('✓ Advanced to Step 4 (Deep Layers). Captured 05_mechanism_step4_deep_layers.png');

  // 6. Step through to Step 5 (Softmax Probabilities)
  await nextBtn.click();
  await new Promise(r => setTimeout(r, 400));
  await page.screenshot({ path: path.join(screenshotDir, '06_mechanism_step5_softmax.png'), fullPage: true });
  console.log('✓ Advanced to Step 5 (Softmax). Captured 06_mechanism_step5_softmax.png');

  // 7. Step through to Step 6 (Loss Evaluation)
  await nextBtn.click();
  await new Promise(r => setTimeout(r, 400));
  await page.screenshot({ path: path.join(screenshotDir, '07_mechanism_step6_loss.png'), fullPage: true });
  console.log('✓ Advanced to Step 6 (Loss). Captured 07_mechanism_step6_loss.png');

  // 8. Step through to Step 7 (Backpropagation & Momentum)
  await nextBtn.click();
  await new Promise(r => setTimeout(r, 400));
  await page.screenshot({ path: path.join(screenshotDir, '08_mechanism_step7_backprop.png'), fullPage: true });
  console.log('✓ Advanced to Step 7 (Backpropagation). Captured 08_mechanism_step7_backprop.png');

  // 9. Test Reset button
  const resetBtn = await page.waitForSelector('[data-testid="btn-mechanism-reset"]');
  await resetBtn.click();
  await new Promise(r => setTimeout(r, 300));
  console.log('✓ Reset back to Step 1.');

  // 10. Switch to Mobile Viewport (375x812 iPhone size)
  console.log('Testing Mobile Viewport (375x812)...');
  await page.setViewport({ width: 375, height: 812 });
  await new Promise(r => setTimeout(r, 500));
  await page.screenshot({ path: path.join(screenshotDir, '09_mobile_375_mechanism.png'), fullPage: true });
  console.log('✓ Captured 09_mobile_375_mechanism.png');

  // 11. Switch back to Graph mode on Mobile
  const graphBtn = await page.waitForSelector('[data-testid="tab-visualizer-graph"]');
  await graphBtn.click();
  await new Promise(r => setTimeout(r, 400));
  await page.screenshot({ path: path.join(screenshotDir, '10_mobile_375_graph.png'), fullPage: true });
  console.log('✓ Captured 10_mobile_375_graph.png');

  await browser.close();
  console.log('🎉 Verification completed successfully!');
}

run().catch(err => {
  console.error('❌ Verification failed:', err);
  process.exit(1);
});
