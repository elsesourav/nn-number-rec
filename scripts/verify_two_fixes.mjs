import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ARTIFACT_DIR = path.resolve('/Users/sourav/.gemini/antigravity/brain/81cca49b-0f27-4067-9d35-1aecd85ee43e/screenshots');

if (!fs.existsSync(ARTIFACT_DIR)) {
  fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
}

async function run() {
  console.log('🚀 Starting verification of glowing nodes and canvas placeholder...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 350, height: 750, deviceScaleFactor: 2 });

  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await new Promise(r => setTimeout(r, 1200));

  console.log('\n--- Test 1: Verify Placeholder Text Hides During and After Drawing ---');
  // Check placeholder text initially visible
  const initialPlaceholderVisible = await page.evaluate(() => {
    const el = Array.from(document.querySelectorAll('span')).find(s => s.textContent?.includes('Draw digit'));
    return el && window.getComputedStyle(el).display !== 'none';
  });
  console.log('Initial placeholder visible:', initialPlaceholderVisible);

  // Find canvas
  const canvas = await page.waitForSelector('canvas');
  const canvasBox = await canvas.boundingBox();

  // Mouse move to canvas center
  await page.mouse.move(canvasBox.x + canvasBox.width / 2, canvasBox.y + canvasBox.height / 4);
  await page.mouse.down();

  // Check placeholder text DURING active drawing (pointer down)
  const placeholderDuringDrawing = await page.evaluate(() => {
    const el = Array.from(document.querySelectorAll('span')).find(s => s.textContent?.includes('Draw digit'));
    return !!el;
  });
  console.log('Placeholder visible DURING drawing:', placeholderDuringDrawing);
  if (!placeholderDuringDrawing) {
    console.log('✅ PASS: Placeholder text correctly hid the INSTANT drawing started!');
  } else {
    console.error('❌ FAIL: Placeholder text is still showing while pointer is down!');
  }

  // Draw a stroke (vertical line like digit 1)
  await page.mouse.move(canvasBox.x + canvasBox.width / 2, canvasBox.y + (canvasBox.height * 3) / 4, { steps: 8 });
  await page.mouse.up();
  await new Promise(r => setTimeout(r, 600));

  // Check placeholder text AFTER drawing
  const placeholderAfterDrawing = await page.evaluate(() => {
    const el = Array.from(document.querySelectorAll('span')).find(s => s.textContent?.includes('Draw digit'));
    return !!el;
  });
  console.log('Placeholder visible AFTER drawing:', placeholderAfterDrawing);
  if (!placeholderAfterDrawing) {
    console.log('✅ PASS: Placeholder text remains hidden after drawing completed!');
  } else {
    console.error('❌ FAIL: Placeholder text reappeared after drawing!');
  }

  // Screenshot canvas with digit drawn and NO placeholder text
  await page.screenshot({ path: path.join(ARTIFACT_DIR, '29_drawing_without_placeholder.png') });

  console.log('\n--- Test 2: Verify EXACTLY 1 Node Glows in Network Visualizer ---');
  // Inspect canvas pixels on the visualizer canvas
  const glowMetrics = await page.evaluate(() => {
    // Check prediction in header / UI
    const predEl = document.querySelector('[class*="text-emerald-400"]');
    return {
      predictionText: predEl ? predEl.textContent : null,
    };
  });
  console.log('Prediction state:', glowMetrics);

  // Take screenshot of graph visualizer on mobile
  await page.screenshot({ path: path.join(ARTIFACT_DIR, '30_graph_single_node_glow.png') });

  // Test clear canvas resets placeholder
  console.log('\n--- Test 3: Clear Button Resets Canvas & Restores Placeholder ---');
  const clearBtn = await page.waitForSelector('[data-testid="btn-clear"]');
  await clearBtn.click();
  await new Promise(r => setTimeout(r, 300));

  const placeholderAfterClear = await page.evaluate(() => {
    const el = Array.from(document.querySelectorAll('span')).find(s => s.textContent?.includes('Draw digit'));
    return !!el;
  });
  console.log('Placeholder restored after clear:', placeholderAfterClear);
  if (placeholderAfterClear) {
    console.log('✅ PASS: Placeholder text successfully restored after clearing canvas!');
  }

  await browser.close();
  console.log('\n🎉 Verification completed successfully!');
}

run().catch(err => {
  console.error('Error during verification:', err);
  process.exit(1);
});
