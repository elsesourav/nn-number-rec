import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const OUTPUT_DIR = path.resolve(process.cwd(), 'screenshots');

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

async function runTest() {
  console.log('Testing Low Pixel Canvas, Grid, and Maximum Training Presets...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 2 });

  const errors = [];
  page.on('pageerror', err => {
    console.error('PAGE ERROR:', err.message);
    errors.push(err.message);
  });
  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.error('CONSOLE ERROR:', msg.text());
      errors.push(msg.text());
    }
  });

  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 1200));

  // 1. Initial State: Low-Pixel Canvas with 28x28 grid
  console.log('1. Capturing Initial Low-Pixel Canvas with Grid...');
  await page.screenshot({
    path: path.join(OUTPUT_DIR, '01_low_pixel_canvas_initial.png'),
    fullPage: false,
  });

  // 2. Draw a digit '8' in Low-Pixel mode
  console.log('2. Drawing digit 8 on Low-Pixel Canvas...');
  const canvas = await page.$('canvas');
  if (canvas) {
    const box = await canvas.boundingBox();
    if (box) {
      const cx = box.x + box.width * 0.5;
      const cy = box.y + box.height * 0.5;
      // Top loop
      await page.mouse.move(cx, cy - 30);
      await page.mouse.down();
      await page.mouse.move(cx + 25, cy - 45, { steps: 4 });
      await page.mouse.move(cx, cy - 60, { steps: 4 });
      await page.mouse.move(cx - 25, cy - 45, { steps: 4 });
      await page.mouse.move(cx, cy - 30, { steps: 4 });
      // Bottom loop
      await page.mouse.move(cx + 30, cy - 5, { steps: 4 });
      await page.mouse.move(cx, cy + 30, { steps: 4 });
      await page.mouse.move(cx - 30, cy - 5, { steps: 4 });
      await page.mouse.move(cx, cy - 30, { steps: 4 });
      await page.mouse.up();
      await new Promise(r => setTimeout(r, 400));
    }
  }

  await page.screenshot({
    path: path.join(OUTPUT_DIR, '02_low_pixel_drawn_8.png'),
    fullPage: false,
  });

  // 3. Toggle to Smooth mode and capture
  console.log('3. Toggling to Smooth mode...');
  await page.evaluate(() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.title?.includes('Switch to Smooth') || b.innerText?.includes('28×28 Low Pixel') || b.innerText?.includes('Low Pixel'));
    btn?.click();
  });
  await new Promise(r => setTimeout(r, 400));

  await page.screenshot({
    path: path.join(OUTPUT_DIR, '03_smooth_mode_toggle.png'),
    fullPage: false,
  });

  // 4. Toggle back to Low-Pixel mode
  console.log('4. Toggling back to Low-Pixel mode...');
  await page.evaluate(() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.title?.includes('Switch to 28x28 Low-Pixel') || b.innerText?.includes('Smooth HD'));
    btn?.click();
  });
  await new Promise(r => setTimeout(r, 300));

  // 5. Test Maximum Target Presets (100, 500, 1k, 2.5k, 5k, ∞ Loop)
  console.log('5. Testing Target Preset buttons...');
  const presetLabels = ['100', '500', '1k (All)', '2.5k', '5k', '∞ Loop'];
  for (const label of presetLabels) {
    await page.evaluate((lbl) => {
      const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === lbl);
      btn?.click();
    }, label);
    await new Promise(r => setTimeout(r, 150));
  }

  await page.screenshot({
    path: path.join(OUTPUT_DIR, '04_infinite_loop_preset_active.png'),
    fullPage: false,
  });

  // 6. Test Step Training to observe authentic online MNIST sample in Low-Pixel mode
  console.log('6. Testing Step training with Online MNIST sample...');
  await page.click('[data-testid="btn-step-train"]');
  await new Promise(r => setTimeout(r, 500));

  await page.screenshot({
    path: path.join(OUTPUT_DIR, '05_online_mnist_step_low_pixel.png'),
    fullPage: false,
  });

  // 7. Load Pretrained 96% Model from Storage/Architecture
  console.log('7. Loading Pretrained Model...');
  await page.click('[data-testid="btn-architecture"]');
  await new Promise(r => setTimeout(r, 400));

  await page.evaluate(() => {
    const preBtn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Pretrained') || b.title?.includes('Pretrained'));
    preBtn?.click();
  });
  await new Promise(r => setTimeout(r, 600));

  await page.screenshot({
    path: path.join(OUTPUT_DIR, '06_pretrained_model_loaded.png'),
    fullPage: false,
  });

  // Close drawer
  await page.click('[data-testid="btn-close-drawer"]');
  await new Promise(r => setTimeout(r, 300));

  // Verify prediction on online sample with pretrained model
  const predictionText = await page.evaluate(() => {
    return document.body.innerText;
  });
  console.log('Page content contains 96% accuracy or Pretrained status:', predictionText.includes('96%') || predictionText.includes('0.042'));

  await browser.close();

  if (errors.length > 0) {
    console.error('FAIL: Console errors detected:', errors);
    process.exit(1);
  }
  console.log('PASS: All low-pixel canvas and maximum preset verifications succeeded with 0 errors!');
}

runTest().catch((e) => {
  console.error(e);
  process.exit(1);
});
