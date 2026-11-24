import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const OUTPUT_DIR = path.resolve(process.cwd(), 'screenshots');

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

async function capture() {
  console.log('Capturing cleaned up UI...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 2 });
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 1200));

  // 1. Initial Clean Fullscreen
  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'clean_01_fullscreen.png'),
    fullPage: false,
  });

  // 2. Crop of Left Panel (directly comparable to user image)
  const leftPanel = await page.$('div.w-full.md\\:w-\\[300px\\]');
  if (leftPanel) {
    await leftPanel.screenshot({
      path: path.join(OUTPUT_DIR, 'clean_02_left_panel_idle.png'),
    });
    console.log('Saved clean_02_left_panel_idle.png');
  }

  // 3. Draw a digit 3
  const canvas = await page.$('canvas');
  if (canvas) {
    const box = await canvas.boundingBox();
    if (box) {
      const cx = box.x + box.width * 0.5;
      const cy = box.y + box.height * 0.45;
      await page.mouse.move(cx - 30, cy - 40);
      await page.mouse.down();
      await page.mouse.move(cx + 25, cy - 40, { steps: 4 });
      await page.mouse.move(cx, cy - 10, { steps: 4 });
      await page.mouse.move(cx + 28, cy + 25, { steps: 4 });
      await page.mouse.move(cx - 30, cy + 30, { steps: 4 });
      await page.mouse.up();
      await new Promise(r => setTimeout(r, 400));
    }
  }

  // Capture with prediction on numpad
  if (leftPanel) {
    await leftPanel.screenshot({
      path: path.join(OUTPUT_DIR, 'clean_03_left_panel_with_prediction.png'),
    });
    console.log('Saved clean_03_left_panel_with_prediction.png');
  }

  // 4. Step online MNIST
  await page.click('[data-testid="btn-step-train"]');
  await new Promise(r => setTimeout(r, 400));

  if (leftPanel) {
    await leftPanel.screenshot({
      path: path.join(OUTPUT_DIR, 'clean_04_left_panel_step_train.png'),
    });
    console.log('Saved clean_04_left_panel_step_train.png');
  }

  await browser.close();
  console.log('Done capturing cleaned UI screenshots!');
}

capture().catch(console.error);
