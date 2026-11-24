import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const OUTPUT_DIR = path.resolve(process.cwd(), 'screenshots');

async function testAutoFitAndStroke() {
  console.log('Testing Auto-Fit and Dynamic Stroke Width Adjustment...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 2 });
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 1200));

  // 1. Draw a digit '2' shifted to the right side (extra left margin)
  console.log('Drawing off-center digit 2 with extra side margin...');
  const canvas = await page.$('canvas');
  if (canvas) {
    const box = await canvas.boundingBox();
    if (box) {
      // Draw towards the right side of the canvas
      const startX = box.x + box.width * 0.65;
      const startY = box.y + box.height * 0.25;
      await page.mouse.move(startX, startY);
      await page.mouse.down();
      await page.mouse.move(startX + 35, startY + 5, { steps: 3 });
      await page.mouse.move(startX + 20, startY + 45, { steps: 3 });
      await page.mouse.move(startX - 20, startY + 90, { steps: 4 });
      await page.mouse.move(startX + 45, startY + 95, { steps: 3 });
      await page.mouse.up();
      await new Promise(r => setTimeout(r, 200));

      await page.screenshot({
        path: path.join(OUTPUT_DIR, 'autofit_1_drawn_extra_side.png'),
        fullPage: false,
      });
      console.log('Saved: autofit_1_drawn_extra_side.png');

      // Click Auto-Fit button to normalize and center it
      console.log('Clicking Auto-Fit button...');
      await page.click('[data-testid="btn-autofit"]');
      await new Promise(r => setTimeout(r, 400));

      await page.screenshot({
        path: path.join(OUTPUT_DIR, 'autofit_2_normalized_centered.png'),
        fullPage: false,
      });
      console.log('Saved: autofit_2_normalized_centered.png');
    }
  }

  // 2. Load online MNIST digit (sample #1)
  console.log('Showing Online MNIST digit #1...');
  await page.click('[data-testid="btn-step-train"]');
  await new Promise(r => setTimeout(r, 400));

  // Change stroke slider to 54px (Thicker)
  console.log('Setting stroke slider to 54px (Thicker)...');
  await page.evaluate(() => {
    const slider = document.querySelector('input[type="range"][min="18"][max="60"]');
    if (slider) {
      const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      nativeSetter.call(slider, '54');
      slider.dispatchEvent(new Event('input', { bubbles: true }));
    }
  });
  await new Promise(r => setTimeout(r, 500));

  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'online_mnist_thick_stroke_54px.png'),
    fullPage: false,
  });
  console.log('Saved: online_mnist_thick_stroke_54px.png');

  // Change stroke slider to 20px (Thinner)
  console.log('Setting stroke slider to 20px (Thinner)...');
  await page.evaluate(() => {
    const slider = document.querySelector('input[type="range"][min="18"][max="60"]');
    if (slider) {
      const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      nativeSetter.call(slider, '20');
      slider.dispatchEvent(new Event('input', { bubbles: true }));
    }
  });
  await new Promise(r => setTimeout(r, 500));

  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'online_mnist_thin_stroke_20px.png'),
    fullPage: false,
  });
  console.log('Saved: online_mnist_thin_stroke_20px.png');

  await browser.close();
  console.log('Finished Auto-Fit and Stroke verification successfully!');
}

testAutoFitAndStroke().catch(err => {
  console.error(err);
  process.exit(1);
});
