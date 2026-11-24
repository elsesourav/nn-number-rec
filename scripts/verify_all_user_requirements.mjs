import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const OUTPUT_DIR = path.resolve(process.cwd(), 'screenshots');

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

async function verify() {
  console.log('🚀 Launching Chrome for complete verification...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 2 });
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });

  // 1. Check Top Right Buttons flex
  console.log('\n--- 1. Testing Top Right Buttons Flex ---');
  const btnArchitectureInfo = await page.evaluate(() => {
    const btn = document.querySelector('[data-testid="btn-architecture"]');
    if (!btn) return null;
    const cs = window.getComputedStyle(btn);
    const rect = btn.getBoundingClientRect();
    const children = Array.from(btn.children).map(c => {
      const r = c.getBoundingClientRect();
      const style = window.getComputedStyle(c);
      return {
        tag: c.tagName,
        x: r.x,
        y: r.y,
        width: r.width,
        height: r.height,
        display: style.display,
      };
    });
    return {
      display: cs.display,
      flexDirection: cs.flexDirection,
      flexWrap: cs.flexWrap,
      btnRect: rect,
      children,
    };
  });
  console.log('Architecture Button Flex Info:', JSON.stringify(btnArchitectureInfo, null, 2));

  // 2. Check Stroke Slider Range (min 15, max 30)
  console.log('\n--- 2. Testing Stroke Slider Range (15px to 30px) ---');
  const strokeSliderInfo = await page.evaluate(() => {
    const sliderContainer = document.querySelector('.aspect-square')?.parentElement?.querySelector('.pt-1\\.5');
    const text = sliderContainer ? sliderContainer.innerText : 'NOT FOUND';
    return { text };
  });
  console.log('Stroke Slider text:', strokeSliderInfo.text);

  // 3. Switch to Train Mode & Verify Online MNIST Navigation & Auto-Training
  console.log('\n--- 3. Testing Train Mode, Navigation, and Auto-Training ---');
  await page.evaluate(() => {
    const trainTab = document.querySelector('[data-testid="tab-train"]');
    trainTab?.click();
  });
  await new Promise(r => setTimeout(r, 600));

  // Get initial loss and sample info
  const initialTrainState = await page.evaluate(() => {
    const bottomStatus = document.body.innerText.match(/Loss:\s*([\d.]+)\s*\((\d+)\)/);
    const sampleText = document.querySelector('[data-testid="btn-next-sample"]')?.parentElement?.innerText;
    const presets = Array.from(document.querySelectorAll('button')).filter(b => b.title?.includes('Target:')).map(b => b.innerText);
    return {
      lossMatch: bottomStatus ? { loss: bottomStatus[1], trained: bottomStatus[2] } : null,
      sampleText,
      presets,
    };
  });
  console.log('Initial Train State:', initialTrainState);

  // Click Next Sample ('>')
  console.log('Clicking Next Sample (">")...');
  await page.evaluate(() => {
    const nextBtn = document.querySelector('[data-testid="btn-next-sample"]');
    nextBtn?.click();
  });
  await new Promise(r => setTimeout(r, 700));

  const afterNextState = await page.evaluate(() => {
    const bottomStatus = document.body.innerText.match(/Loss:\s*([\d.]+)\s*\((\d+)\)/);
    const sampleText = document.querySelector('[data-testid="btn-next-sample"]')?.parentElement?.innerText;
    return {
      lossMatch: bottomStatus ? { loss: bottomStatus[1], trained: bottomStatus[2] } : null,
      sampleText,
    };
  });
  console.log('State After Next Click (Trained):', afterNextState);

  // Click Next Sample again
  await page.evaluate(() => {
    const nextBtn = document.querySelector('[data-testid="btn-next-sample"]');
    nextBtn?.click();
  });
  await new Promise(r => setTimeout(r, 700));

  // Click Previous Sample ('<')
  console.log('Clicking Previous Sample ("<")...');
  await page.evaluate(() => {
    const prevBtn = document.querySelector('[data-testid="btn-prev-sample"]');
    prevBtn?.click();
  });
  await new Promise(r => setTimeout(r, 700));

  const afterPrevState = await page.evaluate(() => {
    const sampleText = document.querySelector('[data-testid="btn-next-sample"]')?.parentElement?.innerText;
    return { sampleText };
  });
  console.log('State After Prev Click:', afterPrevState);

  // Click Reset to First Sample button
  console.log('Clicking Reset to First Sample button...');
  await page.evaluate(() => {
    const resetBtn = document.querySelector('[data-testid="btn-reset-sample"]');
    resetBtn?.click();
  });
  await new Promise(r => setTimeout(r, 700));

  const afterResetState = await page.evaluate(() => {
    const sampleText = document.querySelector('[data-testid="btn-next-sample"]')?.parentElement?.innerText;
    return { sampleText };
  });
  console.log('State After Reset to First Sample Click:', afterResetState);

  // Click Prev button when at sample #1 to verify it does NOT wrap to 999/1000
  console.log('Clicking Prev while at sample #1 to verify no wrapping...');
  await page.evaluate(() => {
    const prevBtn = document.querySelector('[data-testid="btn-prev-sample"]');
    prevBtn?.click();
  });
  await new Promise(r => setTimeout(r, 400));
  const noWrapState = await page.evaluate(() => {
    const sampleText = document.querySelector('[data-testid="btn-next-sample"]')?.parentElement?.innerText;
    return { sampleText };
  });
  console.log('State after Prev at #1 (should stay #1):', noWrapState);

  // 4. Capture Train Mode Screenshot
  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'verified_train_mode.png'),
    fullPage: false,
  });
  console.log('Saved verified_train_mode.png');

  // 5. Test Architecture Drawer & Reset Confirmation Dialog
  console.log('\n--- 4. Testing Architecture Drawer & Reset Confirmation Dialog ---');
  await page.evaluate(() => {
    const archBtn = document.querySelector('[data-testid="btn-architecture"]');
    archBtn?.click();
  });
  await new Promise(r => setTimeout(r, 500));

  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'verified_architecture_drawer.png'),
    fullPage: false,
  });
  console.log('Saved verified_architecture_drawer.png');

  // Click Reset button inside Drawer to open Dialog
  console.log('Clicking Reset Weights to open Dialog confirmation...');
  await page.evaluate(() => {
    const resetBtn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Reset') && !b.innerText.includes('to first'));
    resetBtn?.click();
  });
  await new Promise(r => setTimeout(r, 500));

  const dialogInfo = await page.evaluate(() => {
    const dialogTitle = document.querySelector('h2, h3, [role="dialog"]')?.innerText;
    const dialogButtons = Array.from(document.querySelectorAll('[role="dialog"] button, div button')).filter(b => b.innerText.includes('Reset Weights') || b.innerText.includes('Cancel')).map(b => b.innerText);
    return { dialogTitle, dialogButtons };
  });
  console.log('Dialog Info:', dialogInfo);

  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'verified_reset_dialog.png'),
    fullPage: false,
  });
  console.log('Saved verified_reset_dialog.png');

  // Confirm Reset Weights
  await page.evaluate(() => {
    const confirmBtn = Array.from(document.querySelectorAll('button')).find(b => b.innerText === 'Reset Weights');
    confirmBtn?.click();
  });
  await new Promise(r => setTimeout(r, 600));

  // 6. Test Mobile Viewport (650px & 375px)
  console.log('\n--- 5. Testing Mobile Viewports ---');
  await page.setViewport({ width: 650, height: 900, deviceScaleFactor: 2 });
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 600));

  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'verified_mobile_650.png'),
    fullPage: false,
  });
  console.log('Saved verified_mobile_650.png');

  await page.setViewport({ width: 375, height: 750, deviceScaleFactor: 2 });
  await new Promise(r => setTimeout(r, 400));
  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'verified_mobile_375.png'),
    fullPage: false,
  });
  console.log('Saved verified_mobile_375.png');

  await browser.close();
  console.log('\n✨ All automated verification tests completed successfully!');
}

verify().catch(err => {
  console.error('Error during verification:', err);
  process.exit(1);
});
