import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const OUTPUT_DIR = path.resolve(process.cwd(), 'screenshots');

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

async function verify() {
  console.log('🚀 Launching Chrome for Auto-Fit & Right-Side Drawer Verification...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 2 });
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });

  // 1. Verify Top-Right Header has ONE unified button
  console.log('\n--- 1. Testing Unified Header Button ---');
  const headerButtons = await page.evaluate(() => {
    const header = document.querySelector('header');
    const rightButtons = Array.from(header?.querySelectorAll('button') || []).filter(b => 
      b.innerText.includes('Architecture') || b.innerText.includes('Storage') || b.title?.includes('Storage')
    );
    return rightButtons.map(b => ({
      text: b.innerText.trim(),
      testId: b.getAttribute('data-testid'),
    }));
  });
  console.log('Right Header Buttons found:', headerButtons);
  if (headerButtons.length !== 1) {
    throw new Error(`Expected exactly 1 combined header button, found ${headerButtons.length}`);
  }

  // 2. Verify Auto-Fit Button is HIDDEN by default (since autoFitOnDraw is ON)
  console.log('\n--- 2. Checking Auto-Fit Button visibility when ON (Default) ---');
  const autoFitBtnInitially = await page.$('[data-testid="btn-autofit"]');
  console.log('Is manual auto-fit button [F] visible when auto-fit is ON?:', autoFitBtnInitially !== null ? 'YES (WRONG)' : 'NO (CORRECT)');
  if (autoFitBtnInitially !== null) {
    throw new Error('Manual autofit button should NOT be visible when autofit is enabled!');
  }

  // 3. Draw a digit and observe debounce & auto-fit
  console.log('\n--- 3. Drawing Digit with Auto-Fit Debounce ---');
  const canvas = await page.$('canvas');
  if (canvas) {
    const box = await canvas.boundingBox();
    const startX = box.x + box.width * 0.35;
    const startY = box.y + box.height * 0.3;
    await page.mouse.move(startX, startY);
    await page.mouse.down();
    await page.mouse.move(startX + 30, startY + 40, { steps: 5 });
    await page.mouse.move(startX, startY + 80, { steps: 5 });
    await page.mouse.up();
    console.log('Mouse released. Waiting 400ms for debounce & auto-fit...');
    await new Promise(r => setTimeout(r, 400));
  }

  const predictionAfterDraw = await page.evaluate(() => {
    return document.body.innerText.match(/Predicted:\s*Digit\s*(\d)/)?.[0] || 'No prediction card';
  });
  console.log('Prediction after auto-fit draw:', predictionAfterDraw);

  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'autofit_default_enabled_drawing.png'),
    fullPage: false,
  });
  console.log('Saved autofit_default_enabled_drawing.png');

  // 4. Click the unified button to open the Right-Side Drawer
  console.log('\n--- 4. Testing Right-Side Slide-Over Drawer ---');
  await page.click('[data-testid="btn-architecture"]');
  await new Promise(r => setTimeout(r, 400));

  const drawerPosition = await page.evaluate(() => {
    const drawerWrapper = document.querySelector('.fixed.inset-0.overflow-hidden.flex');
    const drawerPanel = drawerWrapper?.querySelector('.relative.h-full');
    const rect = drawerPanel?.getBoundingClientRect();
    const computedClass = drawerWrapper?.className;
    return {
      justifyClass: computedClass,
      panelRightOffset: window.innerWidth - (rect?.right || 0),
      panelLeft: rect?.left,
      width: rect?.width,
      tabs: Array.from(drawerPanel?.querySelectorAll('button') || []).map(b => b.innerText.trim()),
    };
  });
  console.log('Drawer position info:', drawerPosition);
  if (!drawerPosition.justifyClass.includes('justify-end')) {
    throw new Error('Drawer should be positioned on the RIGHT (justify-end)!');
  }

  // Screenshot open drawer on right
  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'drawer_opened_on_right.png'),
    fullPage: false,
  });
  console.log('Saved drawer_opened_on_right.png');

  // Switch to Storage tab inside drawer
  console.log('Switching to Storage tab inside drawer...');
  await page.click('[data-testid="drawer-tab-storage"]');
  await new Promise(r => setTimeout(r, 300));

  const storageHeaderText = await page.evaluate(() => {
    return document.querySelector('[role="dialog"], .bg-slate-900')?.innerText.slice(0, 100);
  });
  console.log('Storage tab content loaded:', storageHeaderText?.includes('Storage'));

  // Switch back to Architecture tab
  console.log('Switching back to Architecture tab...');
  await page.click('[data-testid="drawer-tab-architecture"]');
  await new Promise(r => setTimeout(r, 300));

  // Toggle Auto-Fit to DISABLED
  console.log('Toggling Auto-Fit to DISABLED...');
  await page.click('[data-testid="toggle-autofit"]');
  await new Promise(r => setTimeout(r, 200));

  const autoFitToggleText = await page.evaluate(() => {
    return document.querySelector('[data-testid="toggle-autofit"]')?.innerText;
  });
  console.log('Auto-Fit toggle text now:', autoFitToggleText);

  // Close Drawer
  await page.click('[data-testid="btn-close-drawer"]');
  await new Promise(r => setTimeout(r, 400));

  // 5. Verify Auto-Fit Button [F] is NOW VISIBLE when Auto-Fit is DISABLED
  console.log('\n--- 5. Checking Auto-Fit Button visibility when OFF ---');
  const autoFitBtnWhenDisabled = await page.$('[data-testid="btn-autofit"]');
  console.log('Is manual auto-fit button [F] visible when auto-fit is OFF?:', autoFitBtnWhenDisabled !== null ? 'YES (CORRECT)' : 'NO (WRONG)');
  if (autoFitBtnWhenDisabled === null) {
    throw new Error('Manual autofit button [F] MUST be visible when autofit is disabled!');
  }

  // Clear canvas
  await page.click('[data-testid="btn-clear"]');
  await new Promise(r => setTimeout(r, 300));

  // Draw a stroke with autofit disabled
  console.log('Drawing stroke with autofit disabled (should keep raw drawing)...');
  const canvasBox = await canvas.boundingBox();
  const rawX = canvasBox.x + canvasBox.width * 0.2;
  const rawY = canvasBox.y + canvasBox.height * 0.2;
  await page.mouse.move(rawX, rawY);
  await page.mouse.down();
  await page.mouse.move(rawX + 20, rawY + 30, { steps: 5 });
  await page.mouse.up();
  await new Promise(r => setTimeout(r, 300));

  // Click manual [F] button to auto-fit manually
  console.log('Clicking manual [F] button...');
  await page.click('[data-testid="btn-autofit"]');
  await new Promise(r => setTimeout(r, 400));

  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'autofit_manual_button_used.png'),
    fullPage: false,
  });
  console.log('Saved autofit_manual_button_used.png');

  await browser.close();
  console.log('\n🎉 All tests passed successfully!');
}

verify().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
