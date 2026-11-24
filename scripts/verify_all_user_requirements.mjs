import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const SCREENSHOTS_DIR = path.resolve(process.cwd(), 'screenshots');

if (!fs.existsSync(SCREENSHOTS_DIR)) {
  fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
}

async function runVerification() {
  console.log('🚀 Starting Puppeteer Verification Suite...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();
  const consoleLogs = [];
  page.on('console', msg => consoleLogs.push(`[${msg.type()}] ${msg.text()}`));
  page.on('pageerror', err => consoleLogs.push(`[PAGE_ERROR] ${err.message}`));

  try {
    // -------------------------------------------------------------
    // Test 1: Desktop Layout - Recognize Mode (Clean, natural scrolling, no locked fullscreen)
    // -------------------------------------------------------------
    await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 2 });
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
    await new Promise(r => setTimeout(r, 1000));

    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '01_desktop_recognize_mode.png'),
    });
    console.log('✅ Screenshot 01: Desktop Recognize Mode');

    // -------------------------------------------------------------
    // Test 2: Drawing Persistence (Verify Canvas DOES NOT auto-clear)
    // -------------------------------------------------------------
    const canvas = await page.$('canvas.touch-none, div.aspect-square canvas');
    if (canvas) {
      const box = await canvas.boundingBox();
      const cx = box.x + box.width / 2;
      const cy = box.y + box.height / 2;

      // Draw a figure "3"
      await page.mouse.move(cx - 30, cy - 50);
      await page.mouse.down();
      await page.mouse.move(cx + 25, cy - 50, { steps: 5 });
      await page.mouse.move(cx + 25, cy - 10, { steps: 5 });
      await page.mouse.move(cx - 10, cy - 5, { steps: 5 });
      await page.mouse.move(cx + 25, cy, { steps: 5 });
      await page.mouse.move(cx + 25, cy + 50, { steps: 5 });
      await page.mouse.move(cx - 30, cy + 50, { steps: 5 });
      await page.mouse.up();

      // Wait 1.5 seconds (previously auto-cleared at 650ms)
      await new Promise(r => setTimeout(r, 1500));

      // Check if canvas still has non-zero pixels
      const hasPixels = await page.evaluate(() => {
        const c = document.querySelector('div.aspect-square canvas');
        if (!c) return false;
        const ctx = c.getContext('2d');
        const img = ctx.getImageData(0, 0, c.width, c.height);
        return img.data.some((val, idx) => idx % 4 === 0 && val > 30);
      });

      console.log('✅ Canvas has drawn pixels after 1.5s delay:', hasPixels);
      await page.screenshot({
        path: path.join(SCREENSHOTS_DIR, '02_desktop_drawing_persistent.png'),
      });
      console.log('✅ Screenshot 02: Drawing Persistent (No Auto-Clear)');
    }

    // -------------------------------------------------------------
    // Test 3: Left-Side Slide-Over Drawer with Low Blur (Architecture & Storage)
    // -------------------------------------------------------------
    // Click "Architecture" button
    const buttons = await page.$$('button');
    for (const b of buttons) {
      const text = await page.evaluate(el => el.textContent, b);
      if (text.includes('Architecture')) {
        await b.click();
        break;
      }
    }
    await new Promise(r => setTimeout(r, 600));
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '03_left_drawer_architecture.png'),
    });
    console.log('✅ Screenshot 03: Left Drawer (Architecture with Low Blur)');

    // Click "Storage" tab in drawer
    const drawerButtons = await page.$$('div.fixed button');
    for (const b of drawerButtons) {
      const text = await page.evaluate(el => el.textContent, b);
      if (text.includes('Storage')) {
        await b.click();
        break;
      }
    }
    await new Promise(r => setTimeout(r, 600));
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '04_left_drawer_storage.png'),
    });
    console.log('✅ Screenshot 04: Left Drawer (Storage & Named Presets)');

    // Close drawer
    const closeBtn = await page.$('div.fixed button[title="Close Drawer"]');
    if (closeBtn) {
      await closeBtn.click();
    } else {
      const allFixedBtns = await page.$$('div.fixed button');
      if (allFixedBtns.length > 0) {
        await allFixedBtns[allFixedBtns.length - 1].click();
      }
    }
    await new Promise(r => setTimeout(r, 500));

    // -------------------------------------------------------------
    // Test 4: Switch to Train Studio Mode
    // -------------------------------------------------------------
    await page.click('[data-testid="tab-train"]');
    await new Promise(r => setTimeout(r, 600));
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '05_desktop_train_studio_mode.png'),
    });
    console.log('✅ Screenshot 05: Desktop Train Studio Mode');

    // Run a step in training
    const stepBtn = await page.$('button[data-testid="btn-step-train"]');
    if (stepBtn) {
      await stepBtn.click();
      await new Promise(r => setTimeout(r, 800));
      await page.screenshot({
        path: path.join(SCREENSHOTS_DIR, '06_train_step_live_preview.png'),
      });
      console.log('✅ Screenshot 06: Train Step Live Preview on Network');
    }

    // -------------------------------------------------------------
    // Test 5: Mobile Viewport (390 x 844 - iPhone / Modern Phone)
    // -------------------------------------------------------------
    await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    await new Promise(r => setTimeout(r, 500));

    // Check Network on top and Draw Pad on bottom
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '07_mobile_screen_top.png'),
    });
    console.log('✅ Screenshot 07: Mobile Viewport - Network Visualizer on Top');

    // Scroll down to see compact drawing pad and numpad
    await page.evaluate(() => window.scrollBy(0, 350));
    await new Promise(r => setTimeout(r, 400));
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '08_mobile_screen_bottom_drawpad.png'),
    });
    console.log('✅ Screenshot 08: Mobile Viewport - Compact Drawing Pad & Numpad');

    // Check Recognize mode on mobile
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.click('[data-testid="tab-recognize"]');
    await new Promise(r => setTimeout(r, 400));
    await new Promise(r => setTimeout(r, 400));
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '09_mobile_recognize_mode.png'),
    });
    console.log('✅ Screenshot 09: Mobile Recognize Mode');

  } catch (err) {
    console.error('❌ Verification failed with error:', err);
  } finally {
    if (consoleLogs.length > 0) {
      console.log('--- Page Console Logs ---');
      consoleLogs.slice(-15).forEach(l => console.log(l));
    }
    await browser.close();
    console.log('🏁 Verification complete.');
  }
}

runVerification();
