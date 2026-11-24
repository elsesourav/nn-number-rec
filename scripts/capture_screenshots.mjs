import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const OUTPUT_DIR = path.resolve(process.cwd(), 'screenshots');

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

const VIEWPORTS = [
  { name: 'desktop', width: 1280, height: 800 },
  { name: 'mobile_650', width: 650, height: 850 },
  { name: 'mobile_350', width: 350, height: 750 },
];

async function capture() {
  console.log('Launching Chrome for Full Screen UI audit...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();

  for (const vp of VIEWPORTS) {
    console.log(`\nTesting viewport: ${vp.name} (${vp.width}x${vp.height})...`);
    await page.setViewport({
      width: vp.width,
      height: vp.height,
      deviceScaleFactor: 2,
    });

    await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });

    // Wait for WASM engine
    try {
      await page.waitForFunction(
        () => document.body.innerText.includes('SIMD'),
        { timeout: 3000 }
      );
    } catch (e) {
      // safe fallback
    }

    // 1. Initial Fullscreen State (Visualizer takes maximum width & height!)
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `${vp.name}_fullscreen_main.png`),
      fullPage: false,
    });
    console.log(`Saved: ${vp.name}_fullscreen_main.png`);

    // 2. Open Architecture Slide-Over Drawer
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid="btn-architecture"]') ||
                  Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Architecture') || b.title?.includes('Architecture'));
      btn?.click();
    });
    await new Promise(r => setTimeout(r, 400));
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `${vp.name}_fullscreen_drawer.png`),
      fullPage: false,
    });
    console.log(`Saved: ${vp.name}_fullscreen_drawer.png`);

    // 2b. Switch to Storage tab in Drawer
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid="drawer-tab-storage"]') ||
                  Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Storage'));
      btn?.click();
    });
    await new Promise(r => setTimeout(r, 400));
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `${vp.name}_fullscreen_storage.png`),
      fullPage: false,
    });
    console.log(`Saved: ${vp.name}_fullscreen_storage.png`);

    // Close drawer
    await page.evaluate(() => {
      const closeBtn = document.querySelector('[data-testid="btn-close-drawer"]') ||
                       Array.from(document.querySelectorAll('button')).find(b => b.title === 'Close Drawer');
      closeBtn?.click();
    });
    await new Promise(r => setTimeout(r, 300));

    // 3. Draw a digit 3 on the drawing canvas
    const canvas = await page.$('canvas');
    if (canvas) {
      const box = await canvas.boundingBox();
      if (box) {
        const startX = box.x + box.width * 0.35;
        const startY = box.y + box.height * 0.25;
        await page.mouse.move(startX, startY);
        await page.mouse.down();
        // Draw number 3
        await page.mouse.move(startX + 50, startY, { steps: 5 });
        await page.mouse.move(startX + 15, startY + 40, { steps: 5 });
        await page.mouse.move(startX + 50, startY + 75, { steps: 5 });
        await page.mouse.move(startX, startY + 85, { steps: 5 });
        await page.mouse.up();
        await new Promise(r => setTimeout(r, 600));

        await page.screenshot({
          path: path.join(OUTPUT_DIR, `${vp.name}_fullscreen_with_drawing.png`),
          fullPage: false,
        });
        console.log(`Saved: ${vp.name}_fullscreen_with_drawing.png`);
      }
    }

    // 4. Test Live Training with Online MNIST images & Screenshot
    console.log(`Testing Online MNIST Live Auto-Train on ${vp.name}...`);
    // Click Step first to immediately display online sample #1 on canvas
    await page.click('[data-testid="btn-step-train"]');
    await new Promise(r => setTimeout(r, 400));
    // Start Live Train
    await page.click('[data-testid="btn-toggle-live-train"]');
    await new Promise(r => setTimeout(r, 1200));

    await page.screenshot({
      path: path.join(OUTPUT_DIR, `${vp.name}_fullscreen_live_train.png`),
      fullPage: false,
    });
    console.log(`Saved: ${vp.name}_fullscreen_live_train.png`);

    // Pause Live Train
    await page.click('[data-testid="btn-toggle-live-train"]');
    await new Promise(r => setTimeout(r, 200));
  }

  await browser.close();
  console.log('\nAll full screen screenshots captured in screenshots/ directory!');
}

capture().catch(err => {
  console.error(err);
  process.exit(1);
});
