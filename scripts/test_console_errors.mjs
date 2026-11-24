import puppeteer from 'puppeteer-core';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

async function testConsole() {
  console.log('Testing app in headless browser for shortcuts, live training, and exceptions...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();
  const errors = [];
  const warnings = [];

  page.on('pageerror', err => {
    console.error('PAGE ERROR:', err.message);
    errors.push(err.message);
  });

  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.error('CONSOLE ERROR:', msg.text());
      errors.push(msg.text());
    } else if (msg.type() === 'warn') {
      warnings.push(msg.text());
    }
  });

  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });

  // Wait for WASM to mount and initialize
  await new Promise(r => setTimeout(r, 1200));

  // 1. Test Keyboard shortcuts: Brush (B), Eraser (E), Clear (C), Auto-Fit (F)
  console.log('Testing keyboard shortcuts B, E, C, F...');
  await page.keyboard.press('KeyE');
  await new Promise(r => setTimeout(r, 100));
  await page.keyboard.press('KeyB');
  await new Promise(r => setTimeout(r, 100));
  await page.keyboard.press('KeyF');
  await new Promise(r => setTimeout(r, 100));
  await page.keyboard.press('KeyC');
  await new Promise(r => setTimeout(r, 100));

  // 1b. Test Auto-Fit button
  console.log('Testing Auto-Fit button...');
  await page.click('[data-testid="btn-autofit"]');
  await new Promise(r => setTimeout(r, 200));

  // 2. Test Key train digit: press '7'
  console.log('Testing digit shortcut training (Key 7)...');
  await page.keyboard.press('Digit7');
  await new Promise(r => setTimeout(r, 300));

  // 3. Test Clicking Big Numpad button '3'
  console.log('Testing Big Numpad click for digit 3...');
  await page.click('[data-testid="btn-train-digit-3"]');
  await new Promise(r => setTimeout(r, 300));

  // 4. Test Live Training Toggle (Play / Pause)
  console.log('Testing Live Auto-Train start...');
  await page.click('[data-testid="btn-toggle-live-train"]');
  // Let it train 5 samples smoothly
  await new Promise(r => setTimeout(r, 1200));

  console.log('Testing Live Auto-Train pause...');
  await page.click('[data-testid="btn-toggle-live-train"]');
  await new Promise(r => setTimeout(r, 300));

  // 5. Test Step training
  console.log('Testing Step train...');
  await page.click('[data-testid="btn-step-train"]');
  await new Promise(r => setTimeout(r, 300));

  // 6. Test Spacebar shortcut for live training
  console.log('Testing Spacebar shortcut for live train...');
  await page.keyboard.press('Space');
  await new Promise(r => setTimeout(r, 600));
  await page.keyboard.press('Space');
  await new Promise(r => setTimeout(r, 200));

  // 7. Test Architecture drawer
  console.log('Testing Architecture drawer...');
  await page.click('[data-testid="btn-architecture"]');
  await new Promise(r => setTimeout(r, 400));

  await page.click('[data-testid="btn-close-drawer"]');
  await new Promise(r => setTimeout(r, 300));

  await browser.close();

  console.log('\n--- TEST RESULTS ---');
  if (errors.length === 0) {
    console.log('✅ PASS: Zero console errors or BindingError exceptions detected!');
  } else {
    console.error(`❌ FAIL: Detected ${errors.length} errors:`, errors);
    process.exit(1);
  }
}

testConsole().catch(err => {
  console.error(err);
  process.exit(1);
});
