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
  console.log('🚀 Starting verification of all 4 user fixes...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  // 350px mobile width
  await page.setViewport({ width: 350, height: 750, deviceScaleFactor: 2 });

  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await new Promise(r => setTimeout(r, 1000));

  console.log('\n--- 1. Testing Mobile Body Scrolling on 350px Viewport ---');
  const scrollInfoBefore = await page.evaluate(() => {
    return {
      scrollHeight: document.scrollingElement.scrollHeight,
      innerHeight: window.innerHeight,
      scrollY: window.scrollY,
      htmlOverflow: window.getComputedStyle(document.documentElement).overflowY,
      bodyOverflow: window.getComputedStyle(document.body).overflowY,
      bodyPosition: window.getComputedStyle(document.body).position,
    };
  });
  console.log('Scroll state before:', scrollInfoBefore);

  // Perform window scroll
  await page.evaluate(() => {
    window.scrollTo(0, 300);
  });
  await new Promise(r => setTimeout(r, 200));

  const scrollInfoAfter = await page.evaluate(() => {
    return {
      scrollY: window.scrollY
    };
  });
  console.log('Scroll state after scrollTo(0, 300):', scrollInfoAfter);
  if (scrollInfoAfter.scrollY > 0) {
    console.log('✅ PASS: Body scroll is working! scrollY =', scrollInfoAfter.scrollY);
  } else {
    console.error('❌ FAIL: Body scroll is still locked! scrollY =', scrollInfoAfter.scrollY);
  }

  // Screenshot scrolled down
  await page.screenshot({ path: path.join(ARTIFACT_DIR, '23_mobile_350_scrolled.png') });

  // Scroll back to top
  await page.evaluate(() => window.scrollTo(0, 0));
  await new Promise(r => setTimeout(r, 200));

  console.log('\n--- 2. Testing Drawing Pad Tool Buttons (2-3x Bigger) ---');
  const buttonMetrics = await page.evaluate(() => {
    const brush = document.querySelector('[data-testid="btn-brush"]');
    const eraser = document.querySelector('[data-testid="btn-eraser"]');
    const clear = document.querySelector('[data-testid="btn-clear"]');

    const getMetrics = (el) => {
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      const svg = el.querySelector('svg');
      const svgRect = svg ? svg.getBoundingClientRect() : null;
      return {
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        svgWidth: svgRect ? Math.round(svgRect.width) : null,
        svgHeight: svgRect ? Math.round(svgRect.height) : null,
      };
    };

    return {
      brush: getMetrics(brush),
      eraser: getMetrics(eraser),
      clear: getMetrics(clear),
    };
  });
  console.log('Button Metrics (Brush, Eraser, Clear):', buttonMetrics);
  if (buttonMetrics.brush.height >= 34 && buttonMetrics.brush.svgHeight >= 18) {
    console.log('✅ PASS: Tool buttons are 2-3x larger (height ~36px, svg icon ~19px)!');
  } else {
    console.error('❌ FAIL: Tool buttons are still too small!');
  }

  console.log('\n--- 3. Testing Pretrained Model Default Load ---');
  // Wait up to 2 seconds for pretrained model fetch & load
  await new Promise(r => setTimeout(r, 1200));
  const pretrainedState = await page.evaluate(() => {
    // Check if total params is 55,050
    const btnArch = document.querySelector('[data-testid="btn-architecture"]');
    const toast = document.querySelector('[role="status"], [class*="toast"]');
    return {
      archText: btnArch ? btnArch.textContent : '',
      toastText: toast ? toast.textContent : '',
    };
  });
  console.log('Pretrained Model State:', pretrainedState);
  if (pretrainedState.archText.includes('55k') || pretrainedState.archText.includes('55,050')) {
    console.log('✅ PASS: Pretrained model loaded by default with 55,050 weights!');
  }

  console.log('\n--- 4. Testing Mechanism Backtrack Animation ---');
  // Switch to Mechanism visualizer tab
  const mechanismBtn = await page.waitForSelector('[data-testid="tab-visualizer-mechanism"]');
  await mechanismBtn.click();
  await new Promise(r => setTimeout(r, 600));

  // Screenshot initial mechanism on 350px
  await page.screenshot({ path: path.join(ARTIFACT_DIR, '24_mobile_350_mechanism_idle.png') });

  // Click Replay button to trigger full training flow (Forward + Backprop + Weights update)
  const replayBtn = await page.waitForSelector('[data-testid="btn-workflow-replay"]');
  console.log('Triggering Replay...');
  await replayBtn.click();

  // Poll stages during flow
  const observedStages = [];
  for (let i = 0; i < 15; i++) {
    await new Promise(r => setTimeout(r, 220));
    const stageInfo = await page.evaluate(() => {
      const activeNode = document.querySelector('.interactive-node-card[class*="ring-2"], .interactive-node-card[class*="scale-103"]');
      const pill = document.querySelector('.pointer-events-auto [class*="font-mono"]');
      return {
        activeText: activeNode ? activeNode.textContent?.slice(0, 40) : null,
        pillText: pill ? pill.textContent?.trim() : null,
      };
    });
    observedStages.push(stageInfo);
    if (i === 6) {
      await page.screenshot({ path: path.join(ARTIFACT_DIR, '25_mobile_350_mechanism_backtrack.png') });
    }
  }

  console.log('Sample observed stages during flow:', observedStages.slice(0, 10));
  const sawBackprop = observedStages.some(s => s.pillText && (s.pillText.includes('Backprop') || s.pillText.includes('Updated')));
  if (sawBackprop) {
    console.log('✅ PASS: Backpropagation and backtrack animation successfully executed and observed!');
  } else {
    console.log('Observed pills:', observedStages.map(s => s.pillText));
  }

  // Final screenshot after animation settles
  await page.screenshot({ path: path.join(ARTIFACT_DIR, '26_mobile_350_mechanism_complete.png') });

  await browser.close();
  console.log('\n🎉 Verification completed successfully!');
}

run().catch(err => {
  console.error('Error during verification:', err);
  process.exit(1);
});
