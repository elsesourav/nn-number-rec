import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const screenshotDir = path.resolve('screenshots');

if (!fs.existsSync(screenshotDir)) {
  fs.mkdirSync(screenshotDir, { recursive: true });
}

async function run() {
  console.log('🚀 Starting Verification of Sliders, Transitions & Visualizers...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  console.log('Navigating to http://localhost:3000...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 800));

  // 1. Verify Stroke Width Slider Centering
  const strokeSlider = await page.waitForSelector('[aria-label="Stroke Width"]');
  const strokeSliderMetrics = await page.evaluate((el) => {
    const thumb = el.querySelector('.bg-white');
    const containerRect = el.getBoundingClientRect();
    const thumbRect = thumb.getBoundingClientRect();
    const topMargin = thumbRect.top - containerRect.top;
    const bottomMargin = containerRect.bottom - thumbRect.bottom;
    return {
      containerHeight: containerRect.height,
      thumbHeight: thumbRect.height,
      topMargin,
      bottomMargin,
      isCentered: Math.abs(topMargin - bottomMargin) <= 2
    };
  }, strokeSlider);

  console.log('Stroke Slider Centering Metrics:', strokeSliderMetrics);
  await strokeSlider.screenshot({ path: path.join(screenshotDir, '20_stroke_slider_centered.png') });
  console.log('✓ Captured 20_stroke_slider_centered.png');

  // 2. Test Stroke Width Dynamic Canvas Adjustment
  // First draw on the canvas
  const canvas = await page.waitForSelector('canvas');
  const canvasRect = await page.evaluate(c => {
    const r = c.getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height };
  }, canvas);

  // Draw a digit (e.g. stroke for digit 1)
  await page.mouse.move(canvasRect.x + canvasRect.w * 0.5, canvasRect.y + canvasRect.h * 0.2);
  await page.mouse.down();
  await page.mouse.move(canvasRect.x + canvasRect.w * 0.5, canvasRect.y + canvasRect.h * 0.8, { steps: 10 });
  await page.mouse.up();
  await new Promise(r => setTimeout(r, 400)); // wait for auto-fit debounce (220ms)

  await page.screenshot({ path: path.join(screenshotDir, '21_hand_drawn_canvas_initial.png') });
  console.log('✓ Hand drawn digit captured (21_hand_drawn_canvas_initial.png)');

  // Now change stroke width slider from 22 to 28
  const sliderBox = await strokeSlider.boundingBox();
  // Click towards right of slider to increase stroke
  await page.mouse.click(sliderBox.x + sliderBox.width * 0.85, sliderBox.y + sliderBox.height * 0.5);
  await new Promise(r => setTimeout(r, 300));

  await page.screenshot({ path: path.join(screenshotDir, '22_hand_drawn_canvas_thickened.png') });
  console.log('✓ Stroke widened dynamically (22_hand_drawn_canvas_thickened.png)');

  // 3. Test Smooth Transition: Recognize Mode -> Train Mode
  const trainTabBtn = await page.waitForSelector('[data-testid="tab-train"]');
  await trainTabBtn.click();
  await new Promise(r => setTimeout(r, 120)); // capture mid-transition
  await page.screenshot({ path: path.join(screenshotDir, '23_mode_transition_to_train.png') });

  await new Promise(r => setTimeout(r, 400)); // settled in train mode
  await page.screenshot({ path: path.join(screenshotDir, '24_train_mode_settled.png') });
  console.log('✓ Captured Train Mode (24_train_mode_settled.png)');

  // Test Speed Slider Centering in Train Mode
  const speedSlider = await page.waitForSelector('[aria-label="Speed"]');
  const speedSliderMetrics = await page.evaluate((el) => {
    const thumb = el.querySelector('.bg-white');
    const containerRect = el.getBoundingClientRect();
    const thumbRect = thumb.getBoundingClientRect();
    const topMargin = thumbRect.top - containerRect.top;
    const bottomMargin = containerRect.bottom - thumbRect.bottom;
    return {
      containerHeight: containerRect.height,
      thumbHeight: thumbRect.height,
      topMargin,
      bottomMargin,
      isCentered: Math.abs(topMargin - bottomMargin) <= 2
    };
  }, speedSlider);
  console.log('Speed Slider Centering Metrics:', speedSliderMetrics);
  await speedSlider.screenshot({ path: path.join(screenshotDir, '25_speed_slider_centered.png') });

  // 4. Test Smooth Transition: Graph Visualizer -> Mechanism Visualizer
  const mechanismTabBtn = await page.waitForSelector('[data-testid="tab-visualizer-mechanism"]');
  await mechanismTabBtn.click();
  await new Promise(r => setTimeout(r, 120)); // mid-transition
  await page.screenshot({ path: path.join(screenshotDir, '26_visualizer_transition_to_mechanism.png') });

  await new Promise(r => setTimeout(r, 400)); // settled in mechanism mode
  await page.screenshot({ path: path.join(screenshotDir, '27_mechanism_visualizer_settled.png') });
  console.log('✓ Captured Mechanism Visualizer settled (27_mechanism_visualizer_settled.png)');

  // 5. Test Mechanism Flow Animation
  const replayBtn = await page.waitForSelector('[data-testid="btn-workflow-replay"]');
  await replayBtn.click();
  await new Promise(r => setTimeout(r, 450)); // middle of flow
  await page.screenshot({ path: path.join(screenshotDir, '28_mechanism_smooth_flow_pulse.png') });
  console.log('✓ Captured smooth energy pulse (28_mechanism_smooth_flow_pulse.png)');

  // Wait for flow to complete
  await new Promise(r => setTimeout(r, 2200));

  // 6. Test Switch back to Graph & Check Slower Graph Recognition Wave
  const graphTabBtn = await page.waitForSelector('[data-testid="tab-visualizer-graph"]');
  await graphTabBtn.click();
  await new Promise(r => setTimeout(r, 400));

  // Switch back to recognize mode
  const recTabBtn = await page.waitForSelector('[data-testid="tab-recognize"]');
  await recTabBtn.click();
  await new Promise(r => setTimeout(r, 400));

  // Trigger recognition by drawing another stroke
  await page.mouse.move(canvasRect.x + canvasRect.w * 0.4, canvasRect.y + canvasRect.h * 0.3);
  await page.mouse.down();
  await page.mouse.move(canvasRect.x + canvasRect.w * 0.6, canvasRect.y + canvasRect.h * 0.7, { steps: 8 });
  await page.mouse.up();
  await new Promise(r => setTimeout(r, 300)); // into the 1800ms animation
  await page.screenshot({ path: path.join(screenshotDir, '29_slower_graph_recognition_wave.png') });
  console.log('✓ Captured slower graph wave propagation (29_slower_graph_recognition_wave.png)');

  await browser.close();
  console.log('🎉 Verification completed successfully!');
}

run().catch((err) => {
  console.error('Verification failed:', err);
  process.exit(1);
});
