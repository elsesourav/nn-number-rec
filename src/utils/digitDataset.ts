/**
 * Procedural Handwritten Digit Generator
 * Generates naturalistic, varied strokes for digits 0-9 on a canvas.
 * Useful for automated training with live visual feedback.
 */

export function renderSyntheticDigit(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  digit: number,
  strokeWidth: number = 36,
  jitterLevel: number = 0.7
): void {
  ctx.save();

  // Clear background with black
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, width, height);

  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#ffffff';

  const rnd = (min: number, max: number) => min + Math.random() * (max - min) * jitterLevel;
  const cx = width / 2 + rnd(-8, 8);
  const cy = height / 2 + rnd(-8, 8);
  const scale = rnd(0.72, 0.88);
  const rw = (width * 0.32) * scale;
  const rh = (height * 0.42) * scale;

  // Use dynamic stroke width controlled by the stroke slider
  const baseStroke = strokeWidth > 0 ? strokeWidth : 36;
  const stroke = baseStroke * rnd(0.9, 1.1);

  ctx.lineWidth = stroke;

  // Apply subtle rotation
  const angle = (rnd(-12, 12) * Math.PI) / 180;
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  ctx.translate(-cx, -cy);

  ctx.beginPath();

  switch (digit) {
    case 0: {
      // Oval with slight slant
      const topX = cx + rnd(-3, 3);
      const topY = cy - rh;
      ctx.ellipse(cx, cy, rw * rnd(0.85, 1.05), rh * rnd(0.9, 1.05), rnd(-0.1, 0.1), 0, Math.PI * 2);
      ctx.stroke();
      break;
    }

    case 1: {
      // Main vertical stroke with optional top slant
      const x = cx + rnd(-5, 5);
      const y1 = cy - rh;
      const y2 = cy + rh;
      if (Math.random() > 0.4) {
        ctx.moveTo(x - rw * 0.4, y1 + rh * 0.25);
        ctx.lineTo(x, y1);
      } else {
        ctx.moveTo(x, y1);
      }
      ctx.lineTo(x + rnd(-4, 4), y2);
      ctx.stroke();
      break;
    }

    case 2: {
      // Top curve -> diagonal down-left -> bottom baseline
      const x1 = cx - rw * 0.7;
      const y1 = cy - rh * 0.6;
      ctx.moveTo(x1, y1);
      ctx.bezierCurveTo(
        cx - rw * 0.6,
        cy - rh * 1.1,
        cx + rw * 0.9,
        cy - rh * 0.8,
        cx + rw * 0.5,
        cy - rh * 0.2
      );
      ctx.lineTo(cx - rw * 0.7 + rnd(-3, 3), cy + rh);
      ctx.lineTo(cx + rw * 0.8 + rnd(-3, 3), cy + rh + rnd(-2, 2));
      ctx.stroke();
      break;
    }

    case 3: {
      // Double arc
      const left = cx - rw * 0.7;
      const right = cx + rw * 0.8;
      ctx.moveTo(left, cy - rh * 0.9);
      ctx.bezierCurveTo(
        cx,
        cy - rh * 1.05,
        right,
        cy - rh * 0.7,
        cx + rnd(-2, 4),
        cy - rh * 0.05
      );
      ctx.bezierCurveTo(
        right + rnd(0, 5),
        cy + rh * 0.2,
        cx,
        cy + rh * 1.05,
        left,
        cy + rh * 0.85
      );
      ctx.stroke();
      break;
    }

    case 4: {
      // Down-left diagonal -> horizontal bar -> vertical stem
      const xStem = cx + rw * 0.35 + rnd(-4, 4);
      const yCross = cy + rh * 0.25;
      ctx.moveTo(cx + rw * 0.1, cy - rh);
      ctx.lineTo(cx - rw * 0.8, yCross);
      ctx.lineTo(cx + rw * 0.8, yCross);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(xStem, cy - rh * 0.6);
      ctx.lineTo(xStem + rnd(-2, 2), cy + rh);
      ctx.stroke();
      break;
    }

    case 5: {
      // Top bar -> vertical down -> bottom open loop
      ctx.moveTo(cx + rw * 0.7, cy - rh);
      ctx.lineTo(cx - rw * 0.6, cy - rh);
      ctx.lineTo(cx - rw * 0.7, cy - rh * 0.1);
      ctx.bezierCurveTo(
        cx + rw * 0.9,
        cy - rh * 0.2,
        cx + rw * 0.9,
        cy + rh * 0.9,
        cx - rw * 0.6,
        cy + rh * 0.85
      );
      ctx.stroke();
      break;
    }

    case 6: {
      // Downward curve -> bottom circle
      ctx.moveTo(cx + rw * 0.5, cy - rh);
      ctx.bezierCurveTo(
        cx - rw * 0.8,
        cy - rh * 0.5,
        cx - rw * 0.9,
        cy + rh * 0.4,
        cx - rw * 0.2,
        cy + rh * 0.85
      );
      ctx.bezierCurveTo(
        cx + rw * 0.8,
        cy + rh * 0.9,
        cx + rw * 0.8,
        cy + rh * 0.1,
        cx - rw * 0.3,
        cy + rh * 0.1
      );
      ctx.stroke();
      break;
    }

    case 7: {
      // Top bar -> diagonal down-left
      ctx.moveTo(cx - rw * 0.8, cy - rh);
      ctx.lineTo(cx + rw * 0.8, cy - rh);
      ctx.lineTo(cx - rw * 0.3 + rnd(-4, 4), cy + rh);
      if (Math.random() > 0.5) {
        // Crossbar
        ctx.moveTo(cx - rw * 0.2, cy);
        ctx.lineTo(cx + rw * 0.4, cy);
      }
      ctx.stroke();
      break;
    }

    case 8: {
      // Figure 8 (two connected loops)
      const rTop = rh * 0.45;
      const rBot = rh * 0.55;
      ctx.arc(cx, cy - rTop, rw * 0.75, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx, cy + rBot * 0.6, rw * 0.9, 0, Math.PI * 2);
      ctx.stroke();
      break;
    }

    case 9: {
      // Top loop -> downward right stroke
      const rTop = rh * 0.45;
      ctx.arc(cx, cy - rTop * 0.7, rw * 0.75, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx + rw * 0.75, cy - rTop * 0.7);
      ctx.lineTo(cx + rw * 0.6, cy + rh * 0.6);
      ctx.bezierCurveTo(
        cx + rw * 0.5,
        cy + rh * 0.9,
        cx - rw * 0.2,
        cy + rh * 0.95,
        cx - rw * 0.5,
        cy + rh * 0.8
      );
      ctx.stroke();
      break;
    }
  }

  ctx.restore();
}
