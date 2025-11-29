"use strict"

//use cssRoot.style.setProperty("key", "value");
const cssRoot = document.querySelector(':root');

// when run this app in mobile is return true
const isMobile = localStorage.mobile || window.navigator.maxTouchPoints > 1;

// minimum window size
const minSize = window.innerWidth > window.innerHeight ? window.innerHeight : window.innerWidth;


class Canvas {
   constructor(width = window.innerWidth, height = window.innerHeight, parent = document.body) {
      this.width = width;
      this.height = height;
      this.parent = parent;
      
      this.cvs = document.createElement("canvas");
      this.ctx = this.cvs.getContext("2d");
      this.cvs.width = this.width;
      this.cvs.height = this.height;
      this.parent.appendChild(this.cvs);

      this.CENTER = "center";
      this.LEFT = "left";
      this.RIGHT = "right";
      this.TOP = "top";
      this.BOTTOM = "bottom";
      this.BUTT = "butt";
      this.ROUND = "round";
      this.SQUARE = "square";
      this.DEFAULT = false;
      this._fontSize_ = 10;
      this._arcMode_ = "center";
      this._textMode_ = false;
      this._rectMode_ = false;
      this._ellipseMode_ = false;
   }

   on = (event, fun) => {
      this.cvs.addEventListener(event, fun);
   };

   textMode(mode = "center") { this._textMode_ = mode; };
   arcMode(mode = "center") { this._arcMode_ = mode; };
   rectMode(mode = "center") { this._rectMode_ = mode; };
   ellipseMode(mode = "center") { this._ellipseMode_ = mode; };

   clrScr() { this.ctx.clearRect(0, 0, this.width, this.height); };

   clearRect(x = 0, y = 0, w = this.width, h = this.height) {
      this.ctx.clearRect(x, y, w, h);
   };

   background(red, green, blue, alpha) {
      this.ctx.fillStyle = this.getColor(red, green, blue, alpha);
      this.ctx.fillRect(0, 0, this.width, this.height);
   };

   fill(red, green, blue, alpha) {
      this.ctx.fillStyle = this.getColor(red, green, blue, alpha);
      this.ctx.fill();
   };

   fillStyle(red, green, blue, alpha) {
      this.ctx.fillStyle = this.getColor(red, green, blue, alpha);
   };

   strokeStyle(red, green, blue, alpha) {
      this.ctx.strokeStyle = this.getColor(red, green, blue, alpha);
   };

   stroke(red, green, blue, alpha) {
      this.ctx.strokeStyle = this.getColor(red, green, blue, alpha);
      this.ctx.stroke();
   };

   globalAlpha(alpha) { this.ctx.globalAlpha = alpha; };

   strokeRect(x, y, w, h) {
      if (this._rectMode_ == "center") {
         this.ctx.strokeRect(x - w / 2, y - h / 2, w, h);
      } else if (this._rectMode_ == "left") {
         this.ctx.strokeRect(x - w, y - h / 2, w, h);
      } else if (this._rectMode_ == "right") {
         this.ctx.strokeRect(x, y - h / 2, w, h);
      } else if (this._rectMode_ == "top") {
         this.ctx.strokeRect(x - w / 2, y - h, w, h);
      } else if (this._rectMode_ == "bottom") {
         this.ctx.strokeRect(x - w / 2, y, w, h);
      } else {
         this.ctx.strokeRect(x, y, w, h);
      }
   };

   rect(x, y, w, h) {
      if (this._rectMode_ == "center") {
         this.ctx.rect(x - w / 2, y - h / 2, w, h);
      } else if (this._rectMode_ == "left") {
         this.ctx.rect(x - w, y - h / 2, w, h);
      } else if (this._rectMode_ == "right") {
         this.ctx.rect(x, y - h / 2, w, h);
      } else if (this._rectMode_ == "top") {
         this.ctx.rect(x - w / 2, y - h, w, h);
      } else if (this._rectMode_ == "bottom") {
         this.ctx.rect(x - w / 2, y, w, h);
      } else {
         this.ctx.rect(x, y, w, h);
      }
   };

   fillRect(x, y, w, h) {
      if (this._rectMode_ == "center") {
         this.ctx.fillRect(x - w / 2, y - h / 2, w, h);
      } else if (this._rectMode_ == "left") {
         this.ctx.fillRect(x - w, y - h / 2, w, h);
      } else if (this._rectMode_ == "right") {
         this.ctx.fillRect(x, y - h / 2, w, h);
      } else if (this._rectMode_ == "top") {
         this.ctx.fillRect(x - w / 2, y - h, w, h);
      } else if (this._rectMode_ == "bottom") {
         this.ctx.fillRect(x - w / 2, y, w, h);
      } else {
         this.ctx.fillRect(x, y, w, h);
      }
   };

   roundRect(x, y, w, h, radiusArray = [50, 20, 50, 20]) {
      this.ctx.beginPath();
      if (this._rectMode_ == "center") {
         this.ctx.roundRect(x - w / 2, y - h / 2, w, h, radiusArray);
      } else if (this._rectMode_ == "left") {
         this.ctx.roundRect(x - w, y - h / 2, w, h, radiusArray);
      } else if (this._rectMode_ == "right") {
         this.ctx.roundRect(x, y - h / 2, w, h, radiusArray);
      } else if (this._rectMode_ == "top") {
         this.ctx.roundRect(x - w / 2, y - h, w, h, radiusArray);
      } else if (this._rectMode_ == "bottom") {
         this.ctx.roundRect(x - w / 2, y, w, h, radiusArray);
      } else {
         this.ctx.roundRect(x, y, w, h, radiusArray);
      }
   };

   beginPath() { this.ctx.beginPath(); };

   closePath() { this.ctx.closePath(); };

   arc(x, y, r, startAngle = 0, endAngle = Math.PI * 2, counterclockwise = false) {
      this.ctx.beginPath();

      if (this._arcMode_ == "center") {
         this.ctx.arc(x, y, r, startAngle, endAngle, counterclockwise);
      } else if (this._arcMode_ == "left") {
         this.ctx.arc(x - r, y, r, startAngle, endAngle, counterclockwise);
      } else if (this._arcMode_ == "right") {
         this.ctx.arc(x + r, y, r, startAngle, endAngle, counterclockwise);
      } else if (this._arcMode_ == "top") {
         this.ctx.arc(x, y - r, r, startAngle, endAngle, counterclockwise);
      } else if (this._arcMode_ == "bottom") {
         this.ctx.arc(x, y + r, r, startAngle, endAngle, counterclockwise);
      } else {
         this.ctx.arc(x, y, r, startAngle, endAngle, counterclockwise);
      }
   };

   moveTo(x, y) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, y);
   };

   lineTo(x, y) { this.ctx.lineTo(x, y); };

   arcTo(x1, y1, x2, y2, radius = 50) {
      this.ctx.arcTo(x1, y1, x2, y2, radius);
   };

   lineWidth(width) { this.ctx.lineWidth = width; };

   lineCap(style) { this.ctx.lineCap = style; };

   lineJoin(style) { this.ctx.lineJoin = style; };

   getLineDash() { return this.ctx.getLineDash(); };

   setLineDash(array) { this.ctx.setLineDash(array); };

   lineDashOffset(offset) { this.ctx.lineDashOffset = offset; };

   drawImage(image, dx, dy) {
      this.ctx.drawImage(image, dx, dy);
   };
   drawImage(image, dx, dy, dWidth, dHeight) {
      this.ctx.drawImage(image, dx, dy, dWidth, dHeight);
   };
   drawImage(image, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight) {
      this.ctx.drawImage(image, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight);
   };

   ellipse(x, y, rediusX, rediusY, rotation = Math.PI / 4, startAngle = 0, endAngle = Math.PI * 2, counterclockwise = false) {
      this.ctx.beginPath();

      if (this._ellipseMode_ == "center") {
         this.ctx.ellipse(x, y, rediusX, rediusY, rotation, startAngle, endAngle, counterclockwise);
      } else if (this._ellipseMode_ == "left") {
         this.ctx.ellipse(x - rediusX, y, rediusX, rediusY, rotation, startAngle, endAngle, counterclockwise);
      } else if (this._ellipseMode_ == "right") {
         this.ctx.ellipse(x + rediusX, y, rediusX, rediusY, rotation, startAngle, endAngle, counterclockwise);
      } else if (this._ellipseMode_ == "top") {
         this.ctx.ellipse(x, y - rediusY, rediusX, rediusY, rotation, startAngle, endAngle, counterclockwise);
      } else if (this._ellipseMode_ == "bottom") {
         this.ctx.ellipse(x, y + rediusY, rediusX, rediusY, rotation, startAngle, endAngle, counterclockwise);
      } else {
         this.ctx.ellipse(x, y, rediusX, rediusY, rotation, startAngle, endAngle, counterclockwise);
      }
   };

   // squice first two sides 
   bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x, y) {
      this.ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x, y);
   };

   quadraticCurveTo(cpx, cpy, x, y) {
      this.ctx.beginPath();
      this.ctx.quadraticCurveTo(cpx, cpy, x, y);
   };

   createPattern(image, repetition = "repeat") {
      return this.ctx.createPattern(image, repetition);
   };
   reset() { this.ctx.reset(); };

   resetTransform() { this.ctx.resetTransform(); };

   restore() { this.ctx.restore(); };

   rotate(angle) { this.ctx.rotate(angle); };

   save() { this.ctx.save(); };

   clip() { this.ctx.clip(); };

   filter(filter) { this.ctx.filter = filter; };

   scale(x, y) { this.ctx.scale(x, y); };
   scaleX(x) { this.ctx.scale(x, 0); };
   scaleY(y) { this.ctx.scale(0, y); };

   translate(x, y) { this.ctx.translate(x, y); };
   translateX(x) { this.ctx.translate(x, 0); };
   translateY(y) { this.ctx.translate(0, y); };

   transform(scaleW = 1, skewingH = 2, skewingW = 2, scaleH = 1, moveW = 0, moveH = 0) {
      this.ctx.transform(scaleW, skewingH, skewingW, scaleH, moveW, moveH);
   };

   setTransform(scaleW = 1, skewingH = 2, skewingW = 2, scaleH = 1, moveW = 0, moveH = 0) {
      this.ctx.setTransform(scaleW, skewingH, skewingW, scaleH, moveW, moveH);
   };

   font(size, family = "sans-serif", style = "") {
      this._fontSize_ = size;
      this.ctx.font = `${style} ${size}px ${family}`;
   };

   textAlign(alignment) { this.ctx.textAlign = alignment; };

   fillText(text, x, y, maxWidth = undefined) {
      if (this._textMode_ == "center") {
         this.textAlign("center");
         this.ctx.fillText(text, x, y + this._fontSize_ / 4, maxWidth);
      } else if (this._textMode_ == "left") {
         this.textAlign("right");
         this.ctx.fillText(text, x, y + this._fontSize_ / 4, maxWidth);
      } else if (this._textMode_ == "right") {
         this.textAlign("left");
         this.ctx.fillText(text, x, y + this._fontSize_ / 4, maxWidth);
      } else if (this._textMode_ == "top") {
         this.textAlign("center");
         this.ctx.fillText(text, x, y - this._fontSize_ / 3.8, maxWidth);
      } else if (this._textMode_ == "bottom") {
         this.textAlign("center");
         this.ctx.fillText(text, x, y + this._fontSize_ / 1.3, maxWidth);
      } else {
         this.ctx.fillText(text, x, y, maxWidth);
      }
   };

   strokeText(text, x, y, maxWidth = undefined) {
      if (this._textMode_ == "center") {
         this.textAlign("center");
         this.ctx.strokeText(text, x, y + this._fontSize_ / 4, maxWidth);
      } else if (this._textMode_ == "left") {
         this.textAlign("right");
         this.ctx.strokeText(text, x, y + this._fontSize_ / 4, maxWidth);
      } else if (this._textMode_ == "right") {
         this.textAlign("left");
         this.ctx.strokeText(text, x, y + this._fontSize_ / 4, maxWidth);
      } else if (this._textMode_ == "top") {
         this.textAlign("center");
         this.ctx.strokeText(text, x, y - this._fontSize_ / 3.8, maxWidth);
      } else if (this._textMode_ == "bottom") {
         this.textAlign("center");
         this.ctx.strokeText(text, x, y + this._fontSize_ / 1.3, maxWidth);
      } else {
         this.ctx.strokeText(text, x, y, maxWidth);
      }
   };

   fontVariantCaps(smallCaps0__allSmallCaps1__petiteCaps2__allPetiteCaps3__unicase4__titlingCaps5 = 0) {
      this.ctx.fontVariantCaps = (['small-caps', 'all-small-caps', 'petite-caps', 'all-petite-caps', 'unicase', 'titling-caps'])[smallCaps0__allSmallCaps1__petiteCaps2__allPetiteCaps3__unicase4__titlingCaps5 > 5 ? 5 : smallCaps0__allSmallCaps1__petiteCaps2__allPetiteCaps3__unicase4__titlingCaps5];
   };

   letterSpacing(space) { this.ctx.letterSpacing = `${space}px`; };

   wordSpacing(space) { this.ctx.wordSpacing = `${space}px`; };

   measureText(text) { return this.ctx.measureText(text) };

   direction(ltr0__OR__rtl1 = 0) {
      this.ctx.direction = ltr0__OR__rtl1 === 0 ? "ltr" : "rtl";
   };

   shadowColor(red, green, blue, alpha) {
      this.ctx.shadowColor = this.getColor(red, green, blue, alpha);
   };

   shadowBlur(size) { this.ctx.shadowBlur = size; };

   shadowOffsetX(offsetX) { this.ctx.shadowOffsetX = offsetX; };
   shadowOffsetY(offsetY) { this.ctx.shadowOffsetY = offsetY; };
   shadowOffset(offsetX, offsetY) {
      this.ctx.shadowOffsetX = offsetX;
      this.ctx.shadowOffsetY = offsetY;
   };

   createImageData(sw, sh, settings = undefined) {
      return this.ctx.createImageData(sw, sh, settings);
   };

   getImageData(sx, sy, sw, sh) { return this.ctx.getImageData(sx, sy, sw, sh); };

   putImageData(image, dx, dy) { this.ctx.putImageData(image, dx, dy); };

   createLinearGradient(x1, y1, x2, y2) {
      return this.ctx.createLinearGradient(x1, y1, x2, y2);
   };

   createConicGradient(startAngle, x, y) {
      return this.ctx.createConicGradient(startAngle, x, y);
   };

   createRadialGradient(x1, y1, r1, x2, y2, r2) {
      return this.ctx.createRadialGradient(x1, y1, r1, x2, y2, r2);
   };

   toDataURL(type = 'image/jpeg', quality = "0.5") {
      return this.cvs.toDataURL(type, quality);
   };

   getTransform() { return this.ctx.getTransform() };

   isContextLost() { return this.ctx.isContextLost() };

   isPointInPath() { return this.ctx.isPointInPath() };

   isPointInStroke() { return this.ctx.isPointInStroke() };

   getColor(red, green, blue, alpha) {
      if (typeof red === "string") {
         return red;
      } else if (typeof alpha === "number") {
         return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
      } else if (typeof blue === "number") {
         return `rgba(${red}, ${green}, ${blue}, ${1})`;
      } else if (typeof green === "number") {
         return `rgba(${red}, ${green}, ${Math.round(Math.random() * 255)}, ${1})`;
      } else if (typeof red === "number") {
         return `rgba(${red}, ${red}, ${red}, ${1})`;
      } else {
         return `rgba(${Math.round(Math.random() * 255)}, ${Math.round(Math.random() * 255)}, ${Math.round(Math.random() * 255)}, ${1})`;
      }
   };
}

class Animation {
   constructor(fps) {
      this.fps = fps;
      this.run = false;
   }

   animate(fun) {
      setTimeout(() => {
         if (this.run) {
            fun();
            this.animate(fun);
         }
      }, 1000 / this.fps);
   }

   start(fun) {
      this.run = true;
      this.animate(fun);
   }

   stop() {
      this.run = false;
   }
}

const ID = (selector) => {
   const self = document.getElementById(selector);
   self.on = (event, fun) => {
      self.addEventListener(event, fun);
   };
   self.click = (fun, once = false) => {
      self.addEventListener("click", fun, { once });
   };
   self.text = (text) => (self.innerText = text);
   self.html = (html) => (self.innerText = html);
   return self;
};

const $ = (selector) => {
   const self = document.querySelector(selector);
   self.on = (event, fun) => {
      self.addEventListener(event, fun);
   };
   self.click = (fun, once = false) => {
      self.addEventListener("click", fun, { once });
   };
   self.text = (text) => (self.innerText = text);
   self.html = (html) => (self.innerText = html);
   return self;
};

const $$ = (selector) => {
   const self = document.querySelectorAll(selector);
   self.on = (event, fun) => {
      self.forEach((element) => {
         element.addEventListener(event, fun);
      });
   };
   self.click = (fun, once = false) => {
      self.forEach((element, i) => {
         element.addEventListener("click", (e) => fun(e, element, i), { once });
      });
   };
   self.removeClass = (className) => {
      self.forEach((element) => {
         element.classList.remove(className);
      });
   };
   self.each = (fun) => self.forEach(fun);
   self.map = (fun) => self.map(fun);
   return self;
};

Math.toRadian = degree => (degree * Math.PI) / 180;// degree to radian
Math.toRadian = radian => (radian * 180) / Math.PI;// radian to Degree

Math.rnd = (start = 0, end = 1, int_floor = false) => {
   const result = start + (Math.random() * (end - start));
   return int_floor ? Math.floor(result) : result;
}

/* e.x 
(0 start) -------.------ (10 end) input . = 5
(10 min) ----------------.---------------- (30 max) output . = 20
*/
Math.map = (point, start, end, min, max) => {
   return ((max - min) * (point - start) / (end - start)) + min;
}


function hover(element) {
   const elements = typeof element == "object" ? element : [element];
   elements.forEach((ele) => {
      ele.addEventListener("touchstart", () => {
         if (isMobile) ele.classList.add("_on_");
      });
      ele.addEventListener("mouseenter", () => {
         if (!isMobile) ele.classList.add("_on_");
      });

      ele.addEventListener("touchend", () => {
         if (isMobile) ele.classList.remove("_on_");
      });
      ele.addEventListener("mouseleave", () => {
         if (!isMobile) ele.classList.remove("_on_");
      });
   })
}
/* ---- example ---- */
// .hover {
//     color: black;
//     transition: linear 0.3s;
// }
// .hover._on_ {
//     color: red;
//     transition: linear 0.3s;
// }
hover(document.querySelectorAll(".hover"));


/* ----  local storage set and get ---- */
function setDataFromLocalStorage(key, object) {
   var data = JSON.stringify(object);
   localStorage.setItem(key, data);
}
function getDataFromLocalStorage(key) {
   return JSON.parse(localStorage.getItem(key))
}

// class add in html
function addClass(array, className = "active") {
   if (array.length == undefined) {
      array.classList.forEach(() => array.classList.add(className));
   } else {
      array.forEach((element) => element.classList.add(className));
   }
}
// claass remove in html
function removeClass(array, className = "active") {
   if (array.length == undefined) {
      array.classList.forEach(() => array.classList.remove(className));
   } else {
      array.forEach((element) => element.classList.remove(className));
   }
}

// 2d vector create
class Vector {
   constructor(x = 0, y = 0) {
      this.x = x;
      this.y = y;
   }

   add(value) {
      if (typeof value == "object") {
         this.x += value.x;
         this.y += value.y;
      } else {
         this.x += value;
         this.y += value;
      }
   }

   subtract(value) {
      if (typeof value == "object") {
         this.x -= value.x;
         this.y -= value.y;
      } else {
         this.x -= value;
         this.y -= value;
      }
   }

   multiply(value) {
      if (typeof value == "object") {
         this.x *= value.x;
         this.y *= value.y;
      } else {
         this.x *= value;
         this.y *= value;
      }
   }

   divide(value) {
      if (typeof value == "object") {
         this.x /= value.x;
         this.y /= value.y;
      } else {
         this.x /= value;
         this.y /= value;
      }
   }

   randomWeights() {
      this.x = Math.random() * 2 - 1;
      this.y = Math.random() * 2 - 1;
   }
}


