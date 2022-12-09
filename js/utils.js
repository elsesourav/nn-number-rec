"use strict"
let cvsw = 0;
let cvsh = 0;
const pixel = 32;
const cvs = document.createElement("canvas");
const ctx = cvs.getContext("2d");



const canvasSize = (width, height) => {
   cvsw = width; cvsh = height;
   document.getElementById("cvs").appendChild(cvs);
   cvs.setAttribute("width", cvsw);
   cvs.setAttribute("height", cvsh);
   // cvs.style.width = `${cvsw}px`;
   // cvs.style.height = `${cvsh}px`;
};
/* ------------------------------------------------------------ */
let windowWidth = window.innerWidth;
let windowHeight = window.innerHeight;
let size = windowWidth > windowHeight ? windowHeight : windowWidth;
size = Math.floor(size / 100);
size = size > 4 ? size * 45 : size * 70;
canvasSize(pixel, pixel); 
/* ------------------------------------------------------------- */

/* ---------- math ---------- */
const PI = Math.PI;
const sin = x => Math.sin(x);
const cos = y => Math.cos(y);
const atan2 = (y, x) => Math.atan2(y, x);
const abs = n => Math.abs(n);

const toRadian = degree => (degree * Math.PI) / 180;// degree convert to radian
const toDegree = radian => (radian * 180) / Math.PI;// radian convert to Degree

const random = (start = 0, end = 1, int_floor = false) => {
   const result = start + (Math.random() * (end - start));
   return int_floor ? Math.floor(result) : result;
}

/* e.x 
(0 start) -------.------ (10 end) input . = 5
(10 min) ----------------.---------------- (30 max) output . = 20
*/
const map = (point, start, end, min, max) => {
   const per = (point - start) / (end - start);
   return ((max - min) * per) + min;
}

const __getColor__ = (r, g, b, a) => {
   if (typeof r === "string") {
      return r;
   } else if ((a || a === 0) && (b || b === 0) && (g || g === 0) && (r || r === 0)) {
      return `rgba(${r}, ${g}, ${b}, ${a})`;
   } else if (!a && !b && !g && (r || r === 0)) {
      return `rgba(${r}, ${r}, ${r}, ${1})`;
   } else {
      return `rgba(${r}, ${g}, ${b}, ${1})`;
   }
}


/* ------------- canvas ------------ */
const color = (red, green, blue, alpha) => {
   return __getColor__(red, green, blue, alpha);
}
const fill = (red, green, blue, alpha) => {
   ctx.fillStyle = __getColor__(red, green, blue, alpha);
   ctx.fill();
}
const fillStyle = (red, green, blue, alpha) => {
   ctx.fillStyle = __getColor__(red, green, blue, alpha);
}

const strokeStyle = (red, green, blue, alpha) => {
   ctx.strokeStyle = __getColor__(red, green, blue, alpha);
}

const background = (red, green, blue, alpha) => {
   ctx.fillStyle = __getColor__(red, green, blue, alpha);
   ctx.fillRect(0, 0, cvs.width, cvs.height);
}

const clrScr = () => {
   ctx.clearRect(0, 0, cvs.width, cvs.height);
}
const clearRect = (x = 0, y = 0, w = cvs.width, h = cvs.height) => {
   ctx.clearRect(x, y, w, h);
}
const translate = (x, y) => {
   ctx.translate(x, y);
}
const translateX = (x) => {
   ctx.translate(x, 0);
}
const translateY = (y) => {
   ctx.translate(0, y);
}
const transform = (ox, nx, oy, ny) => {
   ctx.transform(ox, nx, oy, ny);
}
const font = (font) => {
   ctx.font = font;
   ctx.textAlign = "center";
}
const text = (text, x, y, w) => {
   ctx.fillText(text, x, y, w);
}
const save = () => {
   ctx.save();
}
const restore = () => {
   ctx.restore();
}
const rotate = (angle) => {
   ctx.rotate(angle);
}
const scale = (x, y) => ctx.scale(x, y);
const scaleX = (x) => ctx.scale(x, 0);
const scaleY = (y) => ctx.scale(0, y);

const lineWidth = (width) => ctx.lineWidth = width;


const rounLline = (sx, sy, ex, ey, width = 1, lineDash = []) => {
   ctx.beginPath();
   ctx.lineWidth = width;
   ctx.lineCap = "round";
   ctx.lineJoin = "round";

   lineDash && ctx.setLineDash(lineDash);
   ctx.moveTo(sx, sy);
   ctx.lineTo(ex, ey);
   ctx.stroke();
   ctx.closePath();
}
const beginPath = () => ctx.beginPath();
const closePath = () => ctx.closePath();

const moveTo = (x, y) => {
   ctx.beginPath();
   ctx.moveTo(x, y);
};

const strokeRect = (x, y, w, h) => {
   ctx.strokeRect(x, y, w, h);
}
const stroke = (red, green, blue, alpha) => {
   ctx.strokeStyle = __getColor__(red, green, blue, alpha);
   ctx.stroke();
}
const lineTo = (x, y) => ctx.lineTo(x, y);

const curve = (sx, sy, ex, ey, lineWidth = 1, radius = 20, fill = false) => {
   ctx.beginPath();
   ctx.lineWidth = lineWidth;
   ctx.moveTo(sx, sy);
   const midx = sx + (ex - sx) / 2;
   const midy = sy + (ey - sy) / 2;
   ctx.quadraticCurveTo(midx, midy - radius, ex, ey);
   ctx.stroke();
   fill && ctx.fill();
   ctx.closePath();
}

const rect = (x, y, w, h) => {
   ctx.rect(x, y, w, h);
}

const fillRect = (x, y, w, h) => {
   ctx.beginPath();
   ctx.fillRect(x, y, w, h);
   ctx.closePath();
}
const arc = (x, y, r, fill = true, lineWidth = 0) => {
   ctx.beginPath();
   const nr = lineWidth ? r - lineWidth : r;
   ctx.arc(x, y, nr, 0, PI * 2, false);
   fill && ctx.fill();
   ctx.lineWidth = lineWidth;
   lineWidth && ctx.stroke();
   ctx.closePath();
}
const globalAlpha = (alpha) => ctx.globalAlpha = alpha;
const getImageData = (sx, sy, sw, sh) => ctx.getImageData(sx, sy, sw, sh);
const putImageData = (image, dx, dy) => ctx.putImageData(image, dx, dy);
const toDataURL = () => ctx.toDataURL("image/jpeg");
const drawImage = (image, x, y, w, h) => ctx.drawImage(image, x, y, w, h);
 
const $ = (givMe) => {
   const self = document.querySelectorAll(givMe);
   self.T = (text) => {
      self.forEach((all) => {
         all.innerText = text;
      });
   };
   self.O = (event, fun) => {
      self.forEach((all) => {
         all.addEventListener(event, fun);
      });
   };
   self.S = (object) => {
      const css = Object.entries(object);
      self.forEach((all) => {
         css.forEach(([prorerty, value]) => {
            all.style[prorerty] = value;
         });
      });
   };
   return self;
};
const _$ = (givMe) => {
   return self = document.querySelector(givMe);
};

// return Id
const ID = (id) => {
   const self = document.getElementById(id);
   self.on = (event, fun) => {
      self.addEventListener(event, fun);
   };
   return self;
};

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

const createEle = (elementName, className = null, appendParentName = null, inrHtml = null) => {
   const e = document.createElement(elementName);
   if (className) e.classList.add(className);
   if (inrHtml) e.innerHTML = inrHtml;
   if (appendParentName) appendParentName.appendChild(e);
   e.on = (event, callBackFun) => {
      if (typeof event != "string") {
         e.addEventListener("click", event);
      } else {
         e.addEventListener(event, callBackFun);
      }
   }
   return e;
}

//use cssRoot.style.setProperty("key", "value");
const cssRoot = document.querySelector(':root');

// when run this app in mobile is return true
const isMobile = localStorage.mobile || window.navigator.maxTouchPoints > 1;
function hover(element) {
   element.addEventListener("touchstart", () => {
      if (isMobile) element.classList.add("show");
   });
   element.addEventListener("mouseenter", () => {
      if (!isMobile) element.classList.add("show");
   });

   element.addEventListener("touchend", () => {
      if (isMobile) element.classList.remove("show");
   });
   element.addEventListener("mouseleave", () => {
      if (!isMobile) element.classList.remove("show");
   });
}


function setDataFromLocalStorage(key, object) {
   var data = JSON.stringify(object);
   localStorage.setItem(key, data);
}
function getDataFromLocalStorage(key) {
   return JSON.parse(localStorage.getItem(key))
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


// rotat 2d array right side (90 deg)
function rotate2dArray(array) {
   let ary = array.slice();
   let row = ary.length;
   let col = ary[0].length;
   let nary = [];

   // nary[i][j] fill with 0
   for (let i = 0; i < col; i++) {
      nary[i] = [];
      for (let j = 0; j < row; j++) {
         nary[i][j] = 0;
      }
   }

   // rotate array into nary 
   for (let i = 0; i < row; i++) {
      for (let j = 0; j < col; j++) {
         nary[j][row - 1 - i] = ary[i][j];
      }
   }

   return nary;
}


// get index form array 
function getIndexWhereOne(array) {
   let max = {
      i: 0,
      val: 0
   }
   for (let i = 0; i < array.length; i++) {
      if (max.val < array[i]) {
         max.i = i;
         max.val = array[i];
      }
   }
   return max.i;
}