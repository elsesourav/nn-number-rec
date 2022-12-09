cssRoot.style.setProperty("--scale", `${size / 25}px`);
cssRoot.style.setProperty("--scale-n", size / 100);
const numInp = pixel * pixel;
const numHid0 = 10;
const numHid1 = 10;
const numOut = 10;

let nn = new NeuralNetwork(numInp, numHid0, numHid1, numOut, 1, 8);

const lsd = getDataFromLocalStorage("sb-nn"); // (lsd) local stroage data

if (lsd) {
    nn.bias0.data = lsd.bias0;
    nn.bias1.data = lsd.bias1;
    nn.bias2.data = lsd.bias2;
    nn.weights0.data = lsd.weights0;
    nn.weights1.data = lsd.weights1;
    nn.weights2.data = lsd.weights2;
    nn.lrnRate = lsd.lrnRate;
    nn.lrStape = lsd.lrStape;
    
    ID("learn-rate").value = lsd.lrStape;
    ID("lr-show").innerText = lsd.lrnRate;
}

function draw(x, y, r) {
    fillStyle(255);
    arc(x, y, r / 3);
}

const inpNums = $(".number");
const outputs = ID("outputs");

let index, ary = [];
function getCanvasData() {
    ary = [];
    const cvsData = getImageData(0, 0, pixel, pixel).data;
    for (let i = 0; i < numInp * 4; i += 4) {
        ary.push(map(cvsData[i], 0, 255, 0, 1));
    }
}

function setOutputIndex(index) {
    let ary = [];
    for (let i = 0; i < numOut; i++) {
        if (i != index) ary.push(0);
        else ary.push(1);
    }
    return ary;
}

function training() {
    if (index != undefined) {
        getCanvasData();
        let outAry = setOutputIndex(index);
        nn.train(ary, outAry);
        outputs.innerText = index;
        console.clear();
    }
}

function clearBoard() {
    clrScr();
    background(0);
    index = undefined;
}
clearBoard();

/* ----------- event listiner ----------- */
const mainDiv = _$("main");
const menuButton = ID("menu-icon");
const clearBtn = ID("clear");
const trainingBtn = ID("training");
const trainBtn = ID("train");
const numInpBtn = ID("num-btn");
const learnRate = ID("learn-rate");
const lrShow = ID("lr-show");
const saveBtn = ID("save");
const resteBtn = ID("reste");
const hoverClass = $(".hover");

hoverClass.forEach(hc => {
    hover(hc);
})

// draw events for mouse and touch
let offsetX, offsetY, isDraw = false;
const cvsOffset = cvs.getBoundingClientRect();
cvs.addEventListener("mousemove", (e) => {
    if (isDraw) {
        offsetX = map(e.clientX, cvsOffset.left, cvsOffset.left + cvs.clientWidth, 0, pixel);
        offsetY = map(e.clientY, cvsOffset.top, cvsOffset.top + cvs.clientHeight, 0, pixel);
        draw(Math.round(offsetX), Math.round(offsetY), Math.round(pixel / 6));
    }
});
cvs.addEventListener("touchmove", (e) => {
    if (isDraw) {
        offsetX = map(e.touches[0].clientX, cvsOffset.left, cvsOffset.left + cvs.clientWidth, 0, pixel);
        offsetY = map(e.touches[0].clientY, cvsOffset.top, cvsOffset.top + cvs.clientHeight, 0, pixel);
        draw(Math.round(offsetX), Math.round(offsetY), Math.round(pixel / 6));
    }
});

cvs.addEventListener("mousedown", () => isDraw = true);
document.body.addEventListener("mouseup", () => {
    isDraw = false;
});
let numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
cvs.addEventListener("mouseup", () => {
    getCanvasData();
    let result = nn.feedForward(ary).data[0];
    result = result.map(x => Math.round(x));
    outputs.innerText = numbers[getIndexWhereOne(result)];
});


cvs.addEventListener("touchstart", () => isDraw = true);
cvs.addEventListener("touchend", () => {
    isDraw = false;
    getCanvasData();
    let result = nn.feedForward(ary).data[0];
    result = result.map(x => Math.round(x));
    outputs.innerText = numbers[getIndexWhereOne(result)];
});

inpNums.forEach((e, i) => {
    e.addEventListener("click", () => {
        removeClass(inpNums, "on");
        index = i;
        addClass(e, "on");
    })
})

clearBtn.addEventListener("click",  () => {
    clearBoard();
    removeClass(inpNums, "on");
    outputs.innerText = "?";
});
trainingBtn.addEventListener("click", () => {
    trainingBtn.classList.toggle("on");
    numInpBtn.classList.toggle("active");
    trainBtn.classList.toggle("active");
});
trainBtn.addEventListener("click", () => {
    training();
    trainBtn.classList.toggle("on");
});

if (isMobile) {
    cssRoot.style.setProperty("--cursor", "auto");
}

menuButton.addEventListener("click", () => {
    mainDiv.classList.toggle("active");
})

learnRate.addEventListener("input", (e) => {
    let val = e.target.value;
    nn.lrStape = val;
    val = val / 8;
    val = parseFloat((val * val).toFixed(2));
    val = val >= 3 ? Math.round(val) : val;
    lrShow.innerText = nn.lrnRate = val;
})
saveBtn.addEventListener("click", () => {
    const obj = {
        bias0: nn.bias0.data,
        bias1: nn.bias1.data,
        bias2: nn.bias2.data,
        weights0: nn.weights0.data,
        weights1: nn.weights1.data,
        weights2: nn.weights2.data,
        lrnRate: nn.lrnRate,
        lrStape: nn.lrStape
    }
    setDataFromLocalStorage("sb-nn", obj);
})
resteBtn.addEventListener("click", () => {
    setDataFromLocalStorage("sb-nn", "");
})











































