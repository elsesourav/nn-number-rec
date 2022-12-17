/* ----------- event listiner ----------- */
const mainDiv = Q("main");
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


// draw events for mouse and touch
let offsetX, offsetY, isDraw = false;
const cvsOffset = c.cvs.getBoundingClientRect();
c.on("click", (e) => {
    offsetX = Math.map(e.clientX, cvsOffset.left, cvsOffset.left + c.cvs.clientWidth, 0, pixel);
    offsetY = Math.map(e.clientY, cvsOffset.top, cvsOffset.top + c.cvs.clientHeight, 0, pixel);
    draw(Math.round(offsetX), Math.round(offsetY), Math.round(pixel / 6));
});
c.on("mousemove", (e) => {
    if (isDraw) {
        offsetX = Math.map(e.clientX, cvsOffset.left, cvsOffset.left + c.cvs.clientWidth, 0, pixel);
        offsetY = Math.map(e.clientY, cvsOffset.top, cvsOffset.top + c.cvs.clientHeight, 0, pixel);
        draw(Math.round(offsetX), Math.round(offsetY), Math.round(pixel / 6));
    }
});
c.on("touchmove", (e) => {
    if (isDraw) {
        offsetX = Math.map(e.touches[0].clientX, cvsOffset.left, cvsOffset.left + c.cvs.clientWidth, 0, pixel);
        offsetY = Math.map(e.touches[0].clientY, cvsOffset.top, cvsOffset.top + c.cvs.clientHeight, 0, pixel);
        draw(Math.round(offsetX), Math.round(offsetY), Math.round(pixel / 6));
    }
});

c.on("mousedown", () => isDraw = true);
document.body.addEventListener("mouseup", () => { isDraw = false; });
let numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
c.on("mouseup", () => {
    getCanvasData();
    let result = nn.feedForward(ary).data[0];
    result = result.map(x => Math.round(x));
    // outputs.innerText = numbers[getIndexWhereOne(result)];
});


c.on("touchstart", () => isDraw = true);
c.on("touchend", () => {
    isDraw = false;
    getCanvasData();
    let result = nn.feedForward(ary).data[0];
    result = result.map(x => Math.round(x));
//     outputs.innerText = numbers[getIndexWhereOne(result)];
});

inpNums.each((e, i) => {
    e.addEventListener("click", () => {
        removeClass(inpNums, "on");
        index = i;
        addClass(e, "on");
    })
})

clearBtn.on("click", () => {
    clearBoard();
    removeClass(inpNums, "on");
    // outputs.innerText = "?";
});
trainingBtn.on("click", () => {
    trainingBtn.classList.toggle("on");
    numInpBtn.classList.toggle("active");
    trainBtn.classList.toggle("active");
});
trainBtn.on("click", () => {
    training();
    trainBtn.classList.toggle("on");
});

if (isMobile) {
    cssRoot.style.setProperty("--cursor", "auto");
}

menuButton.on("click", () => {
    mainDiv.classList.toggle("active");
})

learnRate.on("input", (e) => {
    let val = e.target.value;
    nn.lrStape = val;
    val = val / 8;
    val = parseFloat((val * val).toFixed(2));
    val = val >= 3 ? Math.round(val) : val;
    lrShow.innerText = nn.lrnRate = val;
})
saveBtn.on("click", () => {
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
resteBtn.on("click", () => {
    setDataFromLocalStorage("sb-nn", "");
})