const pixel = 32;
const inpNums = $$(".number");
const numInp = pixel * pixel;
const numHid0 = 15;
const numHid1 = 15; 
const numOut = 10;

let _min_ = Math.floor(minSize / 100);
_min_ = window.innerHeight < 600 ? _min_ * 40 : _min_ * 60;
const SIZE = 400;


cssRoot.style.setProperty("--scale-n", _min_ / 120);
cssRoot.style.setProperty("--scale", `${_min_ / 30}px`);

cssRoot.style.setProperty("--view-w", `${_min_ * 1.4}px`);
cssRoot.style.setProperty("--view-h", `${_min_ * 1.1}px`);

const c = new Canvas(pixel, pixel, ID("cvs"));
let nn = new NeuralNetwork(numInp, numHid0, numHid1, numOut, 1, 8, SIZE * 1.4, SIZE * 1.1);

const lsd = getDataFromLocalStorage("sb-nn"); // (lsd) local storage data

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
    c.arc(x, y, r / 3);
    c.fill(255);
}

let index, ary = [];
function getCanvasData() {
    ary = [];
    const cvsData = c.getImageData(0, 0, pixel, pixel).data;
    for (let i = 0; i < numInp * 4; i += 4) {
        ary.push(Math.map(cvsData[i], 0, 255, 0, 1));
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
        console.clear();
    }
}

function clearBoard() {
    c.clrScr();
    c.background(0);
    index = undefined;
}
clearBoard();













































