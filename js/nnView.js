class NNView {
    constructor(nn, width, height) {
        this.nn = nn;
        this.width = width;
        this.height = height;
        this.c = new Canvas(this.width, this.height, ID("vidw-cvs"));
        this.fps = 60;
        this.ani = new Animation(this.fps);

        this.inps = [];
        this.hids0 = [];
        this.hids1 = [];
        this.outs = [];

        this.margin = 10;
        this.sclW = this.width / 3.2 - this.margin;
        this.sclH = this.height / 17;
        this.r = this.sclH / 3;
        this.numCol = 17;

        this.runTime = this.f * 3;
        this.f = 10;

        this.setInps();
        this.setHids0();
        this.setHids1();
        this.setOuts();

        this.inpPnt = undefined;
        this.h0 = undefined;
        this.h1 = undefined;
        this.out = undefined;
        this.maxIndex = undefined;
    }

    aniStart() {
        this.runTime = this.f * 4;
        this.inpPnt = getInputPoints(this.nn._inputs.data[0], this.inps.length);
        this.h0 = this.nn.hidden0.data[0].slice().map(e => e > 1 ? 1 : e);
        this.h1 = this.nn.hidden1.data[0].slice().map(e => e > 1 ? 1 : e);
        this.out = this.nn.outputs.data[0].slice().map(e => e > 1 ? 1 : e);
        this.maxIndex = getMaximumIndex(this.out);

        this.ani.start(() => {
            this.draw();
            this.drawVal();
            this.runTime--;
            if (this.runTime <= 0) this.aniStop();
        });
    }

    aniStop() {
        this.ani.stop();
    }

    draw() {
        this.c.clrScr();
        this.connoction();
        this.inputs();
        this.hiddens0();
        this.hiddens1();
        this.outputs();
    }

    drawVal() {
        this.inps.forEach((v, i) => {
            this.c.arc(v.x, v.y, v.r > 2 ? v.r - 1 : v.r);
            this.c.fill(0, 255 * this.inpPnt[i], 0);
        });

        if (this.runTime < this.f * 3) {
            this.hids0.forEach((v, i) => {
                this.c.arc(v.x, v.y, v.r - 1);
                this.c.fill(0, 255 * this.h0[i], 0);
            })
        }

        if (this.runTime < this.f * 2) {
            this.hids1.forEach((v, i) => {
                this.c.arc(v.x, v.y, v.r - 1);
                this.c.fill(0, 255 * this.h1[i], 0);
            })
        }

        if (this.runTime < this.f) {
            this.outs.forEach((v, i) => {
                this.c.arc(v.x, v.y, v.r - 1);
                this.c.fill(0, 255 * this.out[i], 0);
                if (this.maxIndex == i) {
                    this.c.fillStyle(255);
                    this.c.textMode(this.c.CENTER);
                    this.c.font(this.sclH * .9, "Arial");
                    this.c.fillText(v.index, v.x + v.r * 3, v.y + 2);

                    this.c.moveTo(v.x + v.r * 2, v.y + v.r);
                    this.c.lineTo(v.x + v.r * 4, v.y + v.r);
                    this.c.stroke(0, 255, 0);
                } else {
                    this.c.fillStyle(255 * 0.3);
                    this.c.textMode(this.c.CENTER);
                    this.c.font(this.sclH * .9, "Arial");
                    this.c.fillText(v.index, v.x + v.r * 3, v.y + 2);
                }
            });
        }
    }

    inputs() {
        this.inps.forEach((v, i) => {
            this.c.arc(v.x, v.y, v.r);
            this.c.lineWidth(2);
            this.c.stroke(0, 255, 255);
            this.c.fill(0);
        });
    }

    hiddens0() {
        this.hids0.forEach(v => {
            this.c.arc(v.x, v.y, v.r);
            this.c.lineWidth(2);
            this.c.stroke(0, 255, 255);
            this.c.fill(0);
        })
    }

    hiddens1() {
        this.hids1.forEach(v => {
            this.c.arc(v.x, v.y, v.r);
            this.c.lineWidth(2);
            this.c.stroke(0, 255, 255);
            this.c.fill(0);
        })
    }

    outputs() {
        this.outs.forEach(v => {
            this.c.arc(v.x, v.y, v.r);
            this.c.lineWidth(2);
            this.c.stroke(0, 255, 255);
            this.c.fill(0);
        })
    }

    connoction() {
        this.inps.forEach(ins => {
            this.hids0.forEach(hid0 => {
                this.c.moveTo(ins.x, ins.y);
                this.c.lineTo(hid0.x, hid0.y);
                this.c.lineWidth(this.sclH / 100);
                this.c.stroke(255);
            })
        })

        this.hids0.forEach(hd0 => {
            this.hids1.forEach(hd1 => {
                this.c.moveTo(hd0.x, hd0.y);
                this.c.lineTo(hd1.x, hd1.y);
                this.c.lineWidth(this.sclH / 100);
                this.c.stroke(255);
            })
        })

        this.hids1.forEach(hd1 => {
            this.outs.forEach(ots => {
                this.c.moveTo(hd1.x, hd1.y);
                this.c.lineTo(ots.x, ots.y);
                this.c.lineWidth(this.sclH / 100);
                this.c.stroke(255);
            })
        })
    }

    setInps() {
        this.inps = [];
        for (let i = 0; i < this.numCol; i++) {

            if (i == Math.round(this.numCol / 2) - 1) {
                let gap = this.sclH / 4

                this.inps.push({
                    x: this.margin + this.r,
                    y: this.sclH * i + this.r * 1.5 - gap,
                    r: this.r / 4
                });
                this.inps.push({
                    x: this.margin + this.r,
                    y: this.sclH * i + this.r * 1.5 + gap,
                    r: this.r / 4
                });

            } else {
                this.inps.push({
                    x: this.margin + this.r,
                    y: this.sclH * i + this.r * 1.5,
                    r: this.r
                });
            }
        }
    }

    setHids0() {
        this.hids0 = [];
        for (let i = 1; i < this.nn._numHid0 + 1; i++) {

            this.hids0.push({
                x: this.sclW + this.r * 2,
                y: this.sclH * i + this.r * 1.5,
                r: this.r
            });
        }
    }

    setHids1() {
        this.hids1 = [];
        for (let i = 1; i < this.nn._numHid1 + 1; i++) {

            this.hids1.push({
                x: this.sclW * 2 + this.r * 2,
                y: this.sclH * i + this.r * 1.5,
                r: this.r
            });
        }
    }

    setOuts() {
        this.outs = [];
        for (let i = 0; i < this.nn._numOut; i++) {

            this.outs.push({
                x: this.sclW * 3 + this.r * 2,
                y: this.sclH * 1.7 * i + this.r * 1.5 + this.sclH / 4,
                r: this.r * 1.8,
                index: i
            });
        }
    }
}


function getInputPoints(array, numPoints) {
    const numP = Math.floor(array.length / numPoints);
    let points = [];

    for (let i = 0; i < numPoints; i++) {
        let sum = 0;
        for (let j = 0; j < numP; j++) {
            sum += array[numP * i + j];
        }
        points.push(sum);
    }
    return points.map(e => e > 0 ? 1 : 0);
}

function getMaximumIndex(array) {
    let max = array[0];
    let index = 0;
    for (let i = 1; i < array.length; i++) {
        if (max < array[i]) { max = array[i]; index = i; };
    }
    return index;
}