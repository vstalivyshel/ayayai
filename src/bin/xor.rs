use std::ops;
use wiggle_ml::*;
#[rustfmt::skip]
const OR: [[Float; 3]; 4] = [
    [0., 0., 0.],
    [0., 1., 1.],
    [1., 0., 1.],
    [1., 1., 1.]
];

#[rustfmt::skip]
const AND: [[Float; 3]; 4] = [
    [0., 0., 0.],
    [0., 1., 0.],
    [1., 0., 0.],
    [1., 1., 1.]
];

#[rustfmt::skip]
const NAND: [[Float; 3]; 4] = [
    [0., 0., 1.],
    [0., 1., 1.],
    [1., 0., 1.],
    [1., 1., 0.]
];

#[rustfmt::skip]
const XOR: [[Float; 3]; 4] = [
    [0., 0., 0.],
    [0., 1., 1.],
    [1., 0., 1.],
    [1., 1., 0.]
];

// Current data set
const TRAIN: [[Float; 3]; 4] = NAND;

#[derive(Clone, Debug)]
struct N {
    w1: Float,
    w2: Float,
    b: Float,
}

impl N {
    fn new() -> Self {
        Self {
            w1: rand::random::<Float>(),
            w2: rand::random::<Float>(),
            b: rand::random::<Float>(),
        }
    }

    fn forward(&self, x1: Float, x2: Float) -> Float {
        sigmoid(self.w1 * x1 + self.w2 * x2 + self.b)
    }
}

#[derive(Clone, Debug)]
struct Xor {
    or: N,
    nand: N,
    and: N,
}

impl Xor {
    fn new() -> Self {
        Self {
            or: N::new(),
            nand: N::new(),
            and: N::new(),
        }
    }
    // takes an input and feeds it to the model returning output of each neuron
    // the idea is to feed the output of 'nand' and 'or' neurons to the last 'and' neuron
    // and receive it's output as a result of prediction
    //
    //          bias1
    //           |
    // inp1 - *or*
    // *w1  \/     \
    //	   / \      *and* -> output
    // inp2 - *nand* /
    // *w2      |
    //         bias2
    //
    // Basically this function call pushs data through one layer of neurons
    fn forward(&self, x1: Float, x2: Float) -> Float {
        let a1 = self.or.forward(x1, x2);
        let a2 = self.nand.forward(x1, x2);
        self.and.forward(a1, a2)
    }

    fn cost(&self) -> Float {
        let mut result = 0.0;
        for [x1, x2, outp] in TRAIN {
            let y = self.forward(x1, x2);
            let d = y - outp;
            result += d * d;
        }

        result / TRAIN.len() as Float
    }

    fn finite_diff(&mut self, eps: Float) -> Self {
        let mut g = Self::new();
        let c = self.cost();
        let mut saved: Float;

        saved = self.or.w1;
        self.or.w1 += eps;
        g.or.w1 = (self.cost() - c) / eps;
        self.or.w1 = saved;

        saved = self.or.w2;
        self.or.w2 += eps;
        g.or.w2 = (self.cost() - c) / eps;
        self.or.w2 = saved;

        saved = self.or.b;
        self.or.b += eps;
        g.or.b = (self.cost() - c) / eps;
        self.or.b = saved;

        saved = self.nand.w1;
        self.nand.w1 += eps;
        g.nand.w1 = (self.cost() - c) / eps;
        self.nand.w1 = saved;

        saved = self.nand.w2;
        self.nand.w2 += eps;
        g.nand.w2 = (self.cost() - c) / eps;
        self.nand.w2 = saved;

        saved = self.nand.b;
        self.nand.b += eps;
        g.nand.b = (self.cost() - c) / eps;
        self.nand.b = saved;

        saved = self.and.w1;
        self.and.w1 += eps;
        g.and.w1 = (self.cost() - c) / eps;
        self.and.w1 = saved;

        saved = self.and.w2;
        self.and.w2 += eps;
        g.and.w2 = (self.cost() - c) / eps;
        self.and.w2 = saved;

        saved = self.and.b;
        self.and.b += eps;
        g.and.b = (self.cost() - c) / eps;
        self.and.b = saved;

        g
    }

    fn learn(&mut self, g: &Self, rate: Float) {
        self.or.w1 -= g.or.w1 * rate;
        self.or.w2 -= g.or.w2 * rate;
        self.or.b -= g.or.b * rate;

        self.nand.w1 -= g.nand.w1 * rate;
        self.nand.w2 -= g.nand.w2 * rate;
        self.nand.b -= g.nand.b * rate;

        self.and.w1 -= g.and.w1 * rate;
        self.and.w2 -= g.and.w2 * rate;
        self.and.b -= g.and.b * rate;
    }
}

fn main() {
    let mut m = Xor::new();
    let eps = 1e-1;
    let rate = 1e-1;
    println!("{c}", c = m.cost());
    for _ in 0..1_000_000 {
        let g = m.finite_diff(eps);
        m.learn(&g, rate);
        println!("{c}", c = m.cost());
    }

    for [x1, x2, _] in TRAIN {
        println!("{x1} ^ {x2} = {r}", r = m.forward(x1, x2));
    }
}
