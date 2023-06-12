use std::ops;
use wiggle_ml::*;

#[derive(Debug)]
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
}

impl ops::SubAssign for N {
    fn sub_assign(&mut self, other: Self) {
        self.w1 -= other.w1;
        self.w2 -= other.w2;
        self.b -= other.b;
    }
}

#[derive(Debug)]
struct Xor {
    nand: N,
    or: N,
    and: N,
}

impl Xor {
    const TRAIN: [[Float; 3]; 4] = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];

    fn new() -> Self {
        Self {
            nand: N::new(),
            or: N::new(),
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
        // this is now the x1 input for 'and' neuron
        let or_out = sigmoid(self.or.w1 * x1 + self.or.w2 * x2 + self.or.b);
        // and this is the x2 input for 'and' neuron
        let nand_out = sigmoid(self.nand.w1 * x1 + self.nand.w2 * x2 + self.nand.b);
        sigmoid(self.and.w1 * or_out + self.and.w2 * nand_out + self.and.b)
    }

    fn cost(&self) -> Float {
        let mut result = 0.0;
        for [x1, x2, outp] in Self::TRAIN {
            let y = self.forward(x1, x2);
            let d = y - outp;
            result += d * d;
        }

        result / Self::TRAIN.len() as Float
    }

    fn finite_dif(&mut self, eps: Float, rate: Float) -> Self {
        let mut g = Self::new();
        let c = self.cost();
        let mut saved: Float;

        // !!! JUST FOR EDUCATIONAL PURPOSE !!!

        // or
        saved = self.or.w1;
        self.or.w1 += eps;
        g.or.w1 = rate * ((self.cost() - c) / eps);
        self.or.w1 = saved;

        saved = self.or.w2;
        self.or.w2 += eps;
        g.or.w2 = rate * ((self.cost() - c) / eps);
        self.or.w2 = saved;

        saved = self.or.b;
        self.or.b += eps;
        g.or.b = rate * ((self.cost() - c) / eps);
        self.or.b = saved;

        // nand
        saved = self.nand.w1;
        self.nand.w1 += eps;
        g.nand.w1 = rate * ((self.cost() - c) / eps);
        self.nand.w1 = saved;

        saved = self.nand.w2;
        self.nand.w2 += eps;
        g.nand.w2 = rate * ((self.cost() - c) / eps);
        self.nand.w2 = saved;

        saved = self.nand.b;
        self.nand.b += eps;
        g.nand.b = rate * ((self.cost() - c) / eps);
        self.or.b = saved;

        // and
        saved = self.and.w1;
        self.and.w1 += eps;
        g.and.w1 = rate * ((self.cost() - c) / eps);
        self.and.w1 = saved;

        saved = self.and.w2;
        self.and.w2 += eps;
        g.and.w2 = rate * ((self.cost() - c) / eps);
        self.and.w2 = saved;

        saved = self.and.b;
        self.and.b += eps;
        g.and.b = rate * ((self.cost() - c) / eps);
        self.and.b = saved;

        g
    }

    fn apply_dif(&mut self, g: Xor) {
        self.nand -= g.nand;
        self.or -= g.or;
        self.and -= g.and;
    }
}

fn main() {
    let eps = 1e-1;
    let rate = 1e-1;

    let mut m = Xor::new();
    // println!("{}", m.cost());

    for _ in 0..1000 {
        let g = m.finite_dif(eps, rate);
        m.apply_dif(g);
        println!("{}", m.cost());
    }

    for [x1, x2, _] in Xor::TRAIN {
        println!("{x1} ^ {x2} = {r}", r = m.forward(x1, x2))
    }
}
