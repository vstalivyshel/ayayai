use wiggle_ml::*;

const OR: [[Float; 3]; 4] = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 1.]];

const _AND: [[Float; 3]; 4] = [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]];

const _NAND: [[Float; 3]; 4] = [[0., 0., 1.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];

const _XOR: [[Float; 3]; 4] = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];

// Current data set
const TRAIN: [[Float; 3]; 4] = OR;

fn cost(w1: Float, w2: Float, b: Float) -> Float {
    let mut result = 0.0;
    for [x1, x2, outp] in TRAIN {
        let y = sigmoid(x1 * w1 + x2 * w2 + b);
        let d = y - outp;
        result += d * d;
    }

    result / TRAIN.len() as Float
}

fn main() {
    // Setup the model's parameters
    let mut w1 = rand::random::<Float>() * 10.0 - 5.0;
    let mut w2 = rand::random::<Float>() * 10.0 - 5.0;
    let mut b = rand::random::<Float>() * 10.0 - 5.0;
    let eps = 1e-1;
    let rate = 1e-1;

    // Learning process
    for _ in 0..100_000 {
        let c = cost(w1, w2, b);
        println!(
            "w1: {w1:<align$.flen$} w2: {w2:<align$.flen$} c: {c:.flen$}",
            align = 14,
            flen = 7
        );
        let dw1 = (cost(w1 + eps, w2, b) - c) / eps;
        let dw2 = (cost(w1, w2 + eps, b) - c) / eps;
        let db = (cost(w1, w2, b + eps) - c) / eps;
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
    }

    // Result
    for [inp1, inp2, _] in TRAIN {
        println!(
            "{inp1} | {inp2} = {r}",
            r = sigmoid(inp1 * w1 + inp2 * w2 + b)
        )
    }
}
