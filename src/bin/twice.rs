use wiggle_ml::*;

fn cost(w: Float) -> Float {
    let mut result = 0.0;

    for [x, y] in TRAINING_DATA {
        let d = x * w - y;
        result += d * d;
    }

    result / TRAINING_DATA.len() as Float
}

#[allow(dead_code)]
fn fd(mut w: Float, dbg: bool) {
    // This is out model with 1 parameter

    // With this magic value we will be able to tweek model's parameter
    // and control it's output
    let eps = 1e-3;

    // Actual learning process
    for _ in 0..N {
        let c = cost(w);
        let dw = (cost(eps + w) - c) / eps;
        w -= dw * RATE;
        if dbg {
            println!("c: {c}; w: {w}", c = cost(w));
        }
    }
    println!("c: {c}; w: {w}", c = cost(w));
}

#[allow(dead_code)]
fn g(mut w: Float, dbg: bool) {
    fn dcost(w: Float) -> Float {
        let mut result = 0.0;
        for [x, y] in TRAINING_DATA {
            // derivative of cost function:
            // c'(w) = 2 * (x_i * w - y_i) * x_i
            result += 2. * (x * w - y) * x;
        }
        result / TRAINING_DATA.len() as Float
    }

    // let mut w = rand::random::<Float>();
    for _ in 0..N {
        let dw = dcost(w);
        w -= dw * RATE;
        if dbg {
            println!("c: {c}; w: {w}", c = cost(w));
        }
    }
    println!("c: {c}; w: {w}", c = cost(w));
}

#[rustfmt::skip]
const TRAINING_DATA: [[Float; 2]; 5] = [
    [0., 0.],
    [1., 2.],
    [2., 4.],
    [3., 6.],
    [4., 8.]
];

const N: usize = 11;
const RATE: Float = 1e-1;

fn main() {
    let w = rand::random::<Float>();
    println!("До : {w}");
    println!("gd:");
    g(w, false);
    println!("-------------------------------");
    println!("fd:");
    fd(w, false);
}
