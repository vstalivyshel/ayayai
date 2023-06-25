use wiggle_ml::sample;
use wiggle_ml::*;

fn cost(w1: Float, w2: Float, b: Float) -> Float {
    let mut result = 0.0;
    let train = TRAIN.1;
    for [x1, x2, outp] in train {
        let y = sigmoid(x1 * w1 + x2 * w2 + b);
        let d = y - outp;
        result += d * d;
    }

    result / train.len() as Float
}

// using gradient discent
#[allow(dead_code)]
fn g(mut w1: Float, mut w2: Float, mut b: Float, dbg: bool) {
    //                                           dw1    dw2    db
    fn gcost(w1: Float, w2: Float, b: Float) -> (Float, Float, Float) {
        let mut dw1 = 0.;
        let mut dw2 = 0.;
        let mut db = 0.;
        let train = TRAIN.1;
        for [xi, yi, zi] in train {
            let ai = sigmoid(xi * w1 + yi * w2 + b);
            let db_i = 2. * (ai - zi) * ai * (1. - ai);
            // D_X_i = 2(a_i - z_i) * a_i (1 - a_i) * x_i
            dw1 += db_i * xi;
            // D_Y_i = 2(a_i - z_i) * a_i (1 - a_i) * y_i
            dw2 += db_i * yi;
            // D_B_i = 2(a_i - z_i) * a_i (1 - a_i)
            db += db_i;
        }
        let tl = train.len() as Float;

        (dw1 / tl, dw2 / tl, db / tl)
    }

    for i in 0..N {
        let c = cost(w1, w2, b);
        if dbg {
            println!(
                "({i}) w1: {w1:<align$.m$} w2: {w2:<align$.m$} c: {c:.m$}",
                align = 14,
                m = 7
            );
        }
        let (dw1, dw2, db) = gcost(w1, w2, b);
        w1 -= RATE * dw1;
        w2 -= RATE * dw2;
        b -= RATE * db;
    }

    println!("c: {c}", c = cost(w1, w2, b));
    for [inp1, inp2, _] in TRAIN.1 {
        println!(
            "{inp1} {s} {inp2} = {r}",
            s = TRAIN.0,
            r = sigmoid(inp1 * w1 + inp2 * w2 + b)
        )
    }
}

// using finite difference
#[allow(dead_code)]
fn fd(mut w1: Float, mut w2: Float, mut b: Float, dbg: bool) {
    let eps = 1e-1;

    for i in 0..N {
        let c = cost(w1, w2, b);
        if dbg {
            println!(
                "({i}) w1: {w1:<align$.m$} w2: {w2:<align$.m$} c: {c:.m$}",
                align = 14,
                m = 7
            );
        }
        let dw1 = (cost(w1 + eps, w2, b) - c) / eps;
        let dw2 = (cost(w1, w2 + eps, b) - c) / eps;
        let db = (cost(w1, w2, b + eps) - c) / eps;
        w1 -= RATE * dw1;
        w2 -= RATE * dw2;
        b -= RATE * db;
    }

    println!("c: {c}", c = cost(w1, w2, b));
    for [inp1, inp2, _] in TRAIN.1 {
        println!(
            "{inp1} {s} {inp2} = {r}",
            s = TRAIN.0,
            r = sigmoid(inp1 * w1 + inp2 * w2 + b)
        )
    }
}

const RATE: Float = 1e-1;
const TRAIN: (&str, [[Float; 3]; 4]) = sample::AND;
const N: usize = 100;

fn main() {
    let w1 = rand::random::<Float>();
    let w2 = rand::random::<Float>();
    let b = rand::random::<Float>();

    println!("gd:");
    g(w1, w2, b, false);
    println!("-------------------------------");
    println!("fd:");
    fd(w1, w2, b, false);
}
