use wiggle_ml::sample::arr_sample;
use wiggle_ml::*;

fn fd(mut nn: NN, ti: Mat, to: Mat) {
    let eps = 1e-1;

    for _ in 0..N {
        let g = nn.finite_diff(eps, &ti, &to);
        nn.apply_diff(g, RATE);
        if DBG {
            println!("{c}", c = nn.cost(&ti, &to));
        }
    }

    println!("{c}", c = nn.cost(&ti, &to));
    println!("-----------------------------------------");
    for i in 0..ti.rows {
        let x = ti.get_row(i);
        nn.set_input(&x);
        nn.forward();
        println!(
            "{x1} {s} {x2} = {out}",
            s = D.0,
            x1 = x.get_at(0, 0),
            x2 = x.get_at(0, 1),
            out = nn.get_ref_output().get_at(0, 0)
        );
    }
}

fn bp(mut nn: NN, ti: Mat, to: Mat) {
    for _ in 0..N {
        let g = nn.backprop(&ti, &to);
        nn.apply_diff(g, RATE);
        if DBG {
            println!("{c}", c = nn.cost(&ti, &to));
        }
    }

    println!("{c}", c = nn.cost(&ti, &to));
    println!("-----------------------------------------");
    for i in 0..ti.rows {
        let x = ti.get_row(i);
        nn.set_input(&x);
        nn.forward();
        println!(
            "{x1} {s} {x2} = {out}",
            s = D.0,
            x1 = x.get_at(0, 0),
            x2 = x.get_at(0, 1),
            out = nn.get_ref_output().get_at(0, 0)
        );
    }
}

const D: (&str, [f32; 12]) = arr_sample::AND;
const N: usize = 1000*100;
const DBG: bool = false;
const RATE: Float = 1e-1;

fn main() {
    let ti = Mat::new(4, 2).submat_from(D.1, 3);
    let to = Mat::new(4, 1).submat_from(&D.1[2..], 3);
    let nn = NN::new(&[2, 2, 1]).randomize_fixed().fmt_inputs(true);
    fd(nn, ti, to);
}
