use wiggle_ml::sample::arr_sample;
use wiggle_ml::*;

const D: (&str, [f32; 12]) = arr_sample::AND;
const N: usize = 1000 * 100;
const DBG: bool = false;
const RATE: Float = 1e-1;

fn main() {
    let ti = Mat::new(4, 2).submat_from(D.1, 3);
    let to = Mat::new(4, 1).submat_from(&D.1[2..], 3);
    let mut nn = NN::new(&[2, 2, 1]).randomized_fixed().fmt_inputs(true);
    let mut g = nn.clone();

    for _ in 0..N {
        nn.finite_diff(&mut g, &ti, &to, 1e-1);
        // nn.backprop(&mut g, &ti, &to);
        nn.learn(&g, RATE);
        if DBG {
            println!("{c}", c = nn.cost(&ti, &to));
        }
    }

    println!("{c}", c = nn.cost(&ti, &to));
    println!("-----------------------------------------");
    for i in 0..ti.rows {
        let x = ti.row(i);
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
