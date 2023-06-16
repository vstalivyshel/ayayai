use wiggle_ml::*;

#[rustfmt::skip]
const OR: (&str, [Float; 12]) = ("|", [
    0., 0., 0.,
    0., 1., 1.,
    1., 0., 1.,
    1., 1., 1.
]);

#[rustfmt::skip]
const AND: (&str, [Float; 12]) = ("&", [
    0., 0., 0.,
    0., 1., 0.,
    1., 0., 0.,
    1., 1., 1.
]);

#[rustfmt::skip]
const NAND: (&str, [Float; 12]) = ("~", [
    0., 0., 1.,
    0., 1., 1.,
    1., 0., 1.,
    1., 1., 0.
]);

#[rustfmt::skip]
const XOR: (&str, [Float; 12]) = ("^", [
    0., 0., 0.,
    0., 1., 1.,
    1., 0., 1.,
    1., 1., 0.
]);

const D: (&str, [f32; 12]) = OR;

fn main() {
    let ti = Mat::new(4, 2).submat_from(D.1, 3);
    let to = Mat::new(4, 1).submat_from(&D.1[2..], 3);
    let rate = 1e-1;
    let eps = 1e-1;

    let mut nn = NN::new(&[1, 1, 1, 1, 1]).randomize();

   	println!("{c}", c = nn.cost(&ti, &to));
    for _ in 0..1000 * 10 {
        let g = nn.finite_diff(eps, &ti, &to);
        nn.apply_diff(g, rate);
        println!("{c}", c = nn.cost(&ti, &to));
    }

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
