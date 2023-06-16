use wiggle_ml::*;

#[rustfmt::skip]
const OR: [Float; 12] = [
    0., 0., 0.,
    0., 1., 1.,
    1., 0., 1.,
    1., 1., 1.
];

#[rustfmt::skip]
const AND: [Float; 12] = [
    0., 0., 0.,
    0., 1., 0.,
    1., 0., 0.,
    1., 1., 1.
];

#[rustfmt::skip]
const NAND: [Float; 12] = [
    0., 0., 1.,
    0., 1., 1.,
    1., 0., 1.,
    1., 1., 0.
];

#[rustfmt::skip]
const XOR: [Float; 12] = [
    0., 0., 0.,
    0., 1., 1.,
    1., 0., 1.,
    1., 1., 0.
];

struct Xor {
    w1: Mat,
    b1: Mat,
    w2: Mat,
    b2: Mat,
}

const D: (char, [f32; 12]) = ('|', OR);

fn main() {
    let ti = Mat::new(4, 2).submat_from(D.1, 3);
    let to = Mat::new(4, 1).submat_from(&D.1[2..], 3);

    let a0 = ti.get_row(3).fmt_name("input");
    let mut nn = NN::new(&[2, 2, 1])
        .randomize()
        .input(&a0)
        .forward()
        .fmt_inputs(true);
    println!("{c}", c = nn.compute_cost(&ti, &to));
    // println!("{a0}\n{}", nn.get_output().fmt_name("out"));
}
