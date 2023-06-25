use crate::Float;

pub mod arr_sample {
    use super::*;

    #[rustfmt::skip]
    pub const OR: (&str, [Float; 12]) = ("|", [
        0., 0., 0.,
        0., 1., 1.,
        1., 0., 1.,
        1., 1., 1.
    ]);

    #[rustfmt::skip]
    pub const AND: (&str, [Float; 12]) = ("&", [
        0., 0., 0.,
        0., 1., 0.,
        1., 0., 0.,
        1., 1., 1.
    ]);

    #[rustfmt::skip]
    pub const NAND: (&str, [Float; 12]) = ("~&", [
        0., 0., 1.,
        0., 1., 1.,
        1., 0., 1.,
        1., 1., 0.
    ]);

    #[rustfmt::skip]
    pub const XOR: (&str, [Float; 12]) = ("^", [
        0., 0., 0.,
        0., 1., 1.,
        1., 0., 1.,
        1., 1., 0.
    ]);
}

#[rustfmt::skip]
pub const OR: (&str, [[Float; 3]; 4]) = ("|", [
    [0., 0., 0.],
    [0., 1., 1.],
    [1., 0., 1.],
    [1., 1., 1.]
]);

#[rustfmt::skip]
pub const AND: (&str, [[Float; 3]; 4]) = ("&", [
    [0., 0., 0.],
    [0., 1., 0.],
    [1., 0., 0.],
    [1., 1., 1.]
]);

#[rustfmt::skip]
pub const NAND: (&str, [[Float; 3]; 4]) = ("~&", [
    [0., 0., 1.],
    [0., 1., 1.],
    [1., 0., 1.],
    [1., 1., 0.]
]);

#[rustfmt::skip]
pub const XOR: (&str, [[Float; 3]; 4]) = ("^", [
    [0., 0., 0.],
    [0., 1., 1.],
    [1., 0., 1.],
    [1., 1., 0.]
]);
