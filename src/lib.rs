use rand::{
    self,
    distributions::{Distribution, Uniform},
};
use std::{fmt, ops::Range};

pub type Float = f32;

macro_rules! mat_at {
    ($mat:ident, $raw:ident, $col:ident $($act:tt $v:expr)?) => {
        $mat.elems[$raw * $mat.cols + $col] $($act $v)?
    };
}

pub fn sigmoidf(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone, Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub elems: Vec<Float>,
    pub mantissa_fmt: usize,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            elems: vec![Float::default(); rows * cols],
            mantissa_fmt: 6,
        }
    }

    pub fn new_fill(r: usize, c: usize, v: Float) -> Self {
        let mut new = Self::new(r, c);
        new.elems.fill(v);
        new
    }

    pub fn new_rand(r: usize, c: usize) -> Self {
        let mut new = Self::new(r, c);
        new.fill_rand(0.0..1.1);
        new
    }

    pub fn new_randrang(r: usize, c: usize, range: Range<Float>) -> Self {
        let mut new = Self::new(r, c);
        new.fill_rand(range);
        new
    }

    pub fn fill_rand(&mut self, range: Range<Float>) {
        let mut rng = rand::thread_rng();
        let between = Uniform::from(range);
        self.elems
            .iter_mut()
            .for_each(|i| *i = between.sample(&mut rng));
    }

    pub fn dot_of(a: &Self, b: &Self) -> Self {
        let mut new = Self::new(a.rows, b.cols);
        new.into_dot_of(a, b);
        new
    }

    pub fn into_dot_of(&mut self, a: &Self, b: &Self) {
        assert!(a.cols == b.rows);
        assert!(self.rows == a.rows);
        assert!(self.cols == b.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                mat_at!(self, i, j = 0.);
                for k in 0..a.cols {
                    mat_at!(self, i, j += mat_at!(a, i, k) * mat_at!(b, k, j));
                }
            }
        }
    }

    pub fn sum(&mut self, other: &Self) {
        assert!(self.cols == other.cols);
        assert!(self.rows == other.rows);
        self.elems = self
            .elems
            .iter_mut()
            .zip(&other.elems)
            .map(|(a, b)| *a + b)
            .collect();
    }
}

impl fmt::Display for Mat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();

        for i in 0..self.rows {
            for j in 0..self.cols {
                s.push_str(&format!(
                    "{item:.n$} ",
                    item = mat_at!(self, i, j),
                    n = self.mantissa_fmt
                ))
            }
            s.push('\n');
        }

        write!(f, "{s}", s = s.trim_end())
    }
}
