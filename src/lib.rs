use rand::{
    self,
    distributions::{Distribution, Uniform},
};
use std::{fmt, ops::Range};

pub type Float = f32;

pub fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone, Debug, Default)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
    // TODO:
    pub elems: Vec<Float>,
    pub fmt_mantissa: usize,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            stride: cols,
            elems: vec![Float::default(); rows * cols],
            fmt_mantissa: 6,
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

    pub fn fill_sub_from_slice(&mut self, src: &[Float], stride: usize) {
        let src = src.iter();
        let mut j = 0;
        if self.cols == 1 {
            for e in src.step_by(stride) {
                self.elems[j] = *e;
                j += 1;
            }

            return;
        }

        let mut src = src.enumerate();
        while let Some((i, e)) = src.next() {
            if (i + 1) % stride == 0 && j % self.cols == 0 {
                continue;
            }

            self.elems[j] = *e;
            j += 1;
        }
    }

    pub fn fill_from(&mut self, src: &Mat) {
        assert!(self.rows == src.rows, "On line {}", line!());
        assert!(self.cols == src.cols, "On line {}", line!());
        self.elems.copy_from_slice(src.elems.as_slice())
    }

    pub fn fill_from_slice(&mut self, src: &[Float]) {
        for i in 0..self.elems.len() {
            self.elems[i] = src[i]
        }
    }

    pub fn fill_rand(&mut self, range: Range<Float>) {
        let mut rng = rand::thread_rng();
        let between = Uniform::from(range);
        self.elems
            .iter_mut()
            .for_each(|i| *i = between.sample(&mut rng));
    }

    pub fn from_dot_of(a: &Self, b: &Self) -> Self {
        let mut new = Self::new(a.rows, b.cols);
        new.into_dot_of(a, b);
        new
    }

    pub fn into_dot_of(&mut self, a: &Self, b: &Self) {
        assert!(a.cols == b.rows, "On line {}", line!());
        assert!(self.rows == a.rows, "On line {}", line!());
        assert!(self.cols == b.cols, "On line {}", line!());
        let n = a.cols;
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set_at(i, j, 0.);
                for k in 0..n {
                    self.apply_at(i, j, |v| v + a.get_at(i, k) * b.get_at(k, j));
                }
            }
        }
    }

    pub fn sum(&mut self, other: &Self) {
        assert!(self.cols == other.cols, "On line {}", line!());
        assert!(self.rows == other.rows, "On line {}", line!());
        self.elems = self
            .elems
            .iter_mut()
            .zip(&other.elems)
            .map(|(a, b)| *a + b)
            .collect();
    }

    pub fn apply_all<F: Fn(Float) -> Float>(&mut self, f: F) {
        self.elems.iter_mut().for_each(|e| *e = f(*e))
    }

    pub fn set_at(&mut self, r: usize, c: usize, v: Float) {
        self.elems[r * self.stride + c] = v;
    }

    pub fn get_at(&self, r: usize, c: usize) -> Float {
        self.elems[r * self.stride + c]
    }

    pub fn apply_at<F: Fn(Float) -> Float>(&mut self, r: usize, c: usize, f: F) {
        self.set_at(r, c, f(self.get_at(r, c)))
    }

    pub fn get_row(&self, r: usize) -> Mat {
        let mut new = Mat::new(1, self.cols);
        let l = self.cols * r;
        new.elems = self.elems[l..l + self.cols].into();
        new
    }
}

impl fmt::Display for Mat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        for row in self.elems.chunks(self.cols) {
            s.push_str("    ");
            row.iter()
                .for_each(|e| s.push_str(&format!("{e:.l$}  ", l = self.fmt_mantissa)));
            s.push('\n');
        }

        write!(f, "[\n{s}]",)
    }
}
