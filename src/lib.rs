use rand::{
    self,
    distributions::{Distribution, Uniform},
};
use std::{fmt, ops::Range};

pub type Float = f32;

pub fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

#[macro_export]
macro_rules! mat_print {
    ($($mat:ident),*) => {
        $(println!("{name} = {m}", name = stringify!($mat), m = $mat)); *
    };
    ($($a:ident.$b:ident),*) => {{
        $(let m = &$a.$b; mat_print!(m); )*
    }}
}

#[derive(Clone, Debug, Default)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub elems: Vec<Float>,
    pub fmt_mantissa: usize,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            elems: vec![Float::default(); cols * rows],
            fmt_mantissa: 7,
        }
    }

    pub fn fill(mut self, v: Float) -> Self {
        self.elems.fill(v);
        self
    }

    pub fn all<F: Fn(Float) -> Float>(mut self, f: F) -> Self {
        self.apply_all(|v| f(v));
        self
    }

    pub fn randomize_range(mut self, range: Range<Float>) -> Self {
        let mut rng = rand::thread_rng();
        let between = Uniform::from(range);
        self.elems
            .iter_mut()
            .for_each(|i| *i = between.sample(&mut rng));

        self
    }

    pub fn randomize(self) -> Self {
        self.randomize_range(0.0..1.0)
    }

    pub fn submat_from<S: AsRef<[Float]>>(mut self, src: S, stride: usize) -> Self {
        let src = src.as_ref().iter();
        let mut j = 0;

        if self.cols == 1 {
            for e in src.step_by(stride) {
                self.elems[j] = *e;
                j += 1;
            }

            return self;
        }

        for (i, e) in src.enumerate() {
            if (i + 1) % stride == 0 && j % self.cols == 0 {
                continue;
            }

            self.elems[j] = *e;
            j += 1;
        }

        self
    }

    pub fn sum(mut self, other: &Self) -> Self {
        assert!(self.rows == other.rows);
        assert!(self.cols == other.cols);
        self.elems = self
            .elems
            .iter_mut()
            .zip(other.elems.iter())
            .map(|(a, b)| *a + b)
            .collect::<Vec<Float>>();

        self
    }

    pub fn get_sum(&self, other: &Self) -> Self {
        let new = self.clone();
        new.sum(other)
    }

    pub fn get_dot(&self, other: &Self) -> Self {
        assert!(self.cols == other.rows);
        let mut new = Self::new(self.rows, other.cols);
        for i in 0..new.rows {
            for j in 0..new.cols {
                new.set_at(i, j, 0.);
                for k in 0..self.cols {
                    new.apply_at(i, j, |v| v + self.get_at(i, k) * other.get_at(k, j));
                }
            }
        }

        new
    }

    pub fn get_row(&self, r: usize) -> Self {
        let mut new = Self::new(1, self.cols);
        let l = self.cols * r;
        new.elems = self.elems[l..l + self.cols].into();

        new
    }

    pub fn get_at(&self, r: usize, c: usize) -> Float {
        self.elems[r * self.cols + c]
    }

    pub fn set_at(&mut self, r: usize, c: usize, v: Float) {
        self.elems[r * self.cols + c] = v;
    }

    pub fn apply_at<F: Fn(Float) -> Float>(&mut self, r: usize, c: usize, f: F) {
        self.set_at(r, c, f(self.get_at(r, c)))
    }

    pub fn apply_all<F: Fn(Float) -> Float>(&mut self, f: F) {
        self.elems.iter_mut().for_each(|e| *e = f(*e))
    }
}

impl fmt::Display for Mat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        for row in self.elems.chunks(self.cols) {
            s.push_str("    ");
            for e in row {
                s.push_str(&format!("{e:.l$}  ", l = self.fmt_mantissa));
            }
            s.push('\n');
        }

        write!(f, "[\n{s}]",)
    }
}
