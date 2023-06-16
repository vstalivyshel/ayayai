use rand::{
    self,
    distributions::{Distribution, Uniform},
};
use std::{
    fmt::{self, Display},
    ops::Range,
};

pub type Float = f32;

pub fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone, Debug, Default)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub elems: Vec<Float>,
    pub fmt_mantissa: usize,
    pub fmt_padding: usize,
    pub fmt_name: Option<String>,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            elems: vec![Float::default(); colst * rows],
            fmt_mantissa: 7,
            fmt_padding: 0,
            fmt_name: None,
        }
    }

    pub fn fmt_pad(mut self, n: usize) -> Self {
        self.fmt_padding = n;
        self
    }

    pub fn fmt_mantissa(mut self, n: usize) -> Self {
        self.fmt_mantissa = n;
        self
    }

    pub fn fmt_name<S: Display>(mut self, n: S) -> Self {
        self.fmt_name = Some(n.to_string());
        self
    }

    pub fn fill(mut self, v: Float) -> Self {
        self.elems.fill(v);
        self
    }

    pub fn all<F: FnMut(Float) -> Float>(mut self, f: F) -> Self {
        self.apply_all(f);
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

    pub fn apply_fill_from(&mut self, other: &Self) {
        self.elems.copy_from_slice(other.elems.as_slice());
    }

    pub fn apply_at<F: FnMut(Float) -> Float>(&mut self, r: usize, c: usize, mut f: F) {
        self.set_at(r, c, f(self.get_at(r, c)))
    }

    pub fn apply_all<F: FnMut(Float) -> Float>(&mut self, mut f: F) {
        self.elems.iter_mut().for_each(|e| *e = f(*e))
    }

    pub fn set_at(&mut self, r: usize, c: usize, v: Float) {
        self.elems[r * self.cols + c] = v;
    }
}

#[derive(Debug, Clone, Default)]
pub struct NN {
    count: usize,
    w: Vec<Mat>,
    b: Vec<Mat>,
    a: Vec<Mat>,

    fmt_inputs: bool,
}

impl NN {
    pub fn new(arch: &[usize]) -> Self {
        let arch_count = arch.len();
        let count = arch_count - 1;
        let mut w = Vec::with_capacity(count);
        let mut b = Vec::with_capacity(count);
        let mut a = Vec::with_capacity(count + 1);

        a.push(Mat::new(1, arch[0]).fmt_name("a0: input").fmt_pad(4));
        for i in 1..arch_count {
            w.push(
                Mat::new(a[i - 1].cols, arch[i])
                    .fmt_name(format!("w{i}"))
                    .fmt_pad(4),
            );
            b.push(Mat::new(1, arch[i]).fmt_name(format!("b{i}")).fmt_pad(4));
            a.push(Mat::new(1, arch[i]).fmt_name(format!("a{i}")).fmt_pad(4));
        }
        a[count].fmt_name = Some(format!("a{count}: output"));

        Self {
            count,
            w,
            b,
            a,
            fmt_inputs: false,
        }
    }

    pub fn fmt_inputs(mut self, enable: bool) -> Self {
        self.fmt_inputs = enable;
        self
    }

    pub fn forward(&mut self) {
        for i in 0..self.count {
            let out = self.a[i].get_dot(&self.w[i]).sum(&self.b[i]).all(sigmoid);
            self.a[i + 1].apply_fill_from(&out);
        }
    }

    pub fn get_mut_input(&mut self) -> &mut Mat {
        &mut self.a[0]
    }

    pub fn set_input(&mut self, inp: &Mat) {
        self.get_mut_input().apply_fill_from(inp);
    }

    pub fn get_ref_output(&self) -> &Mat {
        &self.a[self.count]
    }

    pub fn get_output(&self) -> Mat {
        self.get_ref_output().clone().fmt_name("output")
    }

    pub fn cost(&mut self, ti: &Mat, to: &Mat) -> Float {
        let mut c = 0.;
        for i in 0..ti.rows {
            let x = ti.get_row(i);
            let y = to.get_row(i);
            self.set_input(&x);
            self.forward();
            for j in 0..to.cols {
                let d = self.get_ref_output().get_at(0, j) - y.get_at(0, j);
                c += d * d;
            }
        }

        c / ti.rows as Float
    }

    pub fn finite_diff(&mut self, eps: Float, ti: &Mat, to: &Mat) -> Self {
        let c = self.cost(ti, to);
        let mut g = self.clone();
        for i in 0..self.count {
            for j in 0..self.w[i].rows {
                for k in 0..self.w[i].cols {
                    let saved = self.w[i].get_at(j, k);
                    self.w[i].set_at(j, k, saved + eps);
                    g.w[i].set_at(j, k, (self.cost(ti, to) - c) / eps);
                    self.w[i].set_at(j, k, saved);
                }
            }

            for j in 0..self.b[i].rows {
                for k in 0..self.b[i].cols {
                    let saved = self.b[i].get_at(j, k);
                    self.b[i].set_at(j, k, saved + eps);
                    g.b[i].set_at(j, k, (self.cost(ti, to) - c) / eps);
                    self.b[i].set_at(j, k, saved);
                }
            }
        }

        g
    }

    pub fn apply_diff(&mut self, g: Self, rate: Float) {
        self.w.iter_mut().zip(g.w.iter()).for_each(|(m, gm)| {
            m.elems
                .iter_mut()
                .zip(gm.elems.iter())
                .for_each(|(e, ge)| *e -= ge * rate)
        });

        self.b.iter_mut().zip(g.b.iter()).for_each(|(m, gm)| {
            m.elems
                .iter_mut()
                .zip(gm.elems.iter())
                .for_each(|(e, ge)| *e -= ge * rate)
        });
    }

    pub fn randomize_range(mut self, range: Range<Float>) -> Self {
        let mut rng = rand::thread_rng();
        let between = Uniform::from(range);
        self.w
            .iter_mut()
            .zip(self.b.iter_mut())
            .for_each(|(wm, bm)| {
                wm.apply_all(|_| between.sample(&mut rng));
                bm.apply_all(|_| between.sample(&mut rng));
            });
        self
    }

    pub fn randomize(self) -> Self {
        self.randomize_range(0.0..1.0)
    }
}

impl Display for Mat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        let pad = (0..self.fmt_padding).map(|_| ' ').collect::<String>();
        for row in self.elems.chunks(self.cols) {
            s.push_str(&format!("    {pad}"));
            for e in row {
                s.push_str(&format!("{e:.l$}  ", l = self.fmt_mantissa));
            }
            s.push('\n');
        }

        write!(
            f,
            "{pad}{n}[\n{s}{pad}]",
            n = self
                .fmt_name
                .as_ref()
                .map(|name| format!("{name} = "))
                .unwrap_or(String::new())
        )
    }
}

impl Display for NN {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        s.push_str("");

        if self.fmt_inputs {
            s.push_str(self.a[0].to_string().as_str());
            s.push('\n');
            for ((w, b), a) in self.w.iter().zip(self.b.iter()).zip(self.a.iter().skip(1)) {
                s.push_str(&format!("{a}\n{w}\n{b}\n"));
            }
        } else {
            for (w, b) in self.w.iter().zip(self.b.iter()) {
                s.push_str(&format!("{w}\n{b}\n"));
            }
        }

        write!(f, "[\n{s}]")
    }
}
