pub mod sample;
use rand::{
    self,
    distributions::{Distribution, Uniform},
};
use std::{
    fmt::{self, Display},
    ops::Range,
};

extern "C" {
    fn rand() -> u32;
    fn srand(seed: u32);
}

pub type Float = f32;

pub fn set_seed(seed: u32) {
    unsafe { srand(seed) }
}

pub fn gen_rand_fixed() -> Float {
    unsafe { rand() as Float / u32::MAX as Float }
}

pub fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone, Debug, Default)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub es: Vec<Float>,

    pub fmt_mantissa: usize,
    pub fmt_padding: usize,
    pub fmt_name: Option<String>,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            es: vec![Float::default(); cols * rows],
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

    pub fn submat_from<S: AsRef<[Float]>>(mut self, src: S, stride: usize) -> Self {
        self.fill_from_submat(src, stride);
        self
    }

    pub fn filled(mut self, v: Float) -> Self {
        self.es.fill(v);
        self
    }

    pub fn filled_from(mut self, other: &Self) -> Self {
        self.fill_from(other);
        self
    }

    pub fn with_all<F: FnMut(Float) -> Float>(mut self, f: F) -> Self {
        self.apply_all(f);
        self
    }

    pub fn randomized_range(mut self, range: Range<Float>) -> Self {
        self.randomize_range(range);
        self
    }

    pub fn randomized(mut self) -> Self {
        self.randomize_range(0.0..1.0);
        self
    }

    pub fn randomized_fixed(mut self) -> Self {
        self.randomize_fixed();
        self
    }

    pub fn summed_with(mut self, other: &Self) -> Self {
        self.apply_sum(other);
        self
    }

    pub fn fill(&mut self, v: Float) {
        self.es.fill(v);
    }

    pub fn randomize_range(&mut self, range: Range<Float>) {
        let mut rng = rand::thread_rng();
        let between = Uniform::from(range);
        self.es
            .iter_mut()
            .for_each(|i| *i = between.sample(&mut rng));
    }

    pub fn randomize(&mut self) {
        self.randomize_range(0.0..1.0);
    }

    pub fn randomize_fixed(&mut self) {
        self.es.iter_mut().for_each(|i| *i = gen_rand_fixed());
    }

    pub fn fill_from_submat<S: AsRef<[Float]>>(&mut self, src: S, stride: usize) {
        let src = src.as_ref().iter();
        let mut j = 0;
        if self.cols == 1 {
            for e in src.step_by(stride) {
                self.es[j] = *e;
                j += 1;
            }
        } else {
            for (i, e) in src.enumerate() {
                if (i + 1) % stride == 0 && j % self.cols == 0 {
                    continue;
                }

                self.es[j] = *e;
                j += 1;
            }
        }
    }

    pub fn fill_from(&mut self, other: &Self) {
        self.es.copy_from_slice(other.es.as_slice());
    }

    pub fn get_at(&self, r: usize, c: usize) -> Float {
        self.es[r * self.cols + c]
    }

    pub fn get_mut_at(&mut self, r: usize, c: usize) -> &mut Float {
        &mut self.es[r * self.cols + c]
    }

    pub fn apply_at<F: FnMut(Float) -> Float>(&mut self, r: usize, c: usize, mut f: F) {
        self.set_at(r, c, f(self.get_at(r, c)))
    }

    pub fn apply_all<F: FnMut(Float) -> Float>(&mut self, mut f: F) {
        self.es.iter_mut().for_each(|e| *e = f(*e))
    }

    pub fn set_at(&mut self, r: usize, c: usize, v: Float) {
        self.es[r * self.cols + c] = v;
    }

    pub fn apply_sum(&mut self, other: &Self) {
        self.es
            .iter_mut()
            .enumerate()
            .for_each(|(i, a)| *a += other.es[i]);
    }

    pub fn sum(&self, other: &Self) -> Self {
        self.clone().summed_with(other)
    }

    pub fn dot(&self, other: &Self) -> Self {
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

    pub fn row(&self, r: usize) -> Self {
        let mut new = Self::new(1, self.cols);
        let l = self.cols * r;
        new.es = self.es[l..l + self.cols].into();

        new
    }
}

#[derive(Debug, Clone, Default)]
pub struct NN {
    count: usize,
    ws: Vec<Mat>,
    bs: Vec<Mat>,
    a: Vec<Mat>,

    fmt_inputs: bool,
}

impl NN {
    pub fn new(arch: &[usize]) -> Self {
        let arch_count = arch.len();
        let count = arch_count - 1;
        let mut ws = Vec::with_capacity(count);
        let mut bs = Vec::with_capacity(count);
        let mut a = Vec::with_capacity(count + 1);

        a.push(Mat::new(1, arch[0]).fmt_name("a0: input").fmt_pad(4));
        for i in 1..arch_count {
            ws.push(
                Mat::new(a[i - 1].cols, arch[i])
                    .fmt_name(format!("ws{i}"))
                    .fmt_pad(4),
            );
            bs.push(Mat::new(1, arch[i]).fmt_name(format!("bs{i}")).fmt_pad(4));
            a.push(Mat::new(1, arch[i]).fmt_name(format!("a{i}")).fmt_pad(4));
        }
        a[count].fmt_name = Some(format!("a{count}: output"));

        Self {
            count,
            ws,
            bs,
            a,
            fmt_inputs: false,
        }
    }

    pub fn fmt_inputs(mut self, enable: bool) -> Self {
        self.fmt_inputs = enable;
        self
    }

    pub fn randomized_range(mut self, range: Range<Float>) -> Self {
        self.randomize_range(range);
        self
    }

    pub fn randomized(mut self) -> Self {
        self.randomize();
        self
    }

    pub fn randomized_fixed(mut self) -> Self {
        self.randomize_fixed();
        self
    }

    pub fn filled_with(mut self, v: Float) -> Self {
        self.fill(v);
        self
    }

    pub fn with_input(mut self, inp: &Mat) -> Self {
        self.set_input(inp);
        self
    }

    pub fn randomize_range(&mut self, range: Range<Float>) {
        let mut rng = rand::thread_rng();
        let between = Uniform::from(range);
        self.ws
            .iter_mut()
            .zip(self.bs.iter_mut())
            .for_each(|(wm, bm)| {
                wm.apply_all(|_| between.sample(&mut rng));
                bm.apply_all(|_| between.sample(&mut rng));
            });
    }

    pub fn randomize(&mut self) {
        self.randomize_range(0.0..1.0)
    }

    pub fn randomize_fixed(&mut self) {
        self.ws.iter_mut().for_each(|m| m.randomize_fixed());
        self.bs.iter_mut().for_each(|m| m.randomize_fixed());
    }

    pub fn fill(&mut self, v: Float) {
        self.ws.iter_mut().for_each(|m| m.apply_all(|_| v));
        self.bs.iter_mut().for_each(|m| m.apply_all(|_| v));
    }

    pub fn set_input(&mut self, inp: &Mat) {
        self.get_mut_input().fill_from(inp);
    }

    pub fn get_ref_input(&self) -> &Mat {
        &self.a[0]
    }

    pub fn get_mut_input(&mut self) -> &mut Mat {
        &mut self.a[0]
    }

    pub fn get_ref_output(&self) -> &Mat {
        &self.a[self.count]
    }

    pub fn get_mut_output(&mut self) -> &mut Mat {
        &mut self.a[self.count]
    }

    pub fn forward(&mut self) {
        for i in 0..self.count {
            let out = self.a[i]
                .dot(&self.ws[i])
                .sum(&self.bs[i])
                .with_all(sigmoid);
            self.a[i + 1].fill_from(&out);
        }
    }

    pub fn forward_with(&mut self, input: &Mat) {
        self.set_input(input);
        self.forward();
    }

    pub fn cost(&mut self, ti: &Mat, to: &Mat) -> Float {
        let mut c = 0.;
        for i in 0..ti.rows {
            let x = ti.row(i);
            let y = to.row(i);
            self.set_input(&x);
            self.forward();
            for j in 0..to.cols {
                let d = self.get_ref_output().get_at(0, j) - y.get_at(0, j);
                c += d * d;
            }
        }

        c / ti.rows as Float
    }

    pub fn finite_diff(&mut self, g: &mut Self, ti: &Mat, to: &Mat, eps: Float) {
        let c = self.cost(ti, to);
        for i in 0..self.count {
            for j in 0..self.ws[i].rows {
                for k in 0..self.ws[i].cols {
                    let saved = self.ws[i].get_at(j, k);
                    self.ws[i].set_at(j, k, saved + eps);
                    g.ws[i].set_at(j, k, (self.cost(ti, to) - c) / eps);
                    self.ws[i].set_at(j, k, saved);
                }
            }

            for j in 0..self.bs[i].rows {
                for k in 0..self.bs[i].cols {
                    let saved = self.bs[i].get_at(j, k);
                    self.bs[i].set_at(j, k, saved + eps);
                    g.bs[i].set_at(j, k, (self.cost(ti, to) - c) / eps);
                    self.bs[i].set_at(j, k, saved);
                }
            }
        }
    }

    pub fn backprop(&mut self, g: &mut Self, ti: &Mat, to: &Mat) {
        assert!(ti.rows == to.rows);
        assert!(self.get_ref_output().cols == to.cols);
        g.fill(0.);
        let n = ti.rows;

        // i - current sample
        // l - current layer
        // j - current activation
        // k - previous activation

        for i in 0..n {
            self.set_input(&ti.row(i));
            self.forward();

            for j in 0..=self.count {
                g.a[j].apply_all(|_| 0.);
            }

            for j in 0..to.cols {
                g.get_mut_output().set_at(
                    0,
                    j,
                    self.get_ref_output().get_at(0, j) - to.get_at(i, j),
                );
            }

            for l in (1..=self.count).rev() {
                for j in 0..self.a[l].cols {
                    let a = self.a[l].get_at(0, j);
                    let da = g.a[l].get_at(0, j);
                    g.bs[l - 1].apply_at(0, j, |v| v + (2. * da * a * (1. - a)));
                    for k in 0..self.a[l - 1].cols {
                        // j - weight matrix col
                        // k - weight matrix row
                        let w = self.ws[l - 1].get_at(k, j);
                        let pa = self.a[l - 1].get_at(0, k);
                        g.ws[l - 1].apply_at(k, j, |v| v + (2. * da * a * (1. - a) * pa));
                        g.a[l - 1].apply_at(0, k, |v| v + (2. * da * a * (1. - a) * w));
                    }
                }
            }
        }

        for i in 0..g.count {
            for j in 0..g.ws[i].rows {
                for k in 0..g.ws[i].cols {
                    g.ws[i].apply_at(j, k, |v| v / n as Float);
                }
            }

            for j in 0..g.bs[i].rows {
                for k in 0..g.bs[i].cols {
                    g.bs[i].apply_at(j, k, |v| v / n as Float);
                }
            }
        }
    }

    pub fn learn(&mut self, g: &Self, rate: Float) {
        self.ws.iter_mut().zip(g.ws.iter()).for_each(|(m, gm)| {
            m.es
                .iter_mut()
                .zip(gm.es.iter())
                .for_each(|(e, ge)| *e -= ge * rate)
        });

        self.bs.iter_mut().zip(g.bs.iter()).for_each(|(m, gm)| {
            m.es
                .iter_mut()
                .zip(gm.es.iter())
                .for_each(|(e, ge)| *e -= ge * rate)
        });
    }
}

impl Display for Mat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        let pad = (0..self.fmt_padding).map(|_| ' ').collect::<String>();
        for row in self.es.chunks(self.cols) {
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
            for ((w, b), a) in self
                .ws
                .iter()
                .zip(self.bs.iter())
                .zip(self.a.iter().skip(1))
            {
                s.push_str(&format!("{a}\n{w}\n{b}\n"));
            }
        } else {
            for (w, b) in self.ws.iter().zip(self.bs.iter()) {
                s.push_str(&format!("{w}\n{b}\n"));
            }
        }

        write!(f, "[\n{s}]")
    }
}
