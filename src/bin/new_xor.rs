use wiggle_ml::sample::arr_sample;
use wiggle_ml::*;

#[derive(Clone)]
struct Xor {
    w1: Mat,
    b1: Mat,
    w2: Mat,
    b2: Mat,
}

impl Xor {
    fn new() -> Self {
        Self {
            w1: Mat::new(2, 2).randomize(),
            b1: Mat::new(1, 2).randomize(),
            w2: Mat::new(2, 1).randomize(),
            b2: Mat::new(1, 1).randomize(),
        }
    }

    fn forward(&self, x: &Mat) -> Float {
        let a1 = x.get_dot(&self.w1).sum(&self.b1).all(sigmoid);
        let a2 = a1.get_dot(&self.w2).sum(&self.b2).all(sigmoid);
        a2.get_at(0, 0)
    }

    fn cost(&self, ti: &Mat, to: &Mat) -> Float {
        let mut result = 0.;
        for i in 0..ti.rows {
            let o = to.get_at(i, 0);
            let y = self.forward(&ti.get_row(i));
            let d = y - o;
            result += d * d;
        }

        result / ti.rows as Float
    }

    fn finite_diff(&mut self, g: &mut Self, ti: &Mat, to: &Mat, eps: Float) {
        let c = self.cost(ti, to);
        let mut saved: Float;

        for i in 0..self.w1.rows {
            for j in 0..self.w1.cols {
                saved = self.w1.get_at(i, j);
                self.w1.set_at(i, j, saved + eps);
                g.w1.set_at(i, j, (self.cost(ti, to) - c) / eps);
                self.w1.set_at(i, j, saved);
            }
        }

        for i in 0..self.b1.rows {
            for j in 0..self.b1.cols {
                saved = self.b1.get_at(i, j);
                self.b1.set_at(i, j, saved + eps);
                g.b1.set_at(i, j, (self.cost(ti, to) - c) / eps);
                self.b1.set_at(i, j, saved);
            }
        }

        for i in 0..self.w2.rows {
            for j in 0..self.w2.cols {
                saved = self.w2.get_at(i, j);
                self.w2.set_at(i, j, saved + eps);
                g.w2.set_at(i, j, (self.cost(ti, to) - c) / eps);
                self.w2.set_at(i, j, saved);
            }
        }

        for i in 0..self.b2.rows {
            for j in 0..self.b2.cols {
                saved = self.b2.get_at(i, j);
                self.b2.set_at(i, j, saved + eps);
                g.b2.set_at(i, j, (self.cost(ti, to) - c) / eps);
                self.b2.set_at(i, j, saved);
            }
        }
    }

    fn learn(&mut self, g: &Self, rate: Float) {
        for i in 0..self.w1.rows {
            for j in 0..self.w1.cols {
                self.w1.apply_at(i, j, |v| v - g.w1.get_at(i, j) * rate);
            }
        }

        for i in 0..self.w1.rows {
            for j in 0..self.w1.cols {
                self.w1.apply_at(i, j, |v| v - g.w1.get_at(i, j) * rate);
            }
        }

        for i in 0..self.b1.rows {
            for j in 0..self.b1.cols {
                self.b1.apply_at(i, j, |v| v - g.b1.get_at(i, j) * rate);
            }
        }

        for i in 0..self.w2.rows {
            for j in 0..self.w2.cols {
                self.w2.apply_at(i, j, |v| v - g.w2.get_at(i, j) * rate);
            }
        }

        for i in 0..self.b2.rows {
            for j in 0..self.b2.cols {
                self.b2.apply_at(i, j, |v| v - g.b2.get_at(i, j) * rate);
            }
        }
    }
}

const D: (&str, [f32; 12]) = arr_sample::OR;

fn main() {
    let mut x = Mat::new(1, 2);
    x.set_at(0, 0, 0.);
    x.set_at(0, 1, 1.);

    let ti = Mat::new(4, 2).submat_from(D.1, 3);
    let to = Mat::new(4, 1).submat_from(&D.1[2..], 3);

    let mut m = Xor::new();
    let mut g = m.clone();
    let eps = 1e-1;
    let rate = 1e-1;

    for _ in 0..20000 {
        m.finite_diff(&mut g, &ti, &to, eps);
        m.learn(&g, rate);
        println!("{c}", c = m.cost(&ti, &to));
    }

    let mut a0 = Mat::new(1, 2);
    for i in 0..2 {
        for j in 0..2 {
            a0.set_at(0, 0, i as Float);
            a0.set_at(0, 1, j as Float);
            let y = m.forward(&a0);
            println!("{i} {z} {j} = {y}", z = D.0);
        }
    }
}
