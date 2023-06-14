use wiggle_ml::*;

#[derive(Default)]
struct Xor {
    // Input
    a0: Mat,

    // First layer
    w1: Mat,
    b1: Mat,
    a1: Mat, // output

    // Second layer
    w2: Mat,
    b2: Mat,
    a2: Mat, // output
}

impl Xor {
    fn new() -> Self {
        Self {
            a0: Mat::new(1, 2),
            w1: Mat::new_rand(2, 2),
            b1: Mat::new_rand(1, 2),
            a1: Mat::new(1, 2),
            w2: Mat::new_rand(2, 1),
            b2: Mat::new_rand(1, 1),
            a2: Mat::new(1, 1),
        }
    }

    fn forward(&mut self) {
        // Passing input through the first layer
        // out = f(x * w + b)
        self.a1 = Mat::from_dot_of(&self.a0, &self.w1);
        self.a1.sum(&self.b1);
        self.a1.apply_all(|e| sigmoid(e)); // activation function

        // Passing input through the second layer
        self.a2 = Mat::from_dot_of(&self.a1, &self.w2);
        self.a2.sum(&self.b2);
        self.a2.apply_all(|e| sigmoid(e));
    }

    fn cost(&mut self, ti: &Mat, to: &Mat) -> Float {
        assert!(ti.rows == to.rows, "On line {}", line!());
        assert!(to.cols == self.a2.cols, "On line {}", line!());
        let n = ti.rows;
        let mut c = 0.;
        for i in 0..n {
            // input matrix
            let x = ti.get_row(i);
            // output matrix
            let y = to.get_row(i);
            // load input into the model
            self.a0.fill_from(&x);
            // push through the layers
            self.forward();
            // calculate difference
            for j in 0..to.cols {
                let d = self.a2.get_at(0, j) - to.get_at(0, j);
                c += d * d;
            }
        }

        c / n as Float
    }

    fn finite_diff(&mut self, eps: Float, ti: &Mat, to: &Mat) -> Self {
        let mut g = Self::new();
        let mut saved: Float;
        let c = self.cost(ti, to);

        for i in 0..self.w1.rows {
            for j in 0..self.w1.cols {
                saved = self.w1.get_at(i, j);
                self.w1.apply_at(i, j, |v| v + eps);
                g.w1.set_at(i, j, (self.cost(ti, to) - c) / eps);
                self.w1.set_at(i, j, saved);
            }
        }

        for i in 0..self.b1.rows {
            for j in 0..self.b1.cols {
                saved = self.b1.get_at(i, j);
                self.b1.apply_at(i, j, |v| v + eps);
                g.b1.set_at(i, j, (self.cost(ti, to) - c) / eps);
                self.b1.set_at(i, j, saved);
            }
        }

        for i in 0..self.w2.rows {
            for j in 0..self.w2.cols {
                saved = self.w2.get_at(i, j);
                self.w2.apply_at(i, j, |v| v + eps);
                g.w2.set_at(i, j, (self.cost(ti, to) - c) / eps);
                self.w2.set_at(i, j, saved);
            }
        }

        for i in 0..self.b2.rows {
            for j in 0..self.b2.cols {
                saved = self.b2.get_at(i, j);
                self.b2.apply_at(i, j, |v| v + eps);
                g.b2.set_at(i, j, (self.cost(ti, to) - c) / eps);
                self.b2.set_at(i, j, saved);
            }
        }

        g
    }

    fn learn(&mut self, g: Xor, rate: Float) {
        for i in 0..self.w1.rows {
            for j in 0..self.w1.cols {
                self.w1.apply_at(i, j, |v| v - rate * g.w1.get_at(i, j))
            }
        }

        for i in 0..self.b1.rows {
            for j in 0..self.b1.cols {
                self.b1.apply_at(i, j, |v| v - rate * g.b1.get_at(i, j))
            }
        }

        for i in 0..self.w2.rows {
            for j in 0..self.w2.cols {
                self.w2.apply_at(i, j, |v| v - rate * g.w2.get_at(i, j))
            }
        }

        for i in 0..self.b2.rows {
            for j in 0..self.b2.cols {
                self.b2.apply_at(i, j, |v| v - rate * g.b2.get_at(i, j))
            }
        }
    }
}

#[rustfmt::skip]
const TD: [Float; 12] = [
    0., 0., 0.,
    0., 1., 1.,
    1., 0., 1.,
    1., 1., 0.
];

fn main() {
    // Setup training data
    //
    // Input
    //
    let stride = 3; // size of one sample
    let n = TD.len() / stride; // amount of samples
    let mut ti = Mat::new(n, 2);
    ti.fill_sub_from_slice(&TD, stride);
    //
    // Expected output
    //
    let mut to = Mat::new(n, 1);
    to.fill_sub_from_slice(&TD[2..], stride);

    let mut m = Xor::new();
    let mut g: Xor;
    let eps = 1e-1;
    let rate = 1e-1;

    println!("cost = {c}", c = m.cost(&ti, &to));
    for _ in 0..10000 {
        g = m.finite_diff(eps, &ti, &to);
        m.learn(g, rate);
        println!("cost = {c}", c = m.cost(&ti, &to));
    }

    println!("-------------------------------");
    for i in 0..2 {
        for j in 0..2 {
            m.a0.set_at(0, 0, i as Float);
            m.a0.set_at(0, 1, j as Float);
            m.forward();
            println!("{i} ^ {j} = {y}", y = m.a2.get_at(0, 0));
        }
    }
}
