use wiggle_ml::*;

#[derive(Default)]
struct Xor {
    // Input
    a0: Mat,

    // First layer
    w1: Mat,
    b1: Mat,
    a1: Mat,

    // Second layer
    w2: Mat,
    b2: Mat,
    a2: Mat,
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

    fn forward(&mut self) -> Float {
        // Passing input through the first layer
        // out = f(x * w + b)
        self.a1 = Mat::from_dot_of(&self.a0, &self.w1);
        self.a1.sum(&self.b1);
        self.a1.sigmoid(); // activation function

        // Passing input through the second layer
        self.a2 = Mat::from_dot_of(&self.a1, &self.w2);
        self.a2.sum(&self.b2);
        self.a2.sigmoid();

        self.a2.get_at(0, 0)
    }

    // fn cost(&self, ti: Mat, to: Mat) -> Float {
    //     assert!(ti.rows == to.rows);
    //     let n = ti.rows;
    // }
}

fn main() {
    let mut m = Xor::new();
	for i in 0..2 {
    	for j in 0..2 {
            m.a0.set_at(0, 0, i as Float);
            m.a0.set_at(0, 1, j as Float);
        	let y = m.forward();
            println!("{i} ^ {j} = {y}");
    	}
	}

}
