use wiggle_ml::*;
const BITS: usize = 2; 

fn main() {
    let n = 1 << BITS;
    let rows = n * n;
    let mut ti = Mat::new(rows, 2 * BITS).fmt_name("ti");
    let mut to = Mat::new(rows, BITS + 1).fmt_name("to");

    for i in 0..ti.rows {
        let x = i / n;
        let y = i % n;
        let z = x + y;
        for j in 0..BITS {
            ti.set_at(i, j, ((x >> j) & 1) as f32);
            ti.set_at(i, j + BITS, ((y >> j) & 1) as f32);
            to.set_at(i, j, ((z >> j) & 1) as f32);
        }
        to.set_at(i, BITS, ((z >= n) as usize) as f32);
    }

    let mut nn = NN::new(&[2 * BITS, 4 * BITS, BITS + 1]).randomize_fixed();
    let mut g = nn.clone();
    let rate  = 1.;

    for i in 0..5 * 1000 {
        // nn.finite_diff(&mut g, &ti, &to, 1e-1);
        nn.backprop(&mut g, &ti, &to);
        nn.learn(&g, rate);
        println!("{i}: {c}", c = nn.cost(&ti, &to));
    }

    let mut fails = 0;
    for x in 0..n {
        for y in 0..n {
            let z = x + y;
            for j in 0..BITS {
                let inp = nn.get_mut_input();
                inp.set_at(0, j, ((x>>j) & 1) as f32);
                inp.set_at(0, j + BITS, ((y>>j) & 1) as f32);
            }
            nn.forward();
            if nn.get_ref_output().get_at(0, BITS) > 0.5 {
                if z < n {
                    println!("{x} + {y} = (OVERFLOW <> {z})");
                    fails += 1;
                }
            } else {
                let mut a = 0;
                for j in 0..BITS {
                    let bit = (nn.get_ref_output().get_at(0, j) > 0.5) as usize;
                    a |= bit<<j;
                }
                if z != a {
                    println!("{x} + {y} = ({z} <> {a})");
                    fails += 1;
                }
            }
        }
    }

    if fails == 0 {
        println!("Ok");
    }
}
