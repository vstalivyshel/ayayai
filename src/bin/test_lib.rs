use wiggle_ml::*;

fn main() {
    let mut a = Mat::new_randrang(2, 2, 5.0..10.0);
    let b = Mat::new_fill(2, 2, 1.0);
    let dot = Mat::from_dot_of(&a, &b);
    println!("a = {a}");
    println!("-------------------------------");
    println!("b = {b}");
    println!("-------------------------------");
    println!("a * b = {dot}");
    println!("-------------------------------");
    a.sum(&b);
    println!("a + b = {a}");
}
