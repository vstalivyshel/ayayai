use wiggle_ml::*;

fn main() {
    let a = Mat::new_randrang(1, 2, 5.0..10.0);
    let b = Mat::new_fill(2, 2, 1.);
    let dot = Mat::from_dot_of(&a, &b);
    println!("{a}");
    println!("-------------------------------");
    println!("{b}");
    println!("-------------------------------");
    println!("{dot}");
}
