use wiggle_ai::*;

// Training data contains an input value and a expected output
const TRAINING_DATA: [[isize; 2]; 4] = [[0, 0], [1, 2], [2, 4], [3, 8]];

// Iterate through all of the data and compare expected output with the model's prediction.
fn cost(w: Float, b: Float) -> Float {
    let mut result = 0.0;

    for [input, output] in TRAINING_DATA {
        // Feed the data input to the model.
        // Basically filter that data through all the parameters (we have only one btw)
        let prediction = input as Float * w + b;
        // Compute the distance between expected result and the model's output
        let distance = prediction - output as Float;
        // Amplefy all the errors by finding square
        result += distance * distance;
    }

    // Find the average of the distance: 0 is perfect result, means that:
    // expected output == model's prediction
    result / TRAINING_DATA.len() as Float
}

fn main() {
    // This is out model with 1 parameter
    let mut w = rand::random::<Float>();
    let mut b = rand::random::<Float>();

    // With this magic value we will be able to tweek model's parameter
    // and control it's output
    let eps = 1e-3;
    // Another value, through which we can manupulate model's output
    let rate = 1e-3;
    // And another, which is a number of learning cycles
    let n = 600;

    // Actual learning process
    for _ in 0..n {
        let c = cost(w, b);
        let dw = (cost(eps + w, b) - c) / eps;
        let db = (cost(w, eps + b) - c) / eps;
        w -= dw * rate;
        b -= db * rate;
        println!("c: {c}; w: {w}; b: {b}", c = cost(w, b));
    }
    println!("-------------------------------");
    println!("w: {w}; b: {b}");
}
