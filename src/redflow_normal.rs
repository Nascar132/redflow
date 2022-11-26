extern crate round;
use round::round;
use rand::Rng;

//Used only by the library
fn clear_inputs(invec:&Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
    let mut copyvec = invec.clone();
    let mut layer = 0;
    let mut row = 0;
    let mut val = 0;
    //iterates in a process that goes: layer, row, value
    for _x in invec.iter() {
        for _x in invec[0].iter(){
            for _x in invec[0][0].iter(){
                copyvec[layer][row][val] = 0.0;
                val += 1;
            }
        row += 1;
        val = 0;
        }
    layer += 1;
    row = 0;
    }
    return copyvec.to_vec()
}

//Forward propagation
pub fn sigmoid(x:f64) -> f64{
    let mut ans = -x;
    ans = ans.exp();
    ans = 1.0 + ans;
    ans = 1.0 / ans;
    return ans
}

//Used for back propagation
pub fn sigmoid_derivative(x:f64) -> f64{
    let mut ans = 1.0 - x;
    ans = x * ans;
    return ans
}

//Creates the weights
//For now you HAVE to pass a vec<vec<vec<f64 vector, even if its only got 1 layer, which would look like: let var = vec![vec![1.0,2.0],vec![3.0,4.0]];
pub fn generate_weights(input:&Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
    let mut rng = rand::thread_rng();
    let mut weights = clear_inputs(input);
    let mut randval:f64;
    let mut layer = 0;
    let mut row = 0;
    let mut val = 0;
    //iterates in a process that goes: layer, row, value
    //Yes I reused this code and I'm not sorry if its bad practice to
    for _x in input.iter() {
        for _x in input[0].iter(){
            for _x in input[0][0].iter(){
                randval = rng.gen_range(0.0..1.0);
                randval = round(randval,1);
                weights[layer][row][val] = randval;
                val += 1;
            }
        row += 1;
        val = 0;
        }
    layer += 1;
    row = 0;
    }
    return weights
}

pub fn generate_bias(input:&Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
    let mut rng = rand::thread_rng();
    let mut bias = clear_inputs(&input);
    let mut randval:f64;
    let mut layer = 0;
    let mut row = 0;
    let mut val = 0;
    //iterates in a process that goes: layer, row, value
    for _x in input.iter() {
        for _x in input[0].iter(){
            for _x in input[0][0].iter(){
                randval = rng.gen_range(0.0..1.0);
                randval = round(randval,1);
                bias[layer][row][val] = randval;
            }
        row += 1;
        val = 0;
        }
    layer += 1;
    row = 0;
    }
    return bias
}

pub fn forward_propagate(inputs:&Vec<Vec<Vec<f64>>>,weights:&Vec<Vec<Vec<f64>>>,bias:&Vec<Vec<Vec<f64>>>,outputs:&Vec<Vec<Vec<f64>>>,output_weights:&Vec<Vec<Vec<f64>>>,output_bias:&Vec<Vec<Vec<f64>>>){
    let mut hidden = clear_inputs(&inputs);
    let mut output = &hidden.clone();
    let mut layer = 0;
    let mut row = 0;
    let mut val = 0;
    //iterates in a process that goes: layer, row, value
    //Still not sorry I ctrl c and ctrl v
    for _x in inputs.iter(){
        for _x in inputs[0].iter(){
            for _x in inputs[0][0].iter(){
                let mut ans = (weights[layer][row][val] * inputs[layer][row][val]) + bias[layer][row][val];
                ans = sigmoid(ans);
                ans = round(ans,1);
                hidden[layer][row][val] = ans;
                val += 1;
            }
        row += 1;
        val = 0;
        }
    layer += 1;
    row = 0;
    }
    layer = 0;
    row = 0;
    val = 0;
    for _x in outputs.iter(){
        for _x in outputs[0].iter(){
            for _x in outputs[0][0].iter(){
                let mut ans = (output_weights[layer][row][val] * inputs[layer][row][val]) + output_bias[layer][row][val];
                ans = sigmoid(ans);
                ans = round(ans,1);
                output[layer][row][val] = ans;
                val += 1;
            }
        row += 1;
        val = 0;
        }
    layer += 1;
    row = 0;
    }
    println!("{:?}",output;
}