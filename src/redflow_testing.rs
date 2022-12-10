extern crate round;
use round::round;
use rand::Rng;
use std::env;

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
pub fn test_sigmoid(x:f64) -> f64{
    let mut ans = -x;
    ans = ans.exp();
    ans = 1.0 + ans;
    ans = 1.0 / ans;
    return ans
}

//Used for back propagation
pub fn test_sigmoid_derivative(x:f64) -> f64{
    let mut ans = 1.0 - x;
    ans = x * ans;
    return ans
}

//Creates the weights
//For now you HAVE to pass a vec<vec<vec<f64 vector, even if its only got 1 layer, which would look like: let var = vec![vec![1.0,2.0],vec![3.0,4.0]];
pub fn test_generate_weights_and_bias(decimal_place:i32,input:&Vec<Vec<Vec<f64>>>) -> (Vec<Vec<Vec<f64>>>,Vec<Vec<Vec<f64>>>) {
    let mut rng = rand::thread_rng();
    let mut weights = clear_inputs(input);
    let mut bias = clear_inputs(&input);
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
                randval = round(randval,decimal_place);
                weights[layer][row][val] = randval;
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
    //iterates in a process that goes: layer, row, value
    for _x in input.iter() {
        for _x in input[0].iter(){
            for _x in input[0][0].iter(){
                randval = rng.gen_range(0.0..1.0);
                randval = round(randval,decimal_place);
                bias[layer][row][val] = randval;
                val += 1;
            }
        row += 1;
        val = 0;
        }
    layer += 1;
    row = 0;
    }
    return (weights, bias)
}

pub fn test_forward_propagate(decimal_place:i32,inputs:&Vec<Vec<Vec<f64>>>,hidden_wb:&Vec<Vec<Vec<Vec<f64>>>>,outputs:&Vec<Vec<Vec<f64>>>,output_wb:&Vec<Vec<Vec<Vec<f64>>>>) -> (Vec<Vec<Vec<f64>>>,Vec<Vec<Vec<f64>>>) {
    //env::set_var("RUST_BACKTRACE", "1");
    let mut hidden = clear_inputs(&inputs);
    let mut output = clear_inputs(&outputs);
    let mut ans:f64 = 0.0;
    let mut layer = 0;
    let mut row = 0;
    let mut val = 0;
    //iterates in a process that goes: layer, row, value (Still not sorry I ctrl c and ctrl v)
    //Hidden layer
    for _x in inputs.iter(){
        row = 0;
        for _y in inputs[0].iter(){
            for _z in inputs[0][0].iter(){
                ans = (hidden_wb[0][layer][row][val] * inputs[layer][row][val]) + hidden_wb[1][layer][row][val];
                ans = sigmoid(ans);
                ans = round(ans,decimal_place);
                hidden[layer][row][val] = ans;
                val += 1;
            }
        row += 1;
        val = 0;
        }
    layer += 1;
    }
    layer = 0;
    row = 0;
    val = 0;
    ans = 0.0;
    for _x in outputs.iter(){
        for _y in outputs[0].iter(){
            for _x in outputs[0][0].iter(){
                ans = (output_wb[0][layer][row][val] * outputs[layer][row][val]) + output_wb[1][layer][row][val];
                ans = sigmoid(ans);
                ans = round(ans,decimal_place);
                output[layer][row][val] = ans;
                val += 1;
            }
        val = 0;
        row += 1;
        }
    layer += 1;
    row = 0;
    }
    return (hidden,output)
}

pub fn test_backward_propagate(decimal_place:i32,hidden_results:&Vec<Vec<Vec<f64>>>,hidden_wb:&Vec<Vec<Vec<Vec<f64>>>>,output_wb:&Vec<Vec<Vec<Vec<f64>>>>,forward_outputs:&Vec<Vec<Vec<f64>>>,original_outputs:&Vec<Vec<Vec<f64>>>){
    //Model variabbles
    let mut hidden_gradients = clear_inputs(&hidden_results);
    let mut hidden_delta = hidden_gradients.clone();
    let mut output_gradient:f64;
    let mut output_delta = output_gradients.clone();
    //Error variables
    let mut errors = output_gradients.clone();
    let mut error = 0.0;
    //Used for iteration
    let mut ans:f64;
    let mut layer = 0;
    let mut row = 0;
    let mut val = 0;
    let mut hidden_ans:f64;
    let mut output_ans:f64;
    //Temp vectorss
    let mut hidden_temp = hidden_gradients.clone();
    let mut output_temp = output_gradients.clone();

    for _x in forward_outputs.iter(){
        for _y in forward_outputs[0].iter(){
            for _x in forward_outputs[0][0].iter(){
                //Kind of obvious but this calculates the error
                error = original_outputs[layer][row][val] - forward_outputs[layer][row][val];
                error = round(error,decimal_place);
                errors[layer][row][val] = error;
                //Calculates the output layer gradient
                output_ans += sigmoid_derivative(forward_outputs[layer][row][val]);
                output_ans = round(output_ans,decimal_place);
                output_gradient = output_ans;
                //Finds the delta for the output layer
                output_ans = errors[layer][row][val] * output_gradient;
                output_delta[layer][row][val] = round(output_ans,decimal_place);
                //Hidden layer error
                hidden_ans = output_delta[layer][row][val] * output_wb[0][layer][row][val];
                val += 1;
            }
        row += 1;
        val = 0;
        }
    layer += 1;
    row = 0;
    }
    //println!("Layer, row, val: {} {} {}",layer,row,val); use to see if the resets can be removed between each iteration block
    layer = 0;
    println!("Errors: {:?}",errors);
    println!("Output gradient: {:?}",output_gradient);
    println!("Output deltas: {:?}",output_delta);
    for _x in hidden_results.iter(){
        for _y in hidden_results[0].iter(){
            for _z in hidden_results[0][0].iter(){
                //Calculates the hidden layer gradient
                hidden_ans = sigmoid_derivative(hidden_results[layer][row][val]);
                hidden_ans = round(hidden_ans,decimal_place);
                hidden_gradients[layer][row][val] = hidden_ans;
                val += 1;
            }
        row += 1;
        val = 0;
        }
    layer += 1;
    row = 0;
    }
    println!("Hidden gradients: {:?}",hidden_gradients);
}