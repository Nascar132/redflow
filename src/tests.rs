use redflow::redflow_normal::*;

fn main(){
    let out = sigmoid(0.458);
    println!("{}",out);

    let inputs = vec![vec![
                    vec![1.0,2.0,3.0,4.0],
                    vec![5.0,6.0,7.0,8.0]],vec![
                        vec![9.0,10.0,11.0,12.0],
                        vec![13.0,14.0,15.0,16.0]]];

    let testinputs = vec![vec![vec![0.0,1.0,1.0,0.0],vec![0.0,1.0,1.0,1.0],vec![1.0,0.0,0.0,1.0]]];

    let outputs = vec![vec![vec![0.0],vec![1.0],vec![1.0]]];

    println!("{:?}",&inputs);
    println!("{:?}",&inputs[1][1][0]);

    let weights = generate_weights(&testinputs);
    println!("Weights");
    let bias = generate_bias(&testinputs);
    println!("Bias");
    let outweights = generate_weights(&outputs);

    forward_propagate(testinputs,weights,bias,outputs,outweights);
}