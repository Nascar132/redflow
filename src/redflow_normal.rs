//Forward propagation
pub fn sigmoid(x:f64) -> f64{
    let one = 1.0_f64;
    let e = one.exp();
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

//Create the weights and bias
//pub fn weights_and_bias_generation(&input){}