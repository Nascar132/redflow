pub fn sigmoid(x:f64) -> f64{
    let one = 1.0_f64;
    let e = one.exp();
    let mut ans = -x;
    ans = ans.exp();
    ans = 1.0 + ans;
    ans = 1.0 / ans;
    return ans
}