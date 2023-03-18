use crate::BanditArm;

pub trait Agent {
    fn run<T: BanditArm>(&mut self, bandit: &[T], n_episodes: i32, horizon: i32) -> Vec<f64>;
    fn select_arm(&mut self) -> usize;
    fn update(&mut self, arm: usize, reward: f64);
    fn reset(&mut self);
}

pub fn argmax(vals: &[f64]) -> usize {
    let mut max = f64::MIN;
    let mut idx: usize = 0;
    for (i, &val) in vals.iter().enumerate() {
        if val > max {
            max = val;
            idx = i;
        }
    }
    idx
}

pub fn mse(vals: &[f64], idx: usize) -> f64 {
    let error: f64 = vals
        .iter()
        .enumerate()
        .map(|(i, val)| {
            if i == idx {
                let x = (val - 1.0).exp2();
                println!("x: {} val: {}", x, val);
                x
            } else {
                val.exp2()
            }
        }).sum();
    (error / vals.len() as f64).sqrt()
}