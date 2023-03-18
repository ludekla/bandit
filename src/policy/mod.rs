mod epsgreed;
mod softmax;

pub use epsgreed::EpsilonGreedy;
pub use softmax::{AnnealingSoftmax, Softmax};

pub trait Policy {
    fn select_arm(&mut self, values: &[f64]) -> usize;
}

pub fn mse(vals: &[f64], idx: usize) -> f64 {
    let error: f64 = vals
        .iter()
        .enumerate()
        .map(|(i, val)| match i {
            a if a == idx => (val - 1.0).powi(2),
            _a => val.powi(2),
        })
        .sum();
    (error / vals.len() as f64).sqrt()
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
