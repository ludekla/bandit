mod epsgreed;
mod softmax;

pub use epsgreed::EpsilonGreedy;
pub use softmax::{AnnealingSoftmax, Softmax};

use crate::agent::Agent;

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

pub trait Policy {
    fn select_arm<A: Agent>(&self, agent: &A) -> usize;
}
