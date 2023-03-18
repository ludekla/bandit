use rand::{self, rngs::ThreadRng, Rng};

use crate::policy::{argmax, Policy};

#[derive(Debug)]
pub struct EpsilonGreedy {
    epsilon: f64,
    rng: ThreadRng,
}

impl EpsilonGreedy {
    pub fn new(epsilon: f64) -> EpsilonGreedy {
        EpsilonGreedy {
            epsilon,
            rng: rand::thread_rng(),
        }
    }
}

impl Policy for EpsilonGreedy {
    fn select_arm(&mut self, values: &[f64]) -> usize {
        if rand::random::<f64>() < self.epsilon {
            self.rng.gen_range(0..values.len())
        } else {
            argmax(values)
        }
    }
}
