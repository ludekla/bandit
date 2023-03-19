use rand::{self, Rng};

use crate::agent::Agent;
use crate::policy::{argmax, Policy};

#[derive(Debug)]
pub struct EpsilonGreedy {
    epsilon: f64,
}

impl EpsilonGreedy {
    pub fn new(epsilon: f64) -> EpsilonGreedy {
        EpsilonGreedy { epsilon }
    }
}

impl Policy for EpsilonGreedy {
    fn select_arm<A: Agent>(&self, agent: &A) -> usize {
        let values = agent.get_values();
        if rand::random::<f64>() < self.epsilon {
            rand::thread_rng().gen_range(0..values.len())
        } else {
            argmax(values)
        }
    }
}
