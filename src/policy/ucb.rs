use crate::agent::Agent;
use crate::policy::{argmax, Policy};

#[derive(Debug)]
pub struct UCB {}

impl UCB {
    pub fn new() -> Self {
        Self {}
    }
}

impl Policy for UCB {
    fn select_arm<A: Agent>(&self, agent: &A) -> usize {
        let counts = agent.borrow_counts();
        for (idx, &count) in counts.iter().enumerate() {
            if count < 1.0 {
                return idx;
            }
        }
        let norm: f64 = counts.iter().sum::<f64>().ln();
        let values: Vec<f64> = agent
            .borrow_values()
            .iter()
            .zip(counts.iter())
            .map(|(val, cnt)| val + (norm / cnt).sqrt())
            .collect();
        argmax(&values)
    }
}
