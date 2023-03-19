use crate::policy::{argmax, Policy};

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

pub trait Agent {
    fn init(&mut self, n_arms: usize);
    fn select_arm(&self) -> usize;
    fn update(&mut self, arm: usize, reward: f64);
    fn get_values(&self) -> &[f64];
    fn get_counts(&self) -> &[f64];
    fn best_arm(&self) -> usize;
    fn reset(&mut self);
}

pub struct Player<P: Policy> {
    policy: P,
    n_arms: usize,
    values: Vec<f64>,
    counts: Vec<f64>,
}

impl<P: Policy> Player<P> {
    pub fn new(policy: P) -> Self {
        Self {
            policy,
            n_arms: 0,
            values: Vec::new(),
            counts: Vec::new(),
        }
    }
}

impl<P: Policy> Agent for Player<P> {
    fn init(&mut self, n_arms: usize) {
        self.n_arms = n_arms;
        self.values = vec![0.0; n_arms];
        self.counts = vec![0.0; n_arms];
    }

    fn select_arm(&self) -> usize {
        self.policy.select_arm(self)
    }

    fn update(&mut self, arm: usize, reward: f64) {
        self.counts[arm] += 1.0;
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm];
    }

    fn get_values(&self) -> &[f64] {
        &self.values
    }

    fn get_counts(&self) -> &[f64] {
        &self.counts
    }

    fn best_arm(&self) -> usize {
        argmax(&self.values)
    }

    fn reset(&mut self) {
        self.n_arms = 0;
        self.values.clear();
        self.counts.clear();
    }
}
