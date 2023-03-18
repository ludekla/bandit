use crate::arm::BanditArm;
use crate::policy::{argmax, Policy};

pub trait Agent {
    fn run<T: BanditArm>(&mut self, bandit: &[T], n_episodes: i32, horizon: i32) -> Vec<f64>;
    fn select_arm(&mut self) -> usize;
    fn update(&mut self, arm: usize, reward: f64);
    fn reset(&mut self);
}

pub struct Player<P: Policy> {
    policy: P,
    n_arms: usize,
    values: Vec<f64>,
    counts: Vec<f64>,
}

impl<P> Player<P>
where
    P: Policy,
{
    pub fn new(policy: P) -> Self {
        Self {
            policy,
            n_arms: 0,
            values: Vec::new(),
            counts: Vec::new(),
        }
    }
}

impl<P> Agent for Player<P>
where
    P: Policy,
{
    fn run<T: BanditArm>(&mut self, bandit: &[T], n_episodes: i32, horizon: i32) -> Vec<f64> {
        self.n_arms = bandit.len();
        self.values = vec![0.0; self.n_arms];
        self.counts = vec![0.0; self.n_arms];
        let mut freqs = vec![0.0; self.n_arms];
        for _ep in 0..n_episodes {
            for _t in 0..horizon {
                let arm = self.select_arm();
                let reward = bandit[arm].draw();
                self.update(arm, reward);
            }
            let best = argmax(&self.values);
            freqs[best] += 1.0;
        }
        for elem in freqs.iter_mut() {
            *elem /= n_episodes as f64;
        }
        freqs
    }

    fn select_arm(&mut self) -> usize {
        self.policy.select_arm(&self.values)
    }

    fn update(&mut self, arm: usize, reward: f64) {
        self.counts[arm] += 1.0;
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm];
    }

    fn reset(&mut self) {
        self.n_arms = 0;
        self.values.clear();
        self.counts.clear();
    }
}
