use crate::BanditArm;

pub trait Agent {
    fn run<T: BanditArm>(&mut self, bandit: &[T], n_episodes: i32, horizon: i32) -> Vec<f64>;
    fn select_arm(&mut self) -> usize;
    fn update(&mut self, arm: usize, reward: f64);
    fn reset(&mut self);
}

pub struct ProtoAgent {
    n_arms: usize,
    values: Vec<f64>,
    counts: Vec<f64>,
}

impl ProtoAgent {
    pub fn new() -> Self {
        Self {
            n_arms: 0,
            values: Vec::new(),
            counts: Vec::new(),
        }
    }
}

impl Agent for ProtoAgent {
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

    fn select_arm(&mut self) -> usize { 0 }

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
        .map(|(i, val)| match i {
            a if a == idx => (val - 1.0).powi(2),
            _a => val.powi(2),
        })
        .sum();
    (error / vals.len() as f64).sqrt()
}
