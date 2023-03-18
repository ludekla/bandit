use crate::agent::{argmax, Agent};
use crate::BanditArm;

#[derive(Debug)]
pub struct Softmax {
    temperature: f64,
    n_arms: usize,
    values: Vec<f64>,
    counts: Vec<f64>,
}

impl Softmax {
    pub fn new(temperature: f64) -> Softmax {
        Softmax {
            temperature,
            n_arms: 0,
            values: Vec::new(),
            counts: Vec::new(),
        }
    }
    // helper function
    fn choose(distro: &[f64]) -> usize {
        let mut cumsum = 0.0;
        let thresh = rand::random::<f64>();
        for (i, val) in distro.iter().enumerate() {
            cumsum += val;
            if cumsum > thresh {
                return i as usize;
            }
        }
        distro.len() - 1
    }
}

impl Agent for Softmax {
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
        let mut distro: Vec<f64> = self
            .values
            .iter()
            .map(|val| val.exp() / self.temperature)
            .collect();
        let norm: f64 = distro.iter().sum();
        for val in distro.iter_mut() {
            *val /= norm;
        }
        Softmax::choose(&distro)
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

#[derive(Debug)]
pub struct AnnealingSoftmax {
    softmax: Softmax,
}

impl AnnealingSoftmax {
    pub fn new(temperature: f64) -> Self {
        AnnealingSoftmax {
            softmax: Softmax::new(temperature),
        }
    }
}

impl Agent for AnnealingSoftmax {
    fn run<T: BanditArm>(&mut self, bandit: &[T], n_episodes: i32, horizon: i32) -> Vec<f64> {
        self.softmax.run(bandit, n_episodes, horizon)
    }
    fn select_arm(&mut self) -> usize {
        let count: f64 = self.softmax.counts.iter().sum();
        let temp = self.softmax.temperature / (count + 1.00001).ln();
        let mut distro: Vec<f64> = self
            .softmax
            .values
            .iter()
            .map(|val| val.exp() / temp)
            .collect();
        let norm: f64 = distro.iter().sum();
        for val in distro.iter_mut() {
            *val /= norm;
        }
        Softmax::choose(&distro)
    }
    fn update(&mut self, arm: usize, reward: f64) {
        self.softmax.update(arm, reward);
    }
    fn reset(&mut self) {
        self.softmax.reset();
    }
}
