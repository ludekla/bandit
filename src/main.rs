use rand::{self, rngs::ThreadRng, Rng};

fn argmax(vals: &[f64]) -> usize {
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

/// Trait description
trait BanditArm {
    /// Method descr
    /// # Example
    /// ```
    /// let x = b.draw();
    /// ```
    fn draw(&self) -> f64;
}

#[derive(Debug)]
struct BernoulliArm {
    prob: f64,
}

impl BernoulliArm {
    fn new(prob: f64) -> BernoulliArm {
        BernoulliArm { prob }
    }
}

impl BanditArm for BernoulliArm {
    fn draw(&self) -> f64 {
        if rand::random::<f64>() < self.prob {
            1.0
        } else {
            0.0
        }
    }
}

trait BanditAlgo {
    fn run<T: BanditArm>(&mut self, bandit: &Vec<T>, n_episodes: i32, horizon: i32) -> Vec<f64>;
    fn select_arm(&mut self) -> usize;
    fn update(&mut self, arm: usize, reward: f64);
    fn reset(&mut self);
}

#[derive(Debug)]
struct EpsilonGreedy {
    epsilon: f64,
    n_arms: usize,
    values: Vec<f64>,
    counts: Vec<f64>,
    rng: ThreadRng,
}

impl EpsilonGreedy {
    fn new(epsilon: f64) -> EpsilonGreedy {
        EpsilonGreedy {
            epsilon,
            n_arms: 0,
            values: Vec::new(),
            counts: Vec::new(),
            rng: rand::thread_rng(),
        }
    }
}

impl BanditAlgo for EpsilonGreedy {
    fn run<T: BanditArm>(&mut self, bandit: &Vec<T>, n_episodes: i32, horizon: i32) -> Vec<f64> {
        self.n_arms = bandit.len();
        self.values = vec![0.0; self.n_arms];
        self.counts = vec![0.0; self.n_arms];
        let mut freqs = vec![0.0; self.n_arms];
        for ep in 0..n_episodes {
            for t in 0..horizon {
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
        if rand::random::<f64>() < self.epsilon {
            self.rng.gen_range(0..self.n_arms)
        } else {
            argmax(&self.values)
        }
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

fn main() {
    let bandit = vec![
        BernoulliArm::new(0.1),
        BernoulliArm::new(0.3),
        BernoulliArm::new(0.9),
        BernoulliArm::new(0.2),
    ];

    let mut eg = EpsilonGreedy::new(0.1);

    let fq = eg.run(&bandit, 10000, 5);

    println!("Bandit: {:?}, Freqs: {:?}", bandit, fq);
}
