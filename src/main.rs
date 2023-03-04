mod agent;

use crate::agent::{Agent, EpsilonGreedy};

/// Trait description
pub trait BanditArm {
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
