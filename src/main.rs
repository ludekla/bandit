mod agent;
mod epsgreed;
mod softmax;

use crate::agent::{Agent, mse};
use crate::epsgreed::EpsilonGreedy;
use crate::softmax::{Softmax, AnnealingSoftmax};

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
        BernoulliArm::new(0.1),
        BernoulliArm::new(0.15),
        BernoulliArm::new(0.1),
    ];

    let mut eg = EpsilonGreedy::new(0.1);
    let fq = eg.run(&bandit, 10000, 5);
    println!("EpsilonGreedy");
    println!("Bandit: {:?}\nFreqs: {:?}\nError: {:?}", bandit, fq, mse(&fq, 2));

    let mut sm = Softmax::new(1.0);
    let fq = sm.run(&bandit, 200, 5);
    println!("Softmax");
    println!("Bandit: {:?}\nFreqs: {:?}\nError: {:?}", bandit, fq, mse(&fq, 2));

    let mut sm = AnnealingSoftmax::new(1.0);
    let fq = sm.run(&bandit, 200, 5);
    println!("AnnealingSoftmax");
    println!("Bandit: {:?}\nFreqs: {:?}\nError: {:?}", bandit, fq, mse(&fq, 2));
}
