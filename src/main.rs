mod agent;
mod arm;
mod policy;

use std::fmt::Debug;

use crate::agent::{mse, Agent, Player};
use crate::arm::{BanditArm, BernoulliArm};
use crate::policy::{AnnealingSoftmax, EpsilonGreedy, Softmax, UCB};

fn report<B: BanditArm + Debug>(name: &str, bandit: &[B], freqs: &[f64], idx: usize) {
    println!("{}", name);
    println!(
        "Bandit: {:?}\nFreqs: {:?}\nError: {:?}",
        bandit,
        freqs,
        mse(&freqs, idx)
    );
}

fn run<A, B>(agent: &mut A, bandit: &[B], n_episodes: i32, horizon: i32) -> Vec<f64>
where
    A: Agent,
    B: BanditArm,
{
    let n_arms = bandit.len();
    agent.init(n_arms);
    let mut freqs = vec![0.0; n_arms];
    for _ep in 0..n_episodes {
        for _t in 0..horizon {
            let arm = agent.select_arm();
            let reward = bandit[arm].draw();
            agent.update(arm, reward);
        }
        let best = agent.best_arm();
        freqs[best] += 1.0;
    }
    for elem in freqs.iter_mut() {
        *elem /= n_episodes as f64;
    }
    freqs
}

fn main() {

    println!("Hello Bandit!");

    let bandit = vec![
        BernoulliArm::new(0.1),
        BernoulliArm::new(0.1),
        BernoulliArm::new(0.15),
        BernoulliArm::new(0.1),
    ];

    let eg = EpsilonGreedy::new(0.1);
    let mut eg_player = Player::new(eg);
    let fq = run(&mut eg_player, &bandit, 10000, 5);
    report("EpsilonGreedy", &bandit, &fq, 2);

    let sm = Softmax::new(1.0);
    let mut sm_player = Player::new(sm);
    let fq = run(&mut sm_player, &bandit, 10000, 5);
    report("Softmax", &bandit, &fq, 2);

    let am = AnnealingSoftmax::new(1.0);
    let mut am_player = Player::new(am);
    let fq = run(&mut am_player, &bandit, 10000, 5);
    report("AnnealingSoftmax", &bandit, &fq, 2);

    let uc = UCB::new();
    let mut uc_player = Player::new(uc);
    let fq = run(&mut uc_player, &bandit, 10000, 5);
    report("UCB", &bandit, &fq, 2);
}
