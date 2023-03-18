mod agent;
mod arm;
mod policy;

use crate::agent::{Agent, Player};
use crate::arm::BernoulliArm;
use crate::policy::{mse, AnnealingSoftmax, EpsilonGreedy, Softmax};

fn main() {
    let bandit = vec![
        BernoulliArm::new(0.1),
        BernoulliArm::new(0.1),
        BernoulliArm::new(0.15),
        BernoulliArm::new(0.1),
    ];

    let eg = EpsilonGreedy::new(0.1);
    let mut eg_player = Player::new(eg);
    let fq = eg_player.run(&bandit, 10000, 5);
    println!("EpsilonGreedy");
    println!(
        "Bandit: {:?}\nFreqs: {:?}\nError: {:?}",
        bandit,
        fq,
        mse(&fq, 2)
    );

    let sm = Softmax::new(1.0);
    let mut sm_player = Player::new(sm);
    let fq = sm_player.run(&bandit, 10000, 5);
    println!("Softmax");
    println!(
        "Bandit: {:?}\nFreqs: {:?}\nError: {:?}",
        bandit,
        fq,
        mse(&fq, 2)
    );

    let am = AnnealingSoftmax::new(1.0);
    let mut am_player = Player::new(am);
    let fq = am_player.run(&bandit, 10000, 5);
    println!("AnnealingSoftmax");
    println!(
        "Bandit: {:?}\nFreqs: {:?}\nError: {:?}",
        bandit,
        fq,
        mse(&fq, 2)
    );
}
