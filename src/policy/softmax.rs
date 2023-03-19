use crate::agent::Agent;
use crate::policy::Policy;

#[derive(Debug)]
pub struct Softmax {
    temperature: f64,
}

impl Softmax {
    pub fn new(temperature: f64) -> Softmax {
        Softmax { temperature }
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

impl Policy for Softmax {
    fn select_arm<A: Agent>(&self, agent: &A) -> usize {
        let values = agent.get_values();
        let mut distro: Vec<f64> = values
            .iter()
            .map(|val| val.exp() / self.temperature)
            .collect();
        let norm: f64 = distro.iter().sum();
        for val in distro.iter_mut() {
            *val /= norm;
        }
        Softmax::choose(&distro)
    }
}

#[derive(Debug)]
pub struct AnnealingSoftmax {
    temperature: f64,
}

impl AnnealingSoftmax {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }
}

impl Policy for AnnealingSoftmax {
    fn select_arm<A: Agent>(&self, agent: &A) -> usize {
        let values = agent.get_values();
        let count: f64 = agent.get_counts().iter().sum();
        let temp = self.temperature / (count + 1.00001).ln();
        let mut distro: Vec<f64> = values.iter().map(|val| val.exp() / temp).collect();
        let norm: f64 = distro.iter().sum();
        for val in distro.iter_mut() {
            *val /= norm;
        }
        Softmax::choose(&distro)
    }
}
