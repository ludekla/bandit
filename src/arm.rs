use rand::random;

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
pub struct BernoulliArm {
    prob: f64,
}

impl BernoulliArm {
    pub fn new(prob: f64) -> BernoulliArm {
        BernoulliArm { prob }
    }
}

impl BanditArm for BernoulliArm {
    fn draw(&self) -> f64 {
        if random::<f64>() < self.prob {
            1.0
        } else {
            0.0
        }
    }
}
