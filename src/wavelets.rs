use ndarray::Array2;
use crate::utils::is_power_of_two;

#[derive(Clone, Copy)]
pub enum DesignKind { 
    Haar, 
    Daub4, 
}

pub fn build_haar_matrix(m: usize) -> Array2<f64> {
    assert!(is_power_of_two(m));
    // Orthonormal Haar synthesis (like before, compact):
    // Column 0 = DC; then detail columns by scale.
    let mut w = Array2::<f64>::zeros((m, m));
    let a0 = (1.0 / m as f64).sqrt();
    for i in 0..m { w[(i,0)] = a0; }
    let mut col = 1usize;
    let mut blk = 1usize;
    while blk < m {
        let step = m / (2*blk);
        for g in 0..blk {
            let start = g * 2 * step;
            let mid = start + step;
            let end = start + 2*step;
            let amp = (blk as f64 / m as f64).sqrt();
            for i in start..mid { w[(i, col)] =  amp; }
            for i in mid..end  { w[(i, col)] = -amp; }
            col += 1;
        }
        blk *= 2;
    }
    w
}

// Build design X (n x p) for ONE output; we'll block-diag it for M outputs.
pub fn make_wavelet_design(n: usize, p: usize, kind: DesignKind) -> Array2<f64> {
    let w = match kind {
        DesignKind::Haar => build_haar_matrix(n),
        DesignKind::Daub4 => build_haar_matrix(n), // keep simple here; swap if you add Daub4 synth.
    };
    assert!(p <= n);
    let mut x = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        for i in 0..n {
            x[(i, j)] = w[(i, j)];
        }
    }
    x
}

// Simple Besov-like shrinkage prior: V0 = diag(1/g_j) with levelwise inflation.
pub fn build_shrink_v0(p: usize, _n: usize) -> Array2<f64> {
    // assign levels ~ dyadic, heavier shrink at high freq
    let mut v = Array2::<f64>::eye(p);
    let mut level = 0usize;
    let mut idx = 1usize;
    while idx < p {
        let band = (1usize << level).min(p - idx);
        let weight = 1.0 + (level as f64).max(1.0); // grows with level
        for j in 0..band {
            v[(idx + j, idx + j)] = weight;
        }
        idx += band;
        level += 1;
    }
    // scale for stability
    v * 10.0
}

// Conjugate Normal–InvGamma prior in whitened space (as in your code).
#[derive(Clone)]
pub struct NIGPrior {
    pub m0: ndarray::Array1<f64>,
    pub v0: Array2<f64>,
    pub a0: f64,
    pub b0: f64,
}

pub fn log_marginal_whitened(
    y_t: &ndarray::Array1<f64>,
    x_t: &Array2<f64>,
    prior: &NIGPrior,
    logdet_k: f64
) -> f64 {
    use statrs::function::gamma::ln_gamma;
    use std::f64::consts::PI;
    use ndarray_linalg::{Cholesky, Inverse, UPLO};

    let n = y_t.len();
    let p = x_t.shape()[1];

    // No design columns: p = 0. Integrate σ^2 with IG(a0, b0) prior.
    // Posterior a_n = a0 + n/2, b_n = b0 + 0.5 * y^T y.
    // Marginal log-likelihood (up to constants) has no V0/Vn terms here.
    if p == 0 {
        let yty = y_t.dot(y_t);
        let a_n = prior.a0 + 0.5 * (n as f64);
        let b_n = prior.b0 + 0.5 * yty;

        // −(n/2)ln(2π) + a0 ln b0 − a_n ln b_n + ln Γ(a_n) − ln Γ(a0) − 0.5 log|K|
        return -0.5 * (n as f64) * (2.0 * PI).ln()
            + prior.a0 * prior.b0.ln()
            - a_n * b_n.ln()
            + ln_gamma(a_n)
            - ln_gamma(prior.a0)
            - 0.5 * logdet_k;
    }

    // Otherwise (p > 0): original conjugate computation with shrinkage prior
    let v0 = &prior.v0;
    let v0_inv = v0.clone().inv().expect("V0 SPD");
    let xtx = x_t.t().dot(x_t);
    let vn_inv = &v0_inv + &xtx;
    let vn = vn_inv.clone().inv().expect("Vn SPD");

    let l0 = v0.clone().cholesky(UPLO::Lower).expect("V0 chol");
    let ln_det_v0 = 2.0 * (0..p).map(|i| l0[(i, i)].ln()).sum::<f64>();

    let ln = vn.clone().cholesky(UPLO::Lower).expect("Vn chol");
    let ln_det_vn = 2.0 * (0..p).map(|i| ln[(i, i)].ln()).sum::<f64>();

    let xty = x_t.t().dot(y_t);
    let mn = vn.dot(&(v0_inv.dot(&prior.m0) + xty));

    let yty = y_t.dot(y_t);
    let m0_v0inv_m0 = prior.m0.dot(&v0_inv.dot(&prior.m0));
    let mn_vninv_mn = mn.dot(&vn_inv.dot(&mn));

    let a_n = prior.a0 + 0.5 * (n as f64);
    let b_n = prior.b0 + 0.5 * (yty + m0_v0inv_m0 - mn_vninv_mn);

    -0.5 * (n as f64) * (2.0 * PI).ln()
        + 0.5 * (ln_det_v0 - ln_det_vn)
        + prior.a0 * prior.b0.ln()
        - a_n * b_n.ln()
        + ln_gamma(a_n)
        - ln_gamma(prior.a0)
        - 0.5 * logdet_k
}
