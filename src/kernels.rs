use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelFamily {
    SE,
    Matern32,
    Matern52,
    RQ,           // Rational Quadratic
    Periodic,     // Periodic
    Exponential,  // Ornstein-Uhlenbeck
    White,        // White Noise
}

pub const AVAIL_FAMS: &[KernelFamily] = &[
    KernelFamily::SE,
    KernelFamily::Matern32,
    KernelFamily::Matern52,
    KernelFamily::RQ,
    KernelFamily::Periodic,
    KernelFamily::Exponential,
    KernelFamily::White,
];

#[derive(Clone, Copy, Debug)]
pub struct KernelHyper {
    pub ell: f64,
    pub alpha: f64,    // For RQ kernel
    pub period: f64,   // For Periodic kernel
}

impl Default for KernelHyper {
    fn default() -> Self {
        Self { 
            ell: 0.25,
            alpha: 1.0,     // Default for RQ kernel
            period: 1.0,    // Default period
        }
    }
}

pub fn se_corr(r: f64, ell: f64) -> f64 {
    let ell_safe = ell.max(1e-6);
    (-(r * r) / (2.0 * ell_safe * ell_safe)).exp().clamp(0.0, 1.0)
}

pub fn matern32_corr(r: f64, ell: f64) -> f64 {
    let ell_safe = ell.max(1e-6);
    let s = (3.0f64).sqrt() * r / ell_safe;
    (1.0 + s) * (-s).exp().clamp(0.0, 1.0)
}

pub fn matern52_corr(r: f64, ell: f64) -> f64 {
    let ell_safe = ell.max(1e-6);
    let s = (5.0f64).sqrt() * r / ell_safe;
    (1.0 + s + (s * s) / 3.0) * (-s).exp().clamp(0.0, 1.0)
}

// Rational Quadratic kernel
pub fn rq_corr(r: f64, ell: f64, alpha: f64) -> f64 {
    let ell_safe = ell.max(1e-6);
    let alpha_safe = alpha.max(1e-6);
    let ratio = (r * r) / (2.0 * alpha_safe * ell_safe * ell_safe);
    (1.0 + ratio).powf(-alpha_safe).clamp(0.0, 1.0)
}

// Periodic kernel
pub fn periodic_corr(r: f64, ell: f64, period: f64) -> f64 {
    let ell_safe = ell.max(1e-6);
    let period_safe = period.max(1e-6);
    let sin_term = (std::f64::consts::PI * r / period_safe).sin();
    (-2.0 * sin_term * sin_term / (ell_safe * ell_safe)).exp().clamp(0.0, 1.0)
}

// Exponential (Ornstein-Uhlenbeck) kernel
pub fn exp_corr(r: f64, ell: f64) -> f64 {
    let ell_safe = ell.max(1e-6);
    (-r / ell_safe).exp().clamp(0.0, 1.0)
}


// White noise kernel
pub fn white_corr(r: f64, _ell: f64) -> f64 {
    if r == 0.0 { 1.0 } else { 0.0 }
}

pub fn base_corr(fam: KernelFamily, r: f64, h: &KernelHyper) -> f64 {
    match fam {
        KernelFamily::SE => se_corr(r, h.ell),
        KernelFamily::Matern32 => matern32_corr(r, h.ell),
        KernelFamily::Matern52 => matern52_corr(r, h.ell),
        KernelFamily::RQ => rq_corr(r, h.ell, h.alpha),
        KernelFamily::Periodic => periodic_corr(r, h.ell, h.period),
        KernelFamily::Exponential => exp_corr(r, h.ell),
        KernelFamily::White => white_corr(r, h.ell),
    }
}

// Validate kernel parameters for numerical stability
fn validate_kernel_params(fam: KernelFamily, h: &KernelHyper) -> KernelHyper {
    let mut safe_h = h.clone();
    
    // Ensure ell is always positive and reasonable
    safe_h.ell = safe_h.ell.max(1e-4).min(5.0);
    
    match fam {
        KernelFamily::RQ => {
            // RQ kernel: alpha should be positive, not too small
            safe_h.alpha = safe_h.alpha.max(0.2).min(5.0);
        },
        KernelFamily::Periodic => {
            // Periodic kernel: period should be positive, reasonable range
            safe_h.period = safe_h.period.max(0.2).min(5.0);
        },
        KernelFamily::Exponential => {
            // Exponential kernel: ensure ell is not too small
            safe_h.ell = safe_h.ell.max(1e-3).min(2.0);
        },
        _ => {
            // For other kernels, just ensure reasonable ell
        }
    }
    
    safe_h
}

pub fn build_kx(t: &Array1<f64>, fam: KernelFamily, h: &KernelHyper) -> Array2<f64> {
    let safe_h = validate_kernel_params(fam, h);
    let n = t.len();
    let mut k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        k[(i, i)] = 1.0;
        for j in 0..i {
            let r = (t[i] - t[j]).abs();
            let v = base_corr(fam, r, &safe_h);
            k[(i, j)] = v;
            k[(j, i)] = v;
        }
    }
    k
}
