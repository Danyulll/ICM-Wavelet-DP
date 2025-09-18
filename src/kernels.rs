use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelFamily {
    SE,
    Matern32,
    Matern52,
}

pub const AVAIL_FAMS: &[KernelFamily] = &[
    KernelFamily::SE,
    KernelFamily::Matern32,
    KernelFamily::Matern52,
];

#[derive(Clone, Copy, Debug)]
pub struct KernelHyper {
    pub ell: f64,
}

impl Default for KernelHyper {
    fn default() -> Self {
        Self { ell: 0.25 }
    }
}

pub fn se_corr(r: f64, ell: f64) -> f64 {
    (-(r * r) / (2.0 * ell * ell)).exp()
}

pub fn matern32_corr(r: f64, ell: f64) -> f64 {
    let s = (3.0f64).sqrt() * r / ell;
    (1.0 + s) * (-s).exp()
}

pub fn matern52_corr(r: f64, ell: f64) -> f64 {
    let s = (5.0f64).sqrt() * r / ell;
    (1.0 + s + (s * s) / 3.0) * (-s).exp()
}

pub fn base_corr(fam: KernelFamily, r: f64, ell: f64) -> f64 {
    match fam {
        KernelFamily::SE => se_corr(r, ell),
        KernelFamily::Matern32 => matern32_corr(r, ell),
        KernelFamily::Matern52 => matern52_corr(r, ell),
    }
}

pub fn build_kx(t: &Array1<f64>, fam: KernelFamily, h: &KernelHyper) -> Array2<f64> {
    let n = t.len();
    let mut k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        k[(i, i)] = 1.0;
        for j in 0..i {
            let r = (t[i] - t[j]).abs();
            let v = base_corr(fam, r, h.ell);
            k[(i, j)] = v;
            k[(j, i)] = v;
        }
    }
    k
}
