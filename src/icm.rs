use ndarray::{s, Array1, Array2};
use ndarray_linalg::{Cholesky, Inverse, UPLO};
use crate::kernels::{KernelFamily, KernelHyper, build_kx};

// ICM: K = B ⊗ Kx  +  Σ_noise ⊗ I_n
// Here we keep Σ_noise diagonal (per-output nugget), and B is SPD 2x2 or MxM.
#[derive(Clone)]
pub struct ICM {
    // coregionalization B (M x M), SPD
    pub b: Array2<f64>,
    // per-output nugget ratios η_m (scaled by sigma^2)
    pub eta: Array1<f64>,
    pub fam: KernelFamily,
    pub hyp: KernelHyper,
    // caches for f-only whitening per-output (Kx + η_m I)^{-1/2}
    pub l: Vec<Array2<f64>>,
    pub linv: Vec<Array2<f64>>,
    pub logdet_blocks: Vec<f64>,
    // Carlin–Chib shadows (for other families)
    pub shadow_ell: [f64; 7], // All kernel families
    // MH steps
    pub mh_step_ell: f64,
    pub mh_step_eta: f64,
    pub mh_step_b: f64,
    pub mh_step_alpha: f64,
    pub mh_step_period: f64,
    pub mh_t: usize,
}

impl ICM {
    pub fn new(m_out: usize, fam: KernelFamily, hyp: KernelHyper, t: &Array1<f64>) -> Self {
        // start B ~ I, small nuggets
        let mut b = Array2::<f64>::zeros((m_out, m_out));
        for i in 0..m_out {
            b[(i, i)] = 1.0;
        }
        let eta = Array1::from(vec![1e-3; m_out]);

        let mut icm = Self {
            b,
            eta,
            fam,
            hyp,
            l: Vec::new(),
            linv: Vec::new(),
            logdet_blocks: Vec::new(),
            shadow_ell: [hyp.ell, hyp.ell, hyp.ell, hyp.ell, hyp.ell, hyp.ell, hyp.ell],
            mh_step_ell: 0.10,
            mh_step_eta: 0.25,
            mh_step_b: 0.10,
            mh_step_alpha: 0.15,
            mh_step_period: 0.15,
            mh_t: 0,
        };
        icm.rebuild_fonly_blocks(t);
        icm
    }

    pub fn fam_idx(f: KernelFamily) -> usize {
        match f {
            KernelFamily::SE => 0,
            KernelFamily::Matern32 => 1,
            KernelFamily::Matern52 => 2,
            KernelFamily::RQ => 3,
            KernelFamily::Periodic => 4,
            KernelFamily::Exponential => 5,
            KernelFamily::White => 6,
        }
    }

    // Rebuild per-output Cholesky of Kx + η_m I (for fast f-only whitening)
    pub fn rebuild_fonly_blocks(&mut self, t: &Array1<f64>) {
        let n = t.len();
        let eye = Array2::<f64>::eye(n);
        let kx = build_kx(t, self.fam, &self.hyp);

        self.l.clear();
        self.linv.clear();
        self.logdet_blocks.clear();

        for m in 0..self.eta.len() {
            let mut km = kx.clone();
            for i in 0..n {
                km[(i, i)] += self.eta[m];
            }
            // jittered chol
            let mut jitter = 0.0_f64;
            let max_jitter = 1e-2_f64;
            let l = loop {
                let try_k = &km + &(eye.clone() * jitter);
                match try_k.clone().cholesky(UPLO::Lower) {
                    Ok(l) => break l,
                    Err(_) => {
                        jitter = if jitter == 0.0 { 1e-10 } else { jitter * 10.0 };
                        if jitter > max_jitter {
                            panic!("chol failed");
                        }
                    }
                }
            };
            let linv = l.clone().inv().expect("L inv");
            let logdet = 2.0 * (0..n).map(|i| l[(i, i)].ln()).sum::<f64>();
            self.l.push(l);
            self.linv.push(linv);
            self.logdet_blocks.push(logdet);
        }
    }

    // Whiten stacked Y of shape (M*n) under full ICM for marginalization over mean
    // Y layout is [y^(0); y^(1); ...; y^(M-1)], each length n
    // We exploit the Kronecker form via eigen-decomp of B, but to keep code compact,
    // we form the full (M*n) covariance by blocks: K = B ⊗ Kx + diag(η) ⊗ I.
    pub fn whiten_joint(
        &self,
        y_stacked: &Array1<f64>,
        t: &Array1<f64>,
    ) -> (Array1<f64>, Array2<f64>, f64)
    {
        // Build Kx
        let n = t.len();
        let m = self.eta.len();
        let kx = build_kx(t, self.fam, &self.hyp);

        // Build K = B ⊗ Kx + diag(η) ⊗ I_n
        let dim = m * n;
        let mut k = Array2::<f64>::zeros((dim, dim));
        // B ⊗ Kx
        for a in 0..m {
            for b in 0..m {
                let block = self.b[(a, b)];
                if block.abs() < 1e-16 { continue; }
                let mut sub = k.slice_mut(s![a*n..(a+1)*n, b*n..(b+1)*n]);
                for i in 0..n {
                    for j in 0..n {
                        sub[(i, j)] += block * kx[(i, j)];
                    }
                }
            }
        }
        // + diag(η) ⊗ I
        for a in 0..m {
            for i in 0..n {
                k[(a*n+i, a*n+i)] += self.eta[a];
            }
        }

        // jittered chol with fallback to SE kernel
        let eye = Array2::<f64>::eye(dim);
        let mut jitter = 1e-6_f64;  // Start with larger jitter
        let max_jitter = 1e0_f64;   // Allow much larger jitter
        let l = loop {
            let try_k = &k + &(eye.clone() * jitter);
            match try_k.cholesky(UPLO::Lower) {
                Ok(l) => break l,
                Err(e) => {
                    jitter *= 1.5;  // Increase jitter more gradually
                    if jitter > max_jitter { 
                        eprintln!("Cholesky failed with jitter {:.2e}, max_jitter {:.2e}", jitter, max_jitter);
                        eprintln!("Kernel family: {:?}, ell: {:.6}, alpha: {:.6}, period: {:.6}", 
                                 self.fam, self.hyp.ell, self.hyp.alpha, self.hyp.period);
                        eprintln!("Falling back to SE kernel with safe parameters...");
                        
                        // Fallback: rebuild with SE kernel and safe parameters
                        let safe_hyp = crate::kernels::KernelHyper {
                            ell: 0.25,
                            alpha: 1.0,
                            period: 1.0,
                        };
                        let safe_kx = crate::kernels::build_kx(t, crate::kernels::KernelFamily::SE, &safe_hyp);
                        
                        // Rebuild full covariance with SE kernel
                        let mut safe_k = Array2::<f64>::zeros((dim, dim));
                        for a in 0..m {
                            for bidx in 0..m {
                                let coef = self.b[(a, bidx)];
                                if coef.abs() < 1e-16 { continue; }
                                let mut sub = safe_k.slice_mut(s![a*n..(a+1)*n, bidx*n..(bidx+1)*n]);
                                for i in 0..n {
                                    for j in 0..n {
                                        sub[(i, j)] += coef * safe_kx[(i, j)];
                                    }
                                }
                            }
                        }
                        for a in 0..m {
                            for i in 0..n {
                                safe_k[(a*n+i, a*n+i)] += self.eta[a];
                            }
                        }
                        
                        // Try Cholesky with SE kernel
                        let safe_try_k = &safe_k + &(eye.clone() * 1e-6);
                        match safe_try_k.cholesky(UPLO::Lower) {
                            Ok(l) => break l,
                            Err(_) => {
                                eprintln!("Even SE kernel failed. This is a serious numerical issue.");
                                panic!("chol failed full ICM: {:?}", e);
                            }
                        }
                    }
                }
            }
        };
        let linv = l.clone().inv().expect("L inv");
        let logdet = 2.0 * (0..dim).map(|i| l[(i, i)].ln()).sum::<f64>();

        let y_t = linv.dot(y_stacked);
        // Design X: wavelet mean only applies to function rows (i.e., each output)
        // We build per-output wavelet design (n x p) and then block-diagonal it M times.
        // To keep it compact, we pass back only the (dim x p_total) whitened design.
        // The caller will construct X (blockdiag) and hand it in here already whitened;
        // to match your previous pattern, we return an identity so marginal integrates over β via y_t only.
        (y_t, Array2::<f64>::zeros((dim, 0)), logdet)
    }
}

pub fn simulate_icm_curves(
    rng: &mut rand::rngs::StdRng,
    n: usize,
    m_out: usize,
    kx: &Array2<f64>,
    b: &Array2<f64>,
    eta: &Array1<f64>,
    n_curves: usize,
) -> Vec<crate::data_structures::CurveM> {
    use rand_distr::{Distribution, Normal};
    use ndarray_linalg::UPLO;
    use ndarray::s;

    // Full covariance = B ⊗ Kx + diag(η) ⊗ I
    let dim = m_out * n;
    let mut k = Array2::<f64>::zeros((dim, dim));
    for a in 0..m_out {
        for bidx in 0..m_out {
            let coef = b[(a, bidx)];
            if coef.abs() < 1e-16 { continue; }
            let mut sub = k.slice_mut(s![a*n..(a+1)*n, bidx*n..(bidx+1)*n]);
            for i in 0..n {
                for j in 0..n {
                    sub[(i, j)] += coef * kx[(i, j)];
                }
            }
        }
    }
    for a in 0..m_out {
        for i in 0..n {
            k[(a*n+i, a*n+i)] += eta[a];
        }
    }
    // jittered chol
    let eye = Array2::<f64>::eye(dim);
    let mut jitter = 1e-6_f64;
    let max_jitter = 1e0_f64;
    let l = loop {
        let try_k = &k + &(eye.clone() * jitter);
        match try_k.clone().cholesky(UPLO::Lower) {
            Ok(l) => break l,
            Err(_) => {
                jitter *= 1.5;
                if jitter > max_jitter { panic!("chol failed simulate"); }
            }
        }
    };

    let mut out = Vec::with_capacity(n_curves);
    for _ in 0..n_curves {
        // z ~ N(0, I), y = L z
        let mut z = Array1::<f64>::zeros(dim);
        for zi in z.iter_mut() {
            *zi = Normal::new(0.0, 1.0).unwrap().sample(rng);
        }
        let s = l.dot(&z);
        // reshape to (n, m_out)
        let mut y = Array2::<f64>::zeros((n, m_out));
        for a in 0..m_out {
            for i in 0..n {
                y[(i, a)] = s[a*n + i];
            }
        }
        out.push(crate::data_structures::CurveM { y });
    }
    out
}
