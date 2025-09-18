use rand::rngs::StdRng;
use rand_distr::{Beta, Distribution, Gamma, Normal, Uniform};
use std::collections::HashSet;
use ndarray::{Array1, Array2};
use ndarray_linalg::{UPLO, Eigh};
use crate::data_structures::{DPState, Cluster, CurveM, DatasetM};
use crate::kernels::KernelFamily;
use crate::icm::ICM;
use crate::wavelets::{NIGPrior, log_marginal_whitened};
use crate::utils::categorical_from_logp;

// stick-breaking helpers
pub fn stick_from_v(v: &[f64]) -> Vec<f64> {
    let mut pis = vec![0.0; v.len()];
    let mut prod = 1.0;
    for (k, &vk) in v.iter().enumerate() {
        pis[k] = prod * vk;
        prod *= 1.0 - vk;
    }
    pis
}

pub fn resample_v_and_pi(rng: &mut StdRng, alpha: f64, z: &[usize], kmax: usize) -> (Vec<f64>, Vec<f64>) {
    let mut counts = vec![0usize; kmax];
    for &zi in z.iter() {
        if zi < kmax { counts[zi] += 1; }
    }
    let mut v = vec![0.0; kmax];
    let mut right = 0usize;
    for k in (0..kmax).rev() {
        let a = 1.0 + counts[k] as f64;
        let b = alpha + right as f64;
        v[k] = Beta::new(a, b).unwrap().sample(rng);
        right += counts[k];
    }
    let pi = stick_from_v(&v);
    (v, pi)
}

// slice envelope
const XI_RHO: f64 = 0.995;
const XI_C: f64 = 1.0 - XI_RHO;
pub fn xi_k(k: usize) -> f64 { XI_C * XI_RHO.powi(k as i32) }

pub fn resample_slices(rng: &mut StdRng, dp: &mut DPState, revealed_normal: Option<&[bool]>) {
    for i in 0..dp.u.len() {
        if let (Some(mask), Some(k0)) = (revealed_normal, dp.normal_k) {
            if mask[i] { dp.z[i] = k0; }
        }
        let k = dp.z[i];
        let cap = xi_k(k) * dp.pi[k];
        dp.u[i] = Uniform::new(0.0, cap).unwrap().sample(rng);
    }
}

pub fn active_k_from_slices(pi: &[f64], u: &[f64]) -> usize {
    let u_star = u.iter().fold(1.0_f64, |a, &b| a.min(b));
    let mut s = 0.0;
    for (k, &pik) in pi.iter().enumerate() {
        s += xi_k(k) * pik;
        if 1.0 - s < u_star {
            return k + 1;
        }
    }
    pi.len()
}

// log posterior for a cluster (ICM params + integrated mean)
pub fn log_post_icm_with_marginal(
    c: &Cluster,
    members: &[(usize, &CurveM)],
    t: &Array1<f64>,
    _x_block: &Array2<f64>, // (M*n) x (M*p) block-diag(X,...,X), but we integrate β so we can pass zero cols
    prior: &NIGPrior,
) -> f64 {
    // weak prior on ell (log-normal)
    let lell = c.icm.hyp.ell.ln();
    let lp_ell = -0.5 * ((lell - (-1.5)) / 0.7).powi(2);

    // LKJ-ish soft prior on B via log prior on diagonal (keep SPD by sampling in MH step)
    // For compactness, just penalize large Frobenius norm:
    let mut lp_b = 0.0;
    let b = &c.icm.b;
    let m = b.shape()[0];
    let mut frob = 0.0;
    for i in 0..m { for j in 0..m { frob += b[(i,j)]*b[(i,j)]; } }
    lp_b += -0.5 * frob / (m as f64);

    // per-output nuggets log-normal, broad
    let mut lp_eta = 0.0;
    for &e in c.icm.eta.iter() {
        lp_eta += -0.5 * ((e.ln() + 8.0)/2.0).powi(2);
    }

    // integrated likelihood across members
    let mut llik = 0.0;
    for &(_i, cur) in members.iter() {
        // stack outputs
        let (n, m_out) = (cur.y.shape()[0], cur.y.shape()[1]);
        let mut yst = Array1::<f64>::zeros(n * m_out);
        for a in 0..m_out {
            let ya = cur.y.column(a);
            for i in 0..n {
                yst[a*n + i] = ya[i];
            }
        }
        let (yt, xt, logdet) = c.icm.whiten_joint(&yst, t);
        let lml = log_marginal_whitened(&yt, &xt, prior, logdet);
        llik += lml;
    }

    lp_ell + lp_b + lp_eta + llik
}

// Metropolis updates for ICM params (ell, eta, B)
pub fn mh_update_icm(
    rng: &mut StdRng,
    c: &mut Cluster,
    t: &Array1<f64>,
    members: &[(usize, &CurveM)],
    x_block: &Array2<f64>,
    prior: &NIGPrior,
) {
    let target = 0.30;
    let rm_c = 0.01;

    // propose ell (log space)
    let mut prop = c.clone();
    let d_ell: f64 = Normal::new(0.0, c.icm.mh_step_ell).unwrap().sample(rng);
    prop.icm.hyp.ell = (c.icm.hyp.ell.ln() + d_ell).exp().clamp(1e-3, 5.0);
    // rebuild needed caches only if f-only were used; full joint builds on the fly.

    // propose η (log space, per-output)
    let mut prop2 = prop.clone();
    for e in prop2.icm.eta.iter_mut() {
        let de: f64 = Normal::new(0.0, c.icm.mh_step_eta).unwrap().sample(rng);
        *e = (e.ln() + de).exp().clamp(1e-8, 1.0);
    }

    // propose B by symmetric jitter + SPD repair (B' = Q Λ Q^T with Λ >= eps)
    let mut prop3 = prop2.clone();
    {
        let m = prop3.icm.b.shape()[0];
        // add small symmetric noise
        for i in 0..m {
            for j in i..m {
                let u: f64 = Normal::new(0.0, c.icm.mh_step_b).unwrap().sample(rng);
                prop3.icm.b[(i,j)] += u;
                if j != i { prop3.icm.b[(j,i)] = prop3.icm.b[(i,j)]; }
            }
        }
        // SPD repair via eigenvalue floor (simple; cost okay for small M)
        let (mut vals, vecs) = prop3.icm.b.clone().eigh(UPLO::Lower).expect("eigh");

        for v in vals.iter_mut() {
            *v = v.max(1e-3);
        }
        // reconstruct B = V diag(vals) V^T
        let d = Array2::from_diag(&vals);
        let tmp = vecs.dot(&d).dot(&vecs.t());
        prop3.icm.b = tmp;
    }

    // scores
    let s_cur = log_post_icm_with_marginal(c, members, t, x_block, prior);
    let s1 = log_post_icm_with_marginal(&prop, members, t, x_block, prior);
    let s2 = log_post_icm_with_marginal(&prop2, members, t, x_block, prior);
    let s3 = log_post_icm_with_marginal(&prop3, members, t, x_block, prior);

    let acc1 = (s1 - s_cur).exp().min(1.0);
    if Uniform::new(0.0,1.0).unwrap().sample(rng) < acc1 { *c = prop; }
    let acc2 = (s2 - s_cur).exp().min(1.0);
    if Uniform::new(0.0,1.0).unwrap().sample(rng) < acc2 { *c = prop2; }
    let acc3 = (s3 - s_cur).exp().min(1.0);
    if Uniform::new(0.0,1.0).unwrap().sample(rng) < acc3 { *c = prop3; }

    // adapt
    c.icm.mh_t += 1;
    let delta = rm_c / (c.icm.mh_t as f64);
    let upd = |s: f64| ((s - target) * delta).exp();
    c.icm.mh_step_ell = (c.icm.mh_step_ell * upd(acc1)).clamp(0.02, 0.5);
    c.icm.mh_step_eta = (c.icm.mh_step_eta * upd(acc2)).clamp(0.05, 0.8);
    c.icm.mh_step_b   = (c.icm.mh_step_b   * upd(acc3)).clamp(0.02, 0.5);
}

// Carlin–Chib family switching (SE/M32/M52) via pseudo-priors on ell
pub fn carlin_chib_switch(
    rng: &mut StdRng,
    c: &mut Cluster,
    members: &[(usize, &CurveM)],
    t: &Array1<f64>,
    x_block: &Array2<f64>,
    prior: &NIGPrior,
    fams: &[KernelFamily],
) {
    // set shadow ells from pseudo-priors
    for &fam in fams {
        let idx = ICM::fam_idx(fam);
        let logell: f64 = Normal::new(-1.5, 0.7).unwrap().sample(rng);
        c.icm.shadow_ell[idx] = logell.exp();
    }
    // evaluate weights
    let mut trials: Vec<(KernelFamily, Cluster, f64)> = Vec::new();
    for &fam in fams {
        let mut trial = c.clone();
        trial.icm.fam = fam;
        trial.icm.hyp.ell = c.icm.shadow_ell[ICM::fam_idx(fam)];
        let s = log_post_icm_with_marginal(&trial, members, t, x_block, prior);
        trials.push((fam, trial, s));
    }
    let ws: Vec<f64> = trials.iter().map(|(_,_,s)| *s).collect();
    let idx = categorical_from_logp(rng, &ws);
    *c = trials[idx].1.clone();
}

// ------------------------ Assignments ------------------------
pub fn resample_assignments(
    rng: &mut StdRng,
    dp: &mut DPState,
    data: &DatasetM,
    k_active: usize,
    revealed_normal: Option<&[bool]>,
    prior: &NIGPrior,
) {
    // Prebuild X_block (block-diag of per-output wavelet X). We integrate β out here so we won't use it.
    let _x_block_cols = data.m_out * data.p;
    let x_block_dummy = Array2::<f64>::zeros((data.m_out * data.t.len(), 0));

    for (i, cur) in data.curves.iter().enumerate() {
        if let (Some(mask), Some(k0)) = (revealed_normal, dp.normal_k) {
            if mask[i] { dp.z[i] = k0; continue; }
        }
        let mut logp: Vec<f64> = Vec::new();
        let mut ks: Vec<usize> = Vec::new();

        for k in 0..k_active {
            if dp.pi[k] * xi_k(k) <= dp.u[i] { continue; }
            let members = [(i, cur)];
            let s = log_post_icm_with_marginal(&dp.clusters[k], &members, &data.t, &x_block_dummy, prior);
            logp.push( s + (dp.pi[k] + 1e-12).ln() - (xi_k(k) + 1e-12).ln() );
            ks.push(k);
        }
        if ks.is_empty() {
            // fallback: evaluate all
            for k in 0..dp.clusters.len() {
                let members = [(i, cur)];
                let s = log_post_icm_with_marginal(&dp.clusters[k], &members, &data.t, &x_block_dummy, prior);
                logp.push( s + (dp.pi[k] + 1e-12).ln() - (xi_k(k) + 1e-12).ln() );
                ks.push(k);
            }
        }
        let idx = categorical_from_logp(rng, &logp);
        dp.z[i] = ks[idx];
    }
}

// Escobar–West update for alpha
pub fn update_alpha(
    rng: &mut StdRng, dp: &mut DPState, a_alpha: f64, b_alpha: f64, n: usize
) {
    let mut occ: HashSet<usize> = HashSet::new();
    for &k in dp.z.iter() { occ.insert(k); }
    let m = occ.len() as f64;
    let eta = Beta::new(dp.alpha + 1.0, n as f64).unwrap().sample(rng);
    let mix = (a_alpha + m - 1.0) / (n as f64 * (b_alpha - eta.ln()) + a_alpha + m - 1.0);
    let bern: f64 = Uniform::new(0.0,1.0).unwrap().sample(rng);
    let a_post = a_alpha + m - if bern < mix { 0.0 } else { 1.0 };
    let b_post = b_alpha - eta.ln();
    dp.alpha = Gamma::new(a_post, 1.0 / b_post).unwrap().sample(rng);
}
