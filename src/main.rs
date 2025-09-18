
// DP–ICM–GP clustering with wavelet mean + shrinkage, Carlin–Chib switches,
// and Kalli/Walker slice sampling. Two multivariate datasets included:
// (A) anomaly detection, (B) regular clustering.
//
// Build: `cargo run --release`

#![allow(clippy::too_many_arguments)]
use ndarray::{arr1, s, Array1, Array2};
use ndarray_linalg::{Cholesky, Inverse, UPLO};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Beta, Distribution, Gamma, Normal, Uniform};
use statrs::function::gamma::ln_gamma;
use std::collections::HashSet;
use std::f64::consts::PI;
use ndarray_linalg::Eigh;   // <- for .eigh(...)
use rand::seq::SliceRandom;
use plotters::prelude::*;
use std::fs::create_dir_all;
use plotters::style::HSLColor; // for palette generation
use plotters::style::Palette99; // add this near your other imports



#[derive(Clone, Debug, PartialEq, Eq)]
enum AnomType {
    Normal,
    // add a constant offset per output; transient if windowed
    Shift,             // y <- y + c
    // multiply amplitude per output; transient if windowed
    Amplitude,         // y <- γ ⊙ y
    // local shape change (hi-freq bump); transient/persistent by width
    Shape,             // y <- y + bump(t)
    // linear drift
    Trend,             // y <- y + a*t + b
    // small time lag / phase shift via interpolation
    Phase,             // y(t) <- y(t - Δ)
    // break cross-output correlations (decouple)
    Decouple,          // y <- y + ε (indep across outputs)
    // smoothness/lengthscale change (local smoothing/sharpening)
    Smoothness,        // y <- (1-ρ) y + ρ * filt(y)
    // short high-variance noise burst
    NoiseBurst,        // y <- y + σ * N(0,I) on window
}


// Nice short names for filenames
fn anom_slug(t: &AnomType) -> &'static str {
    match t {
        AnomType::Normal    => "normal",
        AnomType::Shift     => "shift",
        AnomType::Amplitude => "amplitude",
        AnomType::Shape     => "shape",
        AnomType::Trend     => "trend",
        AnomType::Phase     => "phase",
        AnomType::Decouple  => "decouple",
        AnomType::Smoothness=> "smoothness",
        AnomType::NoiseBurst=> "noise_burst",
    }
}

fn make_single_anomaly_dataset(
    rng: &mut StdRng,
    n_curves: usize,
    which: AnomType,
) -> LabeledDatasetM {
    assert!(n_curves >= 2, "Need at least 2 curves to have 1 anomaly + normals");

    let n = 64;               // keep consistent with the multitype set
    let m_out = 3;
    let p = 16;
    let t = linspace(0.0, 1.0, n);
    let x = make_wavelet_design(n, p, DesignKind::Haar);

    // "normal" ICM (same spirit/scale as your multitype dataset)
    let fam = KernelFamily::Matern32;
    let kx_norm = build_kx(&t, fam, &KernelHyper { ell: 0.22 });
    let mut b_norm = Array2::<f64>::zeros((m_out, m_out));
    for i in 0..m_out { b_norm[(i,i)] = 1.0; }
    b_norm[(0,1)] = 0.75; b_norm[(1,0)] = 0.75;
    b_norm[(1,2)] = 0.5;  b_norm[(2,1)] = 0.5;
    let eta_norm = arr1(&[1e-3, 1.5e-3, 1e-3]);

    // draw all curves from the normal model
    let mut curves = simulate_icm_curves(rng, n, m_out, &kx_norm, &b_norm, &eta_norm, n_curves);

    // choose an index to flip to anomaly (random for variety)
    let anom_idx = Uniform::new(0, n_curves).unwrap().sample(rng);

    // labels: everything Normal except one curve with `which`
    let mut labels = vec![AnomType::Normal; n_curves];
    labels[anom_idx] = which.clone();

    // apply anomaly to that one curve
    let which_copy = which.clone();
    apply_anomaly_to_curve(rng, &mut curves[anom_idx], &t, &which_copy);

    // No shuffle (keeps the single anomaly at known index), but could shuffle if desired.
    LabeledDatasetM {
        ds: DatasetM { t, x, curves, m_out, p },
        labels,
    }
}

fn make_all_single_anomaly_datasets(
    rng: &mut StdRng,
    n_curves: usize,
) -> Vec<(String, LabeledDatasetM)> {
    let kinds = [
        AnomType::Shift,
        AnomType::Amplitude,
        AnomType::Shape,
        AnomType::Trend,
        AnomType::Phase,
        AnomType::Decouple,
        AnomType::Smoothness,
        AnomType::NoiseBurst,
    ];
    kinds.iter().map(|k| {
        let ds = make_single_anomaly_dataset(rng, n_curves, k.clone());
        (anom_slug(k).to_string(), ds)
    }).collect()
}


fn gaussian_window(t: &Array1<f64>, center: f64, width: f64) -> Array1<f64> {
    let mut w = Array1::<f64>::zeros(t.len());
    for i in 0..t.len() {
        let s = (t[i] - center) / width;
        w[i] = (-0.5 * s * s).exp();
    }
    w
}

// simple linear interp for small phase shifts (clamps at ends)
fn shift_series_linear(t: &Array1<f64>, y: &Array1<f64>, delta: f64) -> Array1<f64> {
    let n = t.len();
    let t0 = t[0];
    let t1 = t[n - 1];
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let ti = (t[i] - delta).clamp(t0, t1);
        // find neighbors
        let mut j = 0usize;
        while j + 1 < n && !(t[j] <= ti && ti <= t[j + 1]) { j += 1; }
        if j + 1 == n { out[i] = y[n - 1]; }
        else {
            let a = (ti - t[j]) / (t[j + 1] - t[j]).max(1e-12);
            out[i] = (1.0 - a) * y[j] + a * y[j + 1];
        }
    }
    out
}


fn plot_dataset_icm(ds: &DatasetM, labels: Option<&[bool]>, out_path: &str) {
    // Make sure the output folder exists (e.g., "plots/anomaly_overview.png")
    if let Some(parent) = std::path::Path::new(out_path).parent() {
        let _ = create_dir_all(parent);
    }

    let n = ds.t.len();
    let m_out = ds.m_out;

    // Precompute min/max per output for nice y-ranges
    let mut y_min = vec![f64::INFINITY; m_out];
    let mut y_max = vec![f64::NEG_INFINITY; m_out];
    for a in 0..m_out {
        for c in &ds.curves {
            for i in 0..n {
                let v = c.y[(i, a)];
                if v < y_min[a] { y_min[a] = v; }
                if v > y_max[a] { y_max[a] = v; }
            }
        }
        // Add a small pad to avoid clipping
        let pad = 0.05 * (y_max[a] - y_min[a]).max(1e-6);
        y_min[a] -= pad;
        y_max[a] += pad;
    }

    // Count normals/anomalies if labels provided (true = normal)
    let (mut n_norm, mut n_anom) = (0usize, 0usize);
    if let Some(lbl) = labels {
        for &b in lbl {
            if b { n_norm += 1 } else { n_anom += 1 }
        }
    }

    let root = BitMapBackend::new(out_path, (1200, 300 * m_out as u32)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let areas = root.split_evenly((m_out, 1));

    let x_start = ds.t[0];
    let x_end   = ds.t[n - 1];

    for a in 0..m_out {
        let mut chart = ChartBuilder::on(&areas[a])
            .margin(15)
            .set_left_and_bottom_label_area_size(40)
            .caption(
                format!(
                    "Dataset A • Output {}{}",
                    a,
                    if labels.is_some() {
                        format!("  |  normals: {}, anomalies: {}", n_norm, n_anom)
                    } else {
                        "".to_string()
                    }
                ),
                ("sans-serif", 18),
            )
            .build_cartesian_2d(x_start..x_end, y_min[a]..y_max[a])
            .unwrap();

        chart.configure_mesh()
            .x_desc("t")
            .y_desc("y")
            .label_style(("sans-serif", 12))
            .draw()
            .unwrap();

        // Draw curves: normals light gray, anomalies red (if labels given),
        // else draw all curves in blue.
        for (i, c) in ds.curves.iter().enumerate() {
            let series_color: ShapeStyle = if let Some(lbl) = labels {
                if lbl[i] {
                    // normal
                    (&RGBColor(160, 160, 160)).stroke_width(1) // light gray
                } else {
                    // anomaly
                    (&RED).stroke_width(2)
                }
            } else {
                (&BLUE).stroke_width(1)
            };

            // (t, y_a(t)) pairs for the a-th output
            let points = (0..n).map(|ix| (ds.t[ix], c.y[(ix, a)]));
            chart.draw_series(LineSeries::new(points, series_color)).unwrap();
        }
    }

    // Optional: annotate footer text
    root.titled(
        "Multivariate ICM dataset visualization",
        ("sans-serif", 14)
    ).ok();
}


// ------------------------ helpers ------------------------
fn logsumexp(mut xs: Vec<f64>) -> f64 {
    let m = xs
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    if !m.is_finite() {
        return m;
    }
    for x in xs.iter_mut() {
        *x = (*x - m).exp();
    }
    m + xs.iter().sum::<f64>().ln()
}
fn categorical_from_logp(rng: &mut StdRng, logp: &[f64]) -> usize {
    let z = logsumexp(logp.to_vec());
    let u: f64 = Uniform::new(0.0, 1.0).unwrap().sample(rng);
    let mut acc = 0.0;
    for (k, &lp) in logp.iter().enumerate() {
        let p = (lp - z).exp();
        acc += p;
        if u <= acc || k == logp.len() - 1 {
            return k;
        }
    }
    logp.len() - 1
}
fn is_power_of_two(x: usize) -> bool {
    x != 0 && (x & (x - 1)) == 0
}
fn linspace(a: f64, b: f64, n: usize) -> Array1<f64> {
    if n == 1 {
        return arr1(&[a]);
    }
    let step = (b - a) / (n as f64 - 1.0);
    Array1::from((0..n).map(|i| a + step * i as f64).collect::<Vec<_>>())
}

// ------------------------ kernels & ICM ------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelFamily {
    SE,
    Matern32,
    Matern52,
}
const AVAIL_FAMS: &[KernelFamily] = &[
    KernelFamily::SE,
    KernelFamily::Matern32,
    KernelFamily::Matern52,
];

#[derive(Clone, Copy, Debug)]
struct KernelHyper {
    ell: f64,
}
impl Default for KernelHyper {
    fn default() -> Self {
        Self { ell: 0.25 }
    }
}
fn se_corr(r: f64, ell: f64) -> f64 {
    (-(r * r) / (2.0 * ell * ell)).exp()
}
fn matern32_corr(r: f64, ell: f64) -> f64 {
    let s = (3.0f64).sqrt() * r / ell;
    (1.0 + s) * (-s).exp()
}
fn matern52_corr(r: f64, ell: f64) -> f64 {
    let s = (5.0f64).sqrt() * r / ell;
    (1.0 + s + (s * s) / 3.0) * (-s).exp()
}
fn base_corr(fam: KernelFamily, r: f64, ell: f64) -> f64 {
    match fam {
        KernelFamily::SE => se_corr(r, ell),
        KernelFamily::Matern32 => matern32_corr(r, ell),
        KernelFamily::Matern52 => matern52_corr(r, ell),
    }
}
fn build_kx(t: &Array1<f64>, fam: KernelFamily, h: &KernelHyper) -> Array2<f64> {
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

// ICM: K = B ⊗ Kx  +  Σ_noise ⊗ I_n
// Here we keep Σ_noise diagonal (per-output nugget), and B is SPD 2x2 or MxM.
#[derive(Clone)]
struct ICM {
    // coregionalization B (M x M), SPD
    b: Array2<f64>,
    // per-output nugget ratios η_m (scaled by sigma^2)
    eta: Array1<f64>,
    fam: KernelFamily,
    hyp: KernelHyper,
    // caches for f-only whitening per-output (Kx + η_m I)^{-1/2}
    l: Vec<Array2<f64>>,
    linv: Vec<Array2<f64>>,
    logdet_blocks: Vec<f64>,
    // Carlin–Chib shadows (for other families)
    shadow_ell: [f64; 3], // SE, M32, M52
    // MH steps
    mh_step_ell: f64,
    mh_step_eta: f64,
    mh_step_b: f64,
    mh_t: usize,
}
impl ICM {
    fn new(m_out: usize, fam: KernelFamily, hyp: KernelHyper, t: &Array1<f64>) -> Self {
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
            shadow_ell: [hyp.ell, hyp.ell, hyp.ell],
            mh_step_ell: 0.10,
            mh_step_eta: 0.25,
            mh_step_b: 0.10,
            mh_t: 0,
        };
        icm.rebuild_fonly_blocks(t);
        icm
    }

    fn fam_idx(f: KernelFamily) -> usize {
        match f {
            KernelFamily::SE => 0,
            KernelFamily::Matern32 => 1,
            KernelFamily::Matern52 => 2,
        }
    }

    // Rebuild per-output Cholesky of Kx + η_m I (for fast f-only whitening)
    fn rebuild_fonly_blocks(&mut self, t: &Array1<f64>) {
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
    fn whiten_joint(
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

        // jittered chol
        let eye = Array2::<f64>::eye(dim);
        let mut jitter = 0.0_f64;
        let max_jitter = 1e-2_f64;
        let l = loop {
            let try_k = &k + &(eye.clone() * jitter);
            match try_k.cholesky(UPLO::Lower) {
                Ok(l) => break l,
                Err(_) => {
                    jitter = if jitter == 0.0 { 1e-10 } else { jitter * 10.0 };
                    if jitter > max_jitter { panic!("chol failed full ICM"); }
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

// ------------------------ Wavelet mean & shrinkage ------------------------
#[derive(Clone, Copy)]
enum DesignKind { Haar, Daub4, }
fn build_haar_matrix(m: usize) -> Array2<f64> {
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
// Build design X (n x p) for ONE output; we’ll block-diag it for M outputs.
fn make_wavelet_design(n: usize, p: usize, kind: DesignKind) -> Array2<f64> {
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
fn build_shrink_v0(p: usize, n: usize) -> Array2<f64> {
    // assign levels ~ dyadic, heavier shrink at high freq
    let mut v = Array2::<f64>::eye(p);
    let mut level = 0usize;
    let mut used = 1usize; // DC
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
struct NIGPrior {
    m0: Array1<f64>,
    v0: Array2<f64>,
    a0: f64,
    b0: f64,
}

fn log_marginal_whitened(
    y_t: &Array1<f64>,
    x_t: &Array2<f64>,
    prior: &NIGPrior,
    logdet_k: f64
) -> f64 {
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

// ------------------------ Data structures ------------------------
#[derive(Clone)]
struct CurveM {
    // M outputs on a common grid t
    y: Array2<f64>, // shape (n, M)
}
#[derive(Clone)]
struct DatasetM {
    t: Array1<f64>, // length n
    x: Array2<f64>, // (n, p) wavelet design used for all outputs (block-diag later)
    curves: Vec<CurveM>,
    m_out: usize,
    p: usize,
}

#[derive(Clone)]
struct Cluster {
    icm: ICM,
    // mean coefficients β per OUTPUT share the same design but are separate per output:
    // To keep the shrinkage simple and compact here, we model the mean stacked across outputs
    // via block-diag(X,...,X) and a single β vector of length M*p.
    beta: Array1<f64>,
    sigma2: f64, // global latent scale for mean model
}
impl Cluster {
    fn new(m_out: usize, fam: KernelFamily, hyp: KernelHyper, t: &Array1<f64>, p: usize) -> Self {
        Self {
            icm: ICM::new(m_out, fam, hyp, t),
            beta: Array1::<f64>::zeros(m_out * p),
            sigma2: 0.1,
        }
    }
}

// DP state
#[derive(Clone)]
struct DPState {
    alpha: f64,
    v: Vec<f64>,
    pi: Vec<f64>,
    clusters: Vec<Cluster>,
    z: Vec<usize>,
    u: Vec<f64>,
    normal_k: Option<usize>, // reserved normal cluster (for anomaly dataset)
}

// stick-breaking helpers
fn stick_from_v(v: &[f64]) -> Vec<f64> {
    let mut pis = vec![0.0; v.len()];
    let mut prod = 1.0;
    for (k, &vk) in v.iter().enumerate() {
        pis[k] = prod * vk;
        prod *= 1.0 - vk;
    }
    pis
}
fn resample_v_and_pi(rng: &mut StdRng, alpha: f64, z: &[usize], kmax: usize) -> (Vec<f64>, Vec<f64>) {
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
fn xi_k(k: usize) -> f64 { XI_C * XI_RHO.powi(k as i32) }

fn resample_slices(rng: &mut StdRng, dp: &mut DPState, revealed_normal: Option<&[bool]>) {
    for i in 0..dp.u.len() {
        if let (Some(mask), Some(k0)) = (revealed_normal, dp.normal_k) {
            if mask[i] { dp.z[i] = k0; }
        }
        let k = dp.z[i];
        let cap = xi_k(k) * dp.pi[k];
        dp.u[i] = Uniform::new(0.0, cap).unwrap().sample(rng);
    }
}
fn active_k_from_slices(pi: &[f64], u: &[f64]) -> usize {
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
fn log_post_icm_with_marginal(
    c: &Cluster,
    members: &[(usize, &CurveM)],
    t: &Array1<f64>,
    x_block: &Array2<f64>, // (M*n) x (M*p) block-diag(X,...,X), but we integrate β so we can pass zero cols
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
fn mh_update_icm(
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
fn carlin_chib_switch(
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
fn resample_assignments(
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
fn update_alpha(
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

// ------------------------ Simulated datasets (multivariate) ------------------------
fn simulate_icm_curves(
    rng: &mut StdRng,
    n: usize,
    m_out: usize,
    kx: &Array2<f64>,
    b: &Array2<f64>,
    eta: &Array1<f64>,
    n_curves: usize,
) -> Vec<CurveM> {
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
    let mut jitter = 0.0_f64;
    let max_jitter = 1e-2_f64;
    let l = loop {
        let try_k = &k + &(eye.clone() * jitter);
        match try_k.clone().cholesky(UPLO::Lower) {
            Ok(l) => break l,
            Err(_) => {
                jitter = if jitter == 0.0 { 1e-10 } else { jitter * 10.0 };
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
        out.push(CurveM { y });
    }
    out
}

fn apply_anomaly_to_curve(
    rng: &mut StdRng,
    curve: &mut CurveM,
    t: &Array1<f64>,
    kind: &AnomType,
) {
    let n = t.len();
    let m = curve.y.shape()[1];

    match kind {
        AnomType::Normal => { /* no-op */ }

        AnomType::Shift => {
            let center: f64 = Uniform::new(0.3_f64, 0.7_f64).unwrap().sample(rng);
            let width:  f64 = Uniform::new(0.05_f64, 0.15_f64).unwrap().sample(rng);
            let transient = Uniform::new(0.0_f64, 1.0_f64).unwrap().sample(rng) < 0.6_f64;
            let w = if transient { gaussian_window(t, center, width) } else { Array1::from_vec(vec![1.0_f64; n]) };

            for a in 0..m {
                let c: f64 = Normal::new(0.3_f64, 0.2_f64).unwrap().sample(rng);
                for i in 0..n { curve.y[(i, a)] += c * w[i]; }
            }
        }

        AnomType::Amplitude => {
            let center: f64 = Uniform::new(0.25_f64, 0.8_f64).unwrap().sample(rng);
            let width:  f64 = Uniform::new(0.05_f64, 0.18_f64).unwrap().sample(rng);
            let transient = Uniform::new(0.0_f64, 1.0_f64).unwrap().sample(rng) < 0.5_f64;
            let w = if transient { gaussian_window(t, center, width) } else { Array1::from_vec(vec![1.0_f64; n]) };

            for a in 0..m {
                let mut gamma: f64 = Normal::new(1.6_f64, 0.3_f64).unwrap().sample(rng);
                if gamma < 0.2_f64 { gamma = 0.2_f64; }
                for i in 0..n {
                    curve.y[(i, a)] = (1.0_f64 + (gamma - 1.0_f64) * w[i]) * curve.y[(i, a)];
                }
            }
        }

        AnomType::Shape => {
            let center: f64 = Uniform::new(0.2_f64, 0.8_f64).unwrap().sample(rng);
            let width:  f64 = Uniform::new(0.02_f64, 0.08_f64).unwrap().sample(rng);
            let mut amp: f64 = Normal::new(0.7_f64, 0.3_f64).unwrap().sample(rng);
            if amp < 0.0 { amp = -amp; }
            let w = gaussian_window(t, center, width);
            for a in 0..m {
                let freq: f64 = Uniform::new(8.0_f64, 16.0_f64).unwrap().sample(rng);
                for i in 0..n {
                    let bump = amp * w[i] * (2.0_f64 * PI * freq * t[i]).sin();
                    curve.y[(i, a)] += bump;
                }
            }
        }

        AnomType::Trend => {
            for a in 0..m {
                let slope: f64 = Normal::new(0.8_f64, 0.4_f64).unwrap().sample(rng);
                let bias:  f64 = Normal::new(0.0_f64, 0.2_f64).unwrap().sample(rng);
                for i in 0..n {
                    curve.y[(i, a)] += bias + slope * (t[i] - t[0]);
                }
            }
        }

        AnomType::Phase => {
            let delta: f64 = Uniform::new(-0.06_f64, 0.06_f64).unwrap().sample(rng);
            for a in 0..m {
                let col = curve.y.column(a).to_owned();
                let shifted = shift_series_linear(t, &col, delta);
                for i in 0..n { curve.y[(i, a)] = shifted[i]; }
            }
        }

        AnomType::Decouple => {
            for a in 0..m {
                let sigma: f64 = Uniform::new(0.3_f64, 0.8_f64).unwrap().sample(rng);
                for i in 0..n {
                    curve.y[(i, a)] += Normal::new(0.0_f64, sigma).unwrap().sample(rng);
                }
            }
        }

        AnomType::Smoothness => {
            let rho: f64 = Uniform::new(0.3_f64, 0.7_f64).unwrap().sample(rng);
            for a in 0..m {
                let mut sm = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let mut s: f64 = 0.0;
                    let mut cnt: f64 = 0.0;
                    for k in -2..=2 {
                        let j = (i as isize + k) as isize;
                        if 0 <= j && j < n as isize {
                            s += curve.y[(j as usize, a)];
                            cnt += 1.0_f64;
                        }
                    }
                    sm[i] = s / if cnt > 0.0 { cnt } else { 1.0_f64 };
                }
                for i in 0..n {
                    curve.y[(i, a)] = (1.0_f64 - rho) * curve.y[(i, a)] + rho * sm[i];
                }
            }
        }

        AnomType::NoiseBurst => {
            let center: f64 = Uniform::new(0.2_f64, 0.85_f64).unwrap().sample(rng);
            let width:  f64 = Uniform::new(0.02_f64, 0.06_f64).unwrap().sample(rng);
            let w = gaussian_window(t, center, width);
            for a in 0..m {
                let sigma: f64 = Uniform::new(1.0_f64, 2.0_f64).unwrap().sample(rng);
                for i in 0..n {
                    let z: f64 = Normal::new(0.0_f64, sigma).unwrap().sample(rng);
                    curve.y[(i, a)] += w[i] * z;
                }
            }
        }
    }
}



struct LabeledDatasetM {
    ds: DatasetM,
    labels: Vec<AnomType>, // per-curve ground truth
}

// proportion vector (must sum ≈ 1.0 over anomaly types)
fn make_multitype_anomaly_dataset(
    rng: &mut StdRng,
    n_curves: usize,
    contam: f64,
) -> LabeledDatasetM {
    let n = 64;          // dyadic grid still OK for your Haar design
    let m_out = 3;
    let p = 16;
    let t = linspace(0.0, 1.0, n);
    let x = make_wavelet_design(n, p, DesignKind::Haar);

    // "normal" ICM (same spirit as before)
    let fam = KernelFamily::Matern32;
    let kx_norm = build_kx(&t, fam, &KernelHyper { ell: 0.22 });
    let mut b_norm = Array2::<f64>::zeros((m_out, m_out));
    for i in 0..m_out { b_norm[(i,i)] = 1.0; }
    b_norm[(0,1)] = 0.75; b_norm[(1,0)] = 0.75;
    b_norm[(1,2)] = 0.5;  b_norm[(2,1)] = 0.5;
    let eta_norm = arr1(&[1e-3, 1.5e-3, 1e-3]);

    // draw base normal curves
    let n_anom = ((contam * n_curves as f64).round() as usize).max(1);
    let n_norm = n_curves - n_anom;
    let mut curves = simulate_icm_curves(rng, n, m_out, &kx_norm, &b_norm, &eta_norm, n_curves);

    // assign anomaly types to the last n_anom indices
    let mut labels = vec![AnomType::Normal; n_curves];
    let kinds = [
        AnomType::Shift,
        AnomType::Amplitude,
        AnomType::Shape,
        AnomType::Trend,
        AnomType::Phase,
        AnomType::Decouple,
        AnomType::Smoothness,
        AnomType::NoiseBurst,
    ];
    // choose types (roughly uniform) for anomalies
    for i in n_norm..n_curves {
        let k = Uniform::new(0, kinds.len()).unwrap().sample(rng);
        labels[i] = kinds[k].clone();
    }

    // apply anomalies in-place
    for i in n_norm..n_curves {
        let kind = labels[i].clone();
        // apply_anomaly_to_curve(rng, &mut curves[i], &kind, &t);
        apply_anomaly_to_curve(rng, &mut curves[i], &t, &kind);
    }

    // shuffle together (keep labels aligned)
    let mut idx: Vec<usize> = (0..n_curves).collect();
    idx.shuffle(rng);
    let curves = idx.iter().map(|&i| curves[i].clone()).collect::<Vec<_>>();
    let labels = idx.iter().map(|&i| labels[i].clone()).collect::<Vec<_>>();

    LabeledDatasetM {
        ds: DatasetM { t, x, curves, m_out, p },
        labels,
    }
}

fn plot_dataset_colored(ds: &DatasetM, labels: &[AnomType], out_path: &str) {
    use plotters::prelude::*;
    use std::fs::create_dir_all;

    if let Some(parent) = std::path::Path::new(out_path).parent() { let _ = create_dir_all(parent); }

    let n = ds.t.len();
    let m_out = ds.m_out;

    // y-range per output
    let mut y_min = vec![f64::INFINITY; m_out];
    let mut y_max = vec![f64::NEG_INFINITY; m_out];
    for a in 0..m_out {
        for c in &ds.curves {
            for i in 0..n {
                let v = c.y[(i, a)];
                if v < y_min[a] { y_min[a] = v; }
                if v > y_max[a] { y_max[a] = v; }
            }
        }
        let pad = 0.05 * (y_max[a] - y_min[a]).max(1e-6);
        y_min[a] -= pad; y_max[a] += pad;
    }

    // color map by type
    let color_for = |t: &AnomType| -> RGBColor {
        match t {
            AnomType::Normal    => RGBColor(160,160,160),
            AnomType::Shift     => RED,
            AnomType::Amplitude => BLUE,
            AnomType::Shape     => MAGENTA,
            AnomType::Trend     => GREEN,
            AnomType::Phase     => CYAN,
            AnomType::Decouple  => RGBColor(255,140,0),   // dark orange
            AnomType::Smoothness=> RGBColor(128,0,128),   // purple
            AnomType::NoiseBurst=> RGBColor(139,69,19),   // saddlebrown
        }
    };

    let mut counts: std::collections::BTreeMap<&'static str, usize> = Default::default();
    let name = |t: &AnomType| -> &'static str {
        match t {
            AnomType::Normal=>"Normal", AnomType::Shift=>"Shift",
            AnomType::Amplitude=>"Amplitude", AnomType::Shape=>"Shape",
            AnomType::Trend=>"Trend", AnomType::Phase=>"Phase",
            AnomType::Decouple=>"Decouple", AnomType::Smoothness=>"Smoothness",
            AnomType::NoiseBurst=>"NoiseBurst",
        }
    };
    for ty in labels { *counts.entry(name(ty)).or_insert(0) += 1; }

    let root = BitMapBackend::new(out_path, (1300, 320 * m_out as u32)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let areas = root.split_evenly((m_out, 1));
    let x0 = ds.t[0]; let x1 = ds.t[n-1];

    for a in 0..m_out {
        let mut chart = ChartBuilder::on(&areas[a])
            .margin(15).set_left_and_bottom_label_area_size(40)
            .caption(
                format!("Multitype anomalies • Output {}  |  {}", a,
                    counts.iter().map(|(k,v)| format!("{}:{}",k,v)).collect::<Vec<_>>().join("  ")),
                ("sans-serif", 18))
            .build_cartesian_2d(x0..x1, y_min[a]..y_max[a]).unwrap();

        chart.configure_mesh().x_desc("t").y_desc("y").label_style(("sans-serif", 12)).draw().unwrap();

        for (i, c) in ds.curves.iter().enumerate() {
            let col = color_for(&labels[i]);
            let style = (&col).stroke_width(if labels[i]==AnomType::Normal {1} else {2});
            let pts = (0..n).map(|ix| (ds.t[ix], c.y[(ix, a)]));
            chart.draw_series(LineSeries::new(pts, style)).unwrap();
        }
    }
    let _ = root.titled("Multivariate ICM dataset visualization (colored by anomaly type)", ("sans-serif", 14));
}

fn plot_by_cluster(ds: &DatasetM, z: &[usize], out_path: &str) {
    use plotters::prelude::*;
    use std::fs::create_dir_all;

    if let Some(parent) = std::path::Path::new(out_path).parent() { let _ = create_dir_all(parent); }

    let n = ds.t.len();
    let m_out = ds.m_out;
    let kmax = 1 + z.iter().copied().max().unwrap_or(0);

    // palette
    // palette
// palette
// palette
// palette as ready-to-use styles
let palette: Vec<ShapeStyle> = (0..kmax)
    .map(|k| Palette99::pick(k).stroke_width(2))
    .collect();




    // y-range per output
    let mut y_min = vec![f64::INFINITY; m_out];
    let mut y_max = vec![f64::NEG_INFINITY; m_out];
    for a in 0..m_out {
        for c in &ds.curves {
            for i in 0..n {
                let v = c.y[(i, a)];
                if v < y_min[a] { y_min[a] = v; }
                if v > y_max[a] { y_max[a] = v; }
            }
        }
        let pad = 0.05 * (y_max[a] - y_min[a]).max(1e-6);
        y_min[a] -= pad; y_max[a] += pad;
    }

    // counts
    let mut counts = vec![0usize; kmax];
    for &zi in z { counts[zi] += 1; }

    let root = BitMapBackend::new(out_path, (1200, 320 * m_out as u32)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let areas = root.split_evenly((m_out, 1));
    let x0 = ds.t[0]; let x1 = ds.t[n-1];

    for a in 0..m_out {
        let mut chart = ChartBuilder::on(&areas[a])
            .margin(15).set_left_and_bottom_label_area_size(40)
            .caption(
                format!("Post-clustering • Output {}  |  {}", a,
                    (0..kmax).map(|k| format!("k{}:{}",k,counts[k])).collect::<Vec<_>>().join("  ")),
                ("sans-serif", 18))
            .build_cartesian_2d(x0..x1, y_min[a]..y_max[a]).unwrap();

        chart.configure_mesh().x_desc("t").y_desc("y").label_style(("sans-serif", 12)).draw().unwrap();

        for (i, c) in ds.curves.iter().enumerate() {
            let style = palette[z[i]].clone();
let pts = (0..n).map(|ix| (ds.t[ix], c.y[(ix, a)]));
chart.draw_series(LineSeries::new(pts, style)).unwrap();

        }
    }
    let _ = root.titled("Curves colored by assigned cluster", ("sans-serif", 14));
}


// Build two datasets:
// A) anomaly: one normal cluster params, inject anomalies with different B or lengthscale
// B) clustering: K clusters with different B/ell, mixed
fn make_datasets(rng: &mut StdRng) -> (DatasetM, Vec<bool>, DatasetM) {
    let n = 32;              // time points (dyadic, matches Haar)
    let m_out = 3;           // outputs/channels
    let p = 16;              // wavelet columns used in mean
    let t = linspace(0.0, 1.0, n);

    // common wavelet design (n x p)
    let x = make_wavelet_design(n, p, DesignKind::Haar);

    // --- (A) anomaly detection ---
    let fam = KernelFamily::Matern32;
    let kx_norm = build_kx(&t, fam, &KernelHyper { ell: 0.20 });
    // coregionalization for normal
    let mut b_norm = Array2::<f64>::zeros((m_out, m_out));
    for i in 0..m_out { b_norm[(i,i)] = 1.0; }
    b_norm[(0,1)] = 0.8; b_norm[(1,0)] = 0.8;
    b_norm[(1,2)] = 0.6; b_norm[(2,1)] = 0.6;
    let eta_norm = arr1(&[1e-3, 1.5e-3, 1e-3]);

    // anomalies: weaker cross-corr and longer ell
    let kx_anom = build_kx(&t, fam, &KernelHyper { ell: 0.45 });
    let mut b_anom = Array2::<f64>::zeros((m_out, m_out));
    for i in 0..m_out { b_anom[(i,i)] = 1.0; }
    b_anom[(0,1)] = 0.2; b_anom[(1,0)] = 0.2;
    b_anom[(1,2)] = 0.1; b_anom[(2,1)] = 0.1;
    let eta_anom = arr1(&[2e-3, 2e-3, 2e-3]);

    let n_curves_a = 80;
    let contam = 0.20;
    let n_anom = ((contam * n_curves_a as f64).round() as usize).max(1);
    let n_norm = n_curves_a - n_anom;

    let mut curves_a = simulate_icm_curves(rng, n, m_out, &kx_norm, &b_norm, &eta_norm, n_norm);
    let mut anoms = simulate_icm_curves(rng, n, m_out, &kx_anom, &b_anom, &eta_anom, n_anom);
    let mut labels_a = vec![true; n_norm]; // true = normal
    labels_a.extend(std::iter::repeat(false).take(n_anom));
    curves_a.append(&mut anoms);
    // shuffle
    {
        let mut idx: Vec<usize> = (0..curves_a.len()).collect();
        idx.shuffle(rng);
        curves_a = idx.iter().map(|&i| curves_a[i].clone()).collect();
        labels_a = idx.iter().map(|&i| labels_a[i]).collect();
    }
    let ds_a = DatasetM { t: t.clone(), x: x.clone(), curves: curves_a, m_out, p };

    // --- (B) clustering ---
    let kx1 = build_kx(&t, KernelFamily::SE, &KernelHyper{ ell: 0.15 });
    let kx2 = build_kx(&t, KernelFamily::Matern52, &KernelHyper{ ell: 0.35 });
    let mut b1 = Array2::<f64>::zeros((m_out, m_out));
    for i in 0..m_out { b1[(i,i)] = 1.0; }
    b1[(0,2)] = 0.7; b1[(2,0)] = 0.7;
    let eta1 = arr1(&[1e-3, 1e-3, 2e-3]);

    let mut b2 = Array2::<f64>::zeros((m_out, m_out));
    for i in 0..m_out { b2[(i,i)] = 1.0; }
    b2[(0,1)] = 0.5; b2[(1,0)] = 0.5;
    b2[(1,2)] = 0.4; b2[(2,1)] = 0.4;
    let eta2 = arr1(&[1.5e-3, 1e-3, 1e-3]);

    let c1 = simulate_icm_curves(rng, n, m_out, &kx1, &b1, &eta1, 40);
    let c2 = simulate_icm_curves(rng, n, m_out, &kx2, &b2, &eta2, 40);
    let mut curves_b = Vec::new(); curves_b.extend(c1); curves_b.extend(c2);
    curves_b.shuffle(rng);
    let ds_b = DatasetM { t, x, curves: curves_b, m_out, p };

    (ds_a, labels_a, ds_b)
}

// ------------------------ Runner ------------------------
fn run_dataset(
    title: &str,
    rng: &mut StdRng,
    data: &DatasetM,
    revealed_normal: Option<Vec<bool>>,
    post_plot_path: Option<&str>,   // NEW
) {
    let n = data.curves.len();
    let m_out = data.m_out;
    let p = data.p;
    let t = &data.t;

    // wavelet shrinkage prior (block for all outputs, but we integrate β out, so keep it simple)
    let v0 = build_shrink_v0(p, t.len());
    let prior = NIGPrior {
        m0: Array1::zeros(0),             // no explicit cols; integrated in whitened form
        v0: Array2::zeros((0,0)),        // (we use xt with 0 columns -> valid marginal)
        a0: 2.0,
        b0: 0.5,
    };

    // DP init
    let kmax = 8usize;
    let mut clusters = Vec::with_capacity(kmax);
    for _ in 0..kmax {
        let fam = AVAIL_FAMS[Uniform::new(0, AVAIL_FAMS.len()).unwrap().sample(rng)];
        let hyp = KernelHyper { ell: Uniform::new(0.08, 0.40).unwrap().sample(rng) };
        let c = Cluster::new(m_out, fam, hyp, t, p);
        clusters.push(c);
    }
    let normal_k = if revealed_normal.is_some() { Some(0usize) } else { None };
    let mut z = vec![0usize; n];
    for i in 0..n {
        z[i] = if let (Some(mask), Some(k0)) = (&revealed_normal, normal_k) {
            if mask[i] { k0 } else { Uniform::new(0, kmax).unwrap().sample(rng) }
        } else {
            Uniform::new(0, kmax).unwrap().sample(rng)
        };
    }
    let alpha0 = 8.0;
    let (v, pi) = resample_v_and_pi(rng, alpha0, &z, kmax);
    let mut dp = DPState {
        alpha: alpha0,
        v,
        pi,
        clusters,
        z,
        u: vec![0.0; n],
        normal_k,
    };

    // sampler controls
    let iters = 1200usize;
    let burnin = 600usize;
    let thin = 5usize;

    // main loop
    let mut kept = 0usize;
    for it in 1..=iters {
        // slices
        resample_slices(rng, &mut dp, revealed_normal.as_deref());

        // active K
        let k_active = active_k_from_slices(&dp.pi, &dp.u);

        // assignments
        resample_assignments(rng, &mut dp, data, k_active, revealed_normal.as_deref(), &prior);

        // sticks
        let (v, pi) = resample_v_and_pi(rng, dp.alpha, &dp.z, kmax);
        dp.v = v; dp.pi = pi;

        // members per cluster
        let mut members: Vec<Vec<(usize, &CurveM)>> = vec![Vec::new(); kmax];
        for (i, zi) in dp.z.iter().copied().enumerate() {
            members[zi].push((i, &data.curves[i]));
        }

        // per-cluster param updates
        for k in 0..kmax {
            if members[k].is_empty() { continue; }
            mh_update_icm(rng, &mut dp.clusters[k], &data.t, members[k].as_slice(), &Array2::zeros((0,0)), &prior);
            carlin_chib_switch(rng, &mut dp.clusters[k], members[k].as_slice(), &data.t, &Array2::zeros((0,0)), &prior, AVAIL_FAMS);
        }

        // alpha
        update_alpha(rng, &mut dp, 20.0, 1.0, n);

        if it > burnin && ((it - burnin) % thin == 0) {
            kept += 1;
        }

        if it % 200 == 0 || it == 1 {
            let k_occ = (0..kmax).filter(|&k| !members[k].is_empty()).count();
            println!("[{}] it {:4} | active {:2} | occupied {:2} | kept {:4}", title, it, k_active, k_occ, kept);
        }

         
    }

    // simple report: cluster sizes
    let mut counts = vec![0usize; kmax];
    for &zi in dp.z.iter() { counts[zi] += 1; }
    println!("\n[{}] Final cluster sizes (nonzero):", title);
    for (k,c) in counts.iter().enumerate() {
        if *c > 0 {
            println!("  k{:02}: {}", k, c);
        }
    }

    if let (Some(mask), Some(k0)) = (revealed_normal, dp.normal_k) {
        // binary metrics: Normal=1 if assigned to k0
        let y_true: Vec<bool> = mask; // revealed normals only (unknowns ignored for “truth”)
        let y_pred: Vec<bool> = (0..n).map(|i| dp.z[i] == k0).collect();
        let mut tp=0; let mut fp=0; let mut fn_=0;
        for i in 0..n {
            if !y_true[i] { continue; } // evaluate only revealed normals like your semi-supervised flow
            let pred_pos = y_pred[i];
            let is_pos = true;
            match (is_pos, pred_pos) {
                (true,true) => tp += 1,
                (true,false)=> fn_ += 1,
                (false,true)=> fp += 1,
                _ => {}
            }
        }
        let prec = if tp+fp>0 { tp as f64/(tp+fp) as f64 } else { 0.0 };
        let rec  = if tp+fn_>0 { tp as f64/(tp+fn_) as f64 } else { 0.0 };
        let f1 = if prec+rec>0.0 { 2.0*prec*rec/(prec+rec) } else { 0.0 };
        println!("[{}] Semi-supervised (revealed normals) F1 = {:.3} (tp {}, fp {}, fn {})", title, f1, tp, fp, fn_);
    }

    if let Some(path) = post_plot_path {
    plot_by_cluster(data, &dp.z, path);
    println!("[{}] Wrote post-clustering plot: {}", title, path);
}

}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // ===== Multitype anomaly dataset (mixture of anomaly types) =====
    let mult = make_multitype_anomaly_dataset(&mut rng, 120, 0.25);
    plot_dataset_colored(&mult.ds, &mult.labels, "plots/multitype_before.png");
    println!("Wrote plot: plots/multitype_before.png");

    run_dataset(
        "A_plus_multitype",
        &mut rng,
        &mult.ds,
        None, // (pass Some(mask) if doing semi-supervised)
        Some("plots/multitype_after.png"),
    );

    // ===== One dataset per anomaly type, with exactly 1 anomalous curve =====
    // (80 curves total by default; adjust if you like)
    for (slug, pack) in make_all_single_anomaly_datasets(&mut rng, 80) {
        let before = format!("plots/single_{}_before.png", slug);
        plot_dataset_colored(&pack.ds, &pack.labels, &before);
        println!("Wrote plot: {}", before);

        let after = format!("plots/single_{}_after.png", slug);
        run_dataset(
            &format!("single_anom_{}", slug),
            &mut rng,
            &pack.ds,
            None,
            Some(&after),
        );
    }

    // ===== Original demos (A = anomaly vs. different kernel; B = clustering) =====
    let (ds_anom, labels_norm, ds_cluster) = make_datasets(&mut rng);

    // A: anomaly demo — BEFORE
    plot_dataset_icm(&ds_anom, Some(&labels_norm), "plots/A_before.png");
    println!("Wrote plot: plots/A_before.png");
    // A: anomaly demo — AFTER
    run_dataset(
        "A_ICM_anomaly",
        &mut rng,
        &ds_anom,
        Some(labels_norm),
        Some("plots/A_after.png"),
    );

    // B: clustering demo — BEFORE
    plot_dataset_icm(&ds_cluster, None, "plots/B_before.png");
    println!("Wrote plot: plots/B_before.png");
    // B: clustering demo — AFTER
    run_dataset(
        "B_ICM_cluster",
        &mut rng,
        &ds_cluster,
        None,
        Some("plots/B_after.png"),
    );

    println!("All datasets generated and plotted (before & after).");
}
