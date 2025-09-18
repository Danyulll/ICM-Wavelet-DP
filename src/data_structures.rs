use ndarray::{Array1, Array2};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnomType {
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

#[derive(Clone)]
pub struct CurveM {
    // M outputs on a common grid t
    pub y: Array2<f64>, // shape (n, M)
}

#[derive(Clone)]
pub struct DatasetM {
    pub t: Array1<f64>, // length n
    pub x: Array2<f64>, // (n, p) wavelet design used for all outputs (block-diag later)
    pub curves: Vec<CurveM>,
    pub m_out: usize,
    pub p: usize,
}

#[derive(Clone)]
pub struct Cluster {
    pub icm: crate::icm::ICM,
    // mean coefficients β per OUTPUT share the same design but are separate per output:
    // To keep the shrinkage simple and compact here, we model the mean stacked across outputs
    // via block-diag(X,...,X) and a single β vector of length M*p.
    pub beta: Array1<f64>,
    pub sigma2: f64, // global latent scale for mean model
}

// DP state
#[derive(Clone)]
pub struct DPState {
    pub alpha: f64,
    pub v: Vec<f64>,
    pub pi: Vec<f64>,
    pub clusters: Vec<Cluster>,
    pub z: Vec<usize>,
    pub u: Vec<f64>,
    pub normal_k: Option<usize>, // reserved normal cluster (for anomaly dataset)
}

pub struct LabeledDatasetM {
    pub ds: DatasetM,
    pub labels: Vec<AnomType>, // per-curve ground truth
}

impl Cluster {
    pub fn new(m_out: usize, fam: crate::kernels::KernelFamily, hyp: crate::kernels::KernelHyper, t: &Array1<f64>, p: usize) -> Self {
        Self {
            icm: crate::icm::ICM::new(m_out, fam, hyp, t),
            beta: Array1::<f64>::zeros(m_out * p),
            sigma2: 0.1,
        }
    }
}
