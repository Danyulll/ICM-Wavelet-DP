use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use ndarray::{arr1, Array1};

// ------------------------ helpers ------------------------
pub fn logsumexp(mut xs: Vec<f64>) -> f64 {
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

pub fn categorical_from_logp(rng: &mut StdRng, logp: &[f64]) -> usize {
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

pub fn is_power_of_two(x: usize) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

pub fn linspace(a: f64, b: f64, n: usize) -> Array1<f64> {
    if n == 1 {
        return arr1(&[a]);
    }
    let step = (b - a) / (n as f64 - 1.0);
    Array1::from((0..n).map(|i| a + step * i as f64).collect::<Vec<_>>())
}

pub fn gaussian_window(t: &Array1<f64>, center: f64, width: f64) -> Array1<f64> {
    let mut w = Array1::<f64>::zeros(t.len());
    for i in 0..t.len() {
        let s = (t[i] - center) / width;
        w[i] = (-0.5 * s * s).exp();
    }
    w
}

// simple linear interp for small phase shifts (clamps at ends)
pub fn shift_series_linear(t: &Array1<f64>, y: &Array1<f64>, delta: f64) -> Array1<f64> {
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
