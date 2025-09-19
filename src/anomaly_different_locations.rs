use rand::{rngs::StdRng, seq::SliceRandom};
use rand_distr::{Distribution, Normal, Uniform};
use ndarray::{Array1, Array2};
use crate::data_structures::{AnomType, CurveM, DatasetM, LabeledDatasetM};
use crate::utils::{gaussian_window, shift_series_linear};
use crate::kernels::{KernelFamily, KernelHyper, build_kx};
use crate::icm::simulate_icm_curves;

// Apply anomaly with different temporal locations per channel
pub fn apply_anomaly_to_curve_different_locations(
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
            for a in 0..m {
                let center: f64 = Uniform::new(0.3_f64, 0.7_f64).unwrap().sample(rng);
                let width:  f64 = Uniform::new(0.05_f64, 0.15_f64).unwrap().sample(rng);
                let transient = Uniform::new(0.0_f64, 1.0_f64).unwrap().sample(rng) < 0.6_f64;
                let w = if transient { gaussian_window(t, center, width) } else { Array1::from_vec(vec![1.0_f64; n]) };

                let c: f64 = Normal::new(0.3_f64, 0.2_f64).unwrap().sample(rng);
                for i in 0..n { curve.y[(i, a)] += c * w[i]; }
            }
        }

        AnomType::Amplitude => {
            for a in 0..m {
                let center: f64 = Uniform::new(0.25_f64, 0.8_f64).unwrap().sample(rng);
                let width:  f64 = Uniform::new(0.05_f64, 0.18_f64).unwrap().sample(rng);
                let transient = Uniform::new(0.0_f64, 1.0_f64).unwrap().sample(rng) < 0.5_f64;
                let w = if transient { gaussian_window(t, center, width) } else { Array1::from_vec(vec![1.0_f64; n]) };

                let mut gamma: f64 = Normal::new(1.6_f64, 0.3_f64).unwrap().sample(rng);
                if gamma < 0.2_f64 { gamma = 0.2_f64; }
                for i in 0..n {
                    curve.y[(i, a)] = (1.0_f64 + (gamma - 1.0_f64) * w[i]) * curve.y[(i, a)];
                }
            }
        }

        AnomType::Shape => {
            for a in 0..m {
                let center: f64 = Uniform::new(0.2_f64, 0.8_f64).unwrap().sample(rng);
                let width:  f64 = Uniform::new(0.02_f64, 0.08_f64).unwrap().sample(rng);
                let mut amp: f64 = Normal::new(0.7_f64, 0.3_f64).unwrap().sample(rng);
                if amp < 0.0 { amp = -amp; }
                let w = gaussian_window(t, center, width);
                let freq: f64 = Uniform::new(8.0_f64, 16.0_f64).unwrap().sample(rng);
                for i in 0..n {
                    let bump = amp * w[i] * (2.0_f64 * std::f64::consts::PI * freq * t[i]).sin();
                    curve.y[(i, a)] += bump;
                }
            }
        }

        AnomType::Trend => {
            // For trend, we keep the same across channels (trend is global)
            for a in 0..m {
                let slope: f64 = Normal::new(0.8_f64, 0.4_f64).unwrap().sample(rng);
                let bias:  f64 = Normal::new(0.0_f64, 0.2_f64).unwrap().sample(rng);
                for i in 0..n {
                    curve.y[(i, a)] += bias + slope * (t[i] - t[0]);
                }
            }
        }

        AnomType::Phase => {
            for a in 0..m {
                let delta: f64 = Uniform::new(-0.06_f64, 0.06_f64).unwrap().sample(rng);
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
            for a in 0..m {
                let rho: f64 = Uniform::new(0.3_f64, 0.7_f64).unwrap().sample(rng);
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
            for a in 0..m {
                let center: f64 = Uniform::new(0.2_f64, 0.85_f64).unwrap().sample(rng);
                let width:  f64 = Uniform::new(0.02_f64, 0.06_f64).unwrap().sample(rng);
                let w = gaussian_window(t, center, width);
                let sigma: f64 = Uniform::new(1.0_f64, 2.0_f64).unwrap().sample(rng);
                for i in 0..n {
                    let z: f64 = Normal::new(0.0_f64, sigma).unwrap().sample(rng);
                    curve.y[(i, a)] += w[i] * z;
                }
            }
        }
    }
}

// Create single anomaly dataset with different locations per channel
pub fn make_single_anomaly_dataset_different_locations(
    rng: &mut StdRng,
    n_curves: usize,
    which: AnomType,
) -> LabeledDatasetM {
    assert!(n_curves >= 2, "Need at least 2 curves to have anomalies + normals");

    let n = 64;               // keep consistent with the multitype set
    let m_out = 3;
    let p = 16;
    let t = crate::utils::linspace(0.0, 1.0, n);
    let x = crate::wavelets::make_wavelet_design(n, p, crate::wavelets::DesignKind::Haar);

    // "normal" ICM (same spirit/scale as your multitype dataset)
    let fam = KernelFamily::Matern32;
    let kx_norm = build_kx(&t, fam, &KernelHyper { 
        ell: 0.22, 
        alpha: 1.0, 
        period: 1.0, 
        gamma: 2.0 
    });
    let mut b_norm = Array2::<f64>::zeros((m_out, m_out));
    for i in 0..m_out { b_norm[(i,i)] = 1.0; }
    b_norm[(0,1)] = 0.75; b_norm[(1,0)] = 0.75;
    b_norm[(1,2)] = 0.5;  b_norm[(2,1)] = 0.5;
    let eta_norm = ndarray::arr1(&[1e-3, 1.5e-3, 1e-3]);

    // 5% contamination rate
    let contam = 0.05;
    let n_anom = ((contam * n_curves as f64).round() as usize).max(1);
    let n_norm = n_curves - n_anom;

    // draw normal curves
    let mut curves = simulate_icm_curves(rng, n, m_out, &kx_norm, &b_norm, &eta_norm, n_norm);

    // create anomaly curves from normal model then apply anomaly with different locations
    let mut anom_curves = simulate_icm_curves(rng, n, m_out, &kx_norm, &b_norm, &eta_norm, n_anom);
    for curve in anom_curves.iter_mut() {
        apply_anomaly_to_curve_different_locations(rng, curve, &t, &which);
    }

    // combine normal curves with anomaly curves
    curves.extend(anom_curves);

    // labels: n_norm normals + n_anom anomalies
    let mut labels = vec![AnomType::Normal; n_norm];
    labels.extend(vec![which.clone(); n_anom]);

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

// Create all single anomaly datasets with different locations per channel
pub fn make_all_single_anomaly_datasets_different_locations(
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
        let ds = make_single_anomaly_dataset_different_locations(rng, n_curves, k.clone());
        (format!("{}_diff_locations", crate::anomaly::anom_slug(k)), ds)
    }).collect()
}
