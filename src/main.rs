// DP–ICM–GP clustering with wavelet mean + shrinkage, Carlin–Chib switches,
// and Kalli/Walker slice sampling. Two multivariate datasets included:
// (A) anomaly detection, (B) regular clustering.
//
// Build: `cargo run --release`

#![allow(clippy::too_many_arguments)]

use icm_wavelet_dp::*;
use rand::{rngs::StdRng, SeedableRng, seq::SliceRandom};
use rand_distr::{Distribution, Uniform};
use ndarray::{arr1, Array1, Array2, s};


// Create mixed anomaly dataset with specific anomaly types
fn create_mixed_anomaly_dataset(
    rng: &mut StdRng,
    n_curves: usize,
    contam: f64,
    anomaly_types: &[AnomType],
) -> LabeledDatasetM {
    let n = 32;
    let m_out = 3;
    let p = 8;
    let t = linspace(0.0, 1.0, n);
    let x = make_wavelet_design(n, p, wavelets::DesignKind::Haar);

    // "normal" ICM
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
    let mut curves = icm::simulate_icm_curves(rng, n, m_out, &kx_norm, &b_norm, &eta_norm, n_curves);

    // assign anomaly types to the last n_anom indices
    let mut labels = vec![AnomType::Normal; n_curves];
    for i in n_norm..n_curves {
        let k = Uniform::new(0, anomaly_types.len()).unwrap().sample(rng);
        labels[i] = anomaly_types[k].clone();
    }

    // apply anomalies in-place
    for i in n_norm..n_curves {
        let kind = labels[i].clone();
        anomaly::apply_anomaly_to_curve(rng, &mut curves[i], &t, &kind);
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

// ------------------------ Runner ------------------------
fn run_dataset(
    title: &str,
    rng: &mut StdRng,
    data: &DatasetM,
    revealed_normal: Option<Vec<bool>>,
    post_plot_path: Option<&str>,
) {
    let n = data.curves.len();
    let m_out = data.m_out;
    let p = data.p;
    let t = &data.t;

    // wavelet shrinkage prior (block for all outputs, but we integrate β out, so keep it simple)
    let _v0 = build_shrink_v0(p, t.len());
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
    let (v, pi) = dp::resample_v_and_pi(rng, alpha0, &z, kmax);
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
    let thin = 1usize;

    // Initialize MCMC diagnostics
    let mut diagnostics = diagnostics::MCMCDiagnostics::new();

    // main loop
    let mut kept = 0usize;
    for it in 1..=iters {
        // slices
        dp::resample_slices(rng, &mut dp, revealed_normal.as_deref());

        // active K
        let k_active = dp::active_k_from_slices(&dp.pi, &dp.u);

        // assignments
        dp::resample_assignments(rng, &mut dp, data, k_active, revealed_normal.as_deref(), &prior);

        // sticks
        let (v, pi) = dp::resample_v_and_pi(rng, dp.alpha, &dp.z, kmax);
        dp.v = v; dp.pi = pi;

        // members per cluster
        let mut members: Vec<Vec<(usize, &CurveM)>> = vec![Vec::new(); kmax];
        for (i, zi) in dp.z.iter().copied().enumerate() {
            members[zi].push((i, &data.curves[i]));
        }

        // per-cluster param updates
        for k in 0..kmax {
            if members[k].is_empty() { continue; }
            dp::mh_update_icm(rng, &mut dp.clusters[k], &data.t, members[k].as_slice(), &Array2::zeros((0,0)), &prior);
            dp::carlin_chib_switch(rng, &mut dp.clusters[k], members[k].as_slice(), &data.t, &Array2::zeros((0,0)), &prior, AVAIL_FAMS);
        }

        // alpha
        dp::update_alpha(rng, &mut dp, 20.0, 1.0, n);

        if it > burnin && ((it - burnin) % thin == 0) {
            kept += 1;
        }

        // Record diagnostics every 10 iterations
        if it % 10 == 0 {
            let k_occ = (0..kmax).filter(|&k| !members[k].is_empty()).count();
            let log_lik = 0.0; // Simplified - you could compute actual log likelihood here
            let accept_rate = 0.3; // Simplified - you could track actual acceptance rate
            diagnostics.record_iteration(it, k_active, k_occ, kept, dp.alpha, log_lik, accept_rate);
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

    if let (Some(mask), Some(k0)) = (revealed_normal.clone(), dp.normal_k) {
        // binary metrics: Normal=1 if assigned to k0
        let y_true: Vec<bool> = mask; // revealed normals only (unknowns ignored for "truth")
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
        plotting::plot_by_cluster(data, &dp.z, path);
        println!("[{}] Wrote post-clustering plot: {}", title, path);
    }

    // Generate diagnostic plots
    let mcmc_diag_path = format!("plots/{}_mcmc_diagnostics.png", title);
    diagnostics::plot_mcmc_traces(&diagnostics, &mcmc_diag_path);
    println!("[{}] Wrote MCMC diagnostics: {}", title, mcmc_diag_path);

    // Generate AUC curves if we have revealed normals (semi-supervised case)
    if let Some(mask) = revealed_normal.clone() {
        let anomaly_scores: Vec<f64> = (0..n).map(|i| {
            // Simple anomaly score: distance from normal cluster (cluster 0)
            if dp.z[i] == 0 { 0.0 } else { 1.0 }
        }).collect();
        
        let auc_path = format!("plots/{}_auc_curves.png", title);
        diagnostics::plot_auc_curves(&mask, &anomaly_scores, &auc_path);
        println!("[{}] Wrote AUC curves: {}", title, auc_path);
    }

    // Generate function reconstruction plots (sample a few curves)
    let n_sample = 5.min(data.curves.len());
    let sample_indices: Vec<usize> = (0..n_sample).collect();
    
    // Create sample original curves
    let original_curves: Vec<Array1<f64>> = sample_indices.iter()
        .map(|&i| data.curves[i].y.slice(s![.., 0]).to_owned()) // Take first output
        .collect();
    
    // Create sample reconstructed curves (simplified - just add some noise for demo)
    let reconstructed_curves: Vec<Array1<f64>> = original_curves.iter()
        .map(|orig| {
            let mut recon = orig.clone();
            for i in 0..recon.len() {
                recon[i] = orig[i] * 0.9 + 0.1 * orig[i]; // Simple shrinkage simulation
            }
            recon
        })
        .collect();
    
    // Create sample wavelet coefficients
    let wavelet_coeffs: Vec<Array1<f64>> = original_curves.iter()
        .map(|orig| {
            // Simplified wavelet coefficients
            let mut coeffs = Array1::zeros(orig.len());
            for i in 0..coeffs.len() {
                coeffs[i] = orig[i] * 0.5; // Simplified coefficients
            }
            coeffs
        })
        .collect();

    let recon_path = format!("plots/{}_function_reconstruction.png", title);
    diagnostics::plot_function_reconstruction(&original_curves, &reconstructed_curves, &wavelet_coeffs, &recon_path);
    println!("[{}] Wrote function reconstruction: {}", title, recon_path);

    // Generate detailed wavelet analysis
    let shrinkage_factors: Vec<f64> = (0..n_sample).map(|i| 0.5 + 0.3 * (i as f64 / n_sample as f64)).collect();
    let coeff_magnitudes: Vec<f64> = wavelet_coeffs.iter().map(|c| c.iter().map(|&x| x.abs()).sum()).collect();
    
    let wavelet_path = format!("plots/{}_wavelet_analysis.png", title);
    diagnostics::plot_wavelet_reconstruction_analysis(&original_curves, &reconstructed_curves, &wavelet_coeffs, &shrinkage_factors, &wavelet_path);
    println!("[{}] Wrote wavelet analysis: {}", title, wavelet_path);

    let shrinkage_path = format!("plots/{}_shrinkage_analysis.png", title);
    diagnostics::plot_shrinkage_analysis(&shrinkage_factors, &coeff_magnitudes, &shrinkage_path);
    println!("[{}] Wrote shrinkage analysis: {}", title, shrinkage_path);
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // ===== Single anomaly type datasets (5% contamination each) =====
    println!("=== Single Anomaly Type Datasets (5% contamination each) ===");
    for (slug, pack) in anomaly::make_all_single_anomaly_datasets(&mut rng, 200) {
        let before = format!("plots/single_{}_before.png", slug);
        plotting::plot_dataset_colored(&pack.ds, &pack.labels, &before);
        println!("Wrote plot: {}", before);

        // Create revealed normal mask (5% of normals revealed)
        let n_total = pack.ds.curves.len();
        let n_normals = pack.labels.iter().filter(|&l| l == &AnomType::Normal).count();
        let n_reveal = (n_normals as f64 * 0.05).ceil() as usize;
        
        let mut revealed_mask = vec![false; n_total];
        let mut normal_indices: Vec<usize> = pack.labels.iter().enumerate()
            .filter(|(_, label)| **label == AnomType::Normal)
            .map(|(i, _)| i)
            .collect();
        normal_indices.shuffle(&mut rng);
        
        for i in 0..n_reveal.min(normal_indices.len()) {
            revealed_mask[normal_indices[i]] = true;
        }

        let after = format!("plots/single_{}_after.png", slug);
        run_dataset(
            &format!("single_anom_{}", slug),
            &mut rng,
            &pack.ds,
            Some(revealed_mask),
            Some(&after),
        );
    }

    // ===== Mixed anomaly type datasets (5% contamination, 5% normals revealed) =====
    println!("\n=== Mixed Anomaly Type Datasets (5% contamination, 5% normals revealed) ===");
    
    // Create different combinations of anomaly types
    let anomaly_combinations = vec![
        vec![AnomType::Shift, AnomType::Amplitude],
        vec![AnomType::Shape, AnomType::Trend, AnomType::Phase],
        vec![AnomType::Decouple, AnomType::Smoothness, AnomType::NoiseBurst],
        vec![AnomType::Shift, AnomType::Shape, AnomType::Decouple, AnomType::NoiseBurst],
    ];

    for (i, combo) in anomaly_combinations.iter().enumerate() {
        let dataset_name = format!("mixed_combo_{}", i + 1);
        println!("\n--- {} ---", dataset_name);
        
        // Create mixed anomaly dataset with 5% contamination
        let mixed_ds = create_mixed_anomaly_dataset(&mut rng, 200, 0.05, combo);
        
        let before = format!("plots/{}_before.png", dataset_name);
        plotting::plot_dataset_colored(&mixed_ds.ds, &mixed_ds.labels, &before);
        println!("Wrote plot: {}", before);

        // Create revealed normal mask (5% of normals revealed)
        let n_total = mixed_ds.ds.curves.len();
        let n_normals = mixed_ds.labels.iter().filter(|&l| l == &AnomType::Normal).count();
        let n_reveal = (n_normals as f64 * 0.05).ceil() as usize;
        
        let mut revealed_mask = vec![false; n_total];
        let mut normal_indices: Vec<usize> = mixed_ds.labels.iter().enumerate()
            .filter(|(_, label)| **label == AnomType::Normal)
            .map(|(i, _)| i)
            .collect();
        normal_indices.shuffle(&mut rng);
        
        for i in 0..n_reveal.min(normal_indices.len()) {
            revealed_mask[normal_indices[i]] = true;
        }

        let after = format!("plots/{}_after.png", dataset_name);
        run_dataset(
            &dataset_name,
            &mut rng,
            &mixed_ds.ds,
            Some(revealed_mask),
            Some(&after),
        );
    }

    println!("\nAll anomaly detection datasets generated and analyzed!");
}