use plotters::prelude::*;
use std::fs::create_dir_all;
use ndarray::Array1;

/// MCMC diagnostics and monitoring
pub struct MCMCDiagnostics {
    pub iterations: Vec<usize>,
    pub active_clusters: Vec<usize>,
    pub occupied_clusters: Vec<usize>,
    pub kept_samples: Vec<usize>,
    pub alpha_values: Vec<f64>,
    pub log_likelihood: Vec<f64>,
    pub acceptance_rates: Vec<f64>,
}

impl MCMCDiagnostics {
    pub fn new() -> Self {
        Self {
            iterations: Vec::new(),
            active_clusters: Vec::new(),
            occupied_clusters: Vec::new(),
            kept_samples: Vec::new(),
            alpha_values: Vec::new(),
            log_likelihood: Vec::new(),
            acceptance_rates: Vec::new(),
        }
    }

    pub fn record_iteration(&mut self, iter: usize, active: usize, occupied: usize, 
                           kept: usize, alpha: f64, log_lik: f64, accept_rate: f64) {
        self.iterations.push(iter);
        self.active_clusters.push(active);
        self.occupied_clusters.push(occupied);
        self.kept_samples.push(kept);
        self.alpha_values.push(alpha);
        self.log_likelihood.push(log_lik);
        self.acceptance_rates.push(accept_rate);
    }
}

/// Plot MCMC trace plots
pub fn plot_mcmc_traces(diagnostics: &MCMCDiagnostics, out_path: &str) {
    if let Some(parent) = std::path::Path::new(out_path).parent() {
        let _ = create_dir_all(parent);
    }

    let root = BitMapBackend::new(out_path, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    
    // Split into 2x3 grid
    let areas = root.split_evenly((3, 2));

    // 1. Active clusters over time
    let mut chart1 = ChartBuilder::on(&areas[0])
        .margin(15)
        .set_left_and_bottom_label_area_size(40)
        .caption("Active Clusters", ("sans-serif", 16))
        .build_cartesian_2d(
            *diagnostics.iterations.first().unwrap() as f64..*diagnostics.iterations.last().unwrap() as f64,
            0.0..(*diagnostics.active_clusters.iter().max().unwrap() + 2) as f64
        )
        .unwrap();

    chart1.configure_mesh()
        .x_desc("Iteration")
        .y_desc("Active Clusters")
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    let points1: Vec<(f64, f64)> = diagnostics.iterations.iter()
        .zip(diagnostics.active_clusters.iter())
        .map(|(&x, &y)| (x as f64, y as f64))
        .collect();
    chart1.draw_series(LineSeries::new(points1, RED.stroke_width(2))).unwrap();

    // 2. Occupied clusters over time
    let mut chart2 = ChartBuilder::on(&areas[1])
        .margin(15)
        .set_left_and_bottom_label_area_size(40)
        .caption("Occupied Clusters", ("sans-serif", 16))
        .build_cartesian_2d(
            *diagnostics.iterations.first().unwrap() as f64..*diagnostics.iterations.last().unwrap() as f64,
            0.0..(*diagnostics.occupied_clusters.iter().max().unwrap() + 2) as f64
        )
        .unwrap();

    chart2.configure_mesh()
        .x_desc("Iteration")
        .y_desc("Occupied Clusters")
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    let points2: Vec<(f64, f64)> = diagnostics.iterations.iter()
        .zip(diagnostics.occupied_clusters.iter())
        .map(|(&x, &y)| (x as f64, y as f64))
        .collect();
    chart2.draw_series(LineSeries::new(points2, BLUE.stroke_width(2))).unwrap();

    // 3. Kept samples over time
    let mut chart3 = ChartBuilder::on(&areas[2])
        .margin(15)
        .set_left_and_bottom_label_area_size(40)
        .caption("Kept Samples", ("sans-serif", 16))
        .build_cartesian_2d(
            *diagnostics.iterations.first().unwrap() as f64..*diagnostics.iterations.last().unwrap() as f64,
            0.0..(*diagnostics.kept_samples.iter().max().unwrap() + 10) as f64
        )
        .unwrap();

    chart3.configure_mesh()
        .x_desc("Iteration")
        .y_desc("Kept Samples")
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    let points3: Vec<(f64, f64)> = diagnostics.iterations.iter()
        .zip(diagnostics.kept_samples.iter())
        .map(|(&x, &y)| (x as f64, y as f64))
        .collect();
    chart3.draw_series(LineSeries::new(points3, GREEN.stroke_width(2))).unwrap();

    // 4. Alpha values over time
    let mut chart4 = ChartBuilder::on(&areas[3])
        .margin(15)
        .set_left_and_bottom_label_area_size(40)
        .caption("DP Concentration Parameter (α)", ("sans-serif", 16))
        .build_cartesian_2d(
            *diagnostics.iterations.first().unwrap() as f64..*diagnostics.iterations.last().unwrap() as f64,
            diagnostics.alpha_values.iter().cloned().fold(f64::INFINITY, f64::min) * 0.9..
            diagnostics.alpha_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1.1
        )
        .unwrap();

    chart4.configure_mesh()
        .x_desc("Iteration")
        .y_desc("α")
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    let points4: Vec<(f64, f64)> = diagnostics.iterations.iter()
        .zip(diagnostics.alpha_values.iter())
        .map(|(&x, &y)| (x as f64, y))
        .collect();
    chart4.draw_series(LineSeries::new(points4, MAGENTA.stroke_width(2))).unwrap();

    // 5. Log likelihood over time
    let mut chart5 = ChartBuilder::on(&areas[4])
        .margin(15)
        .set_left_and_bottom_label_area_size(40)
        .caption("Log Likelihood", ("sans-serif", 16))
        .build_cartesian_2d(
            *diagnostics.iterations.first().unwrap() as f64..*diagnostics.iterations.last().unwrap() as f64,
            diagnostics.log_likelihood.iter().cloned().fold(f64::INFINITY, f64::min) * 1.1..
            diagnostics.log_likelihood.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 0.9
        )
        .unwrap();

    chart5.configure_mesh()
        .x_desc("Iteration")
        .y_desc("Log Likelihood")
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    let points5: Vec<(f64, f64)> = diagnostics.iterations.iter()
        .zip(diagnostics.log_likelihood.iter())
        .map(|(&x, &y)| (x as f64, y))
        .collect();
    chart5.draw_series(LineSeries::new(points5, CYAN.stroke_width(2))).unwrap();

    // 6. Acceptance rates over time
    let mut chart6 = ChartBuilder::on(&areas[5])
        .margin(15)
        .set_left_and_bottom_label_area_size(40)
        .caption("MH Acceptance Rate", ("sans-serif", 16))
        .build_cartesian_2d(
            *diagnostics.iterations.first().unwrap() as f64..*diagnostics.iterations.last().unwrap() as f64,
            0.0..1.0
        )
        .unwrap();

    chart6.configure_mesh()
        .x_desc("Iteration")
        .y_desc("Acceptance Rate")
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    let points6: Vec<(f64, f64)> = diagnostics.iterations.iter()
        .zip(diagnostics.acceptance_rates.iter())
        .map(|(&x, &y)| (x as f64, y))
        .collect();
    chart6.draw_series(LineSeries::new(points6, RGBColor(255, 165, 0).stroke_width(2))).unwrap();

    // Add target acceptance rate line
    chart6.draw_series(LineSeries::new(
        vec![(*diagnostics.iterations.first().unwrap() as f64, 0.3), 
             (*diagnostics.iterations.last().unwrap() as f64, 0.3)],
        RED.stroke_width(1)
    )).unwrap();

    let _ = root.titled("MCMC Diagnostics", ("sans-serif", 20));
}

/// Plot AUC curves for anomaly detection performance
pub fn plot_auc_curves(
    true_labels: &[bool], 
    anomaly_scores: &[f64], 
    out_path: &str
) {
    if let Some(parent) = std::path::Path::new(out_path).parent() {
        let _ = create_dir_all(parent);
    }

    let root = BitMapBackend::new(out_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    
    // Split into 2x1 grid
    let areas = root.split_evenly((1, 2));

    // 1. ROC Curve
    let mut chart1 = ChartBuilder::on(&areas[0])
        .margin(15)
        .set_left_and_bottom_label_area_size(40)
        .caption("ROC Curve", ("sans-serif", 16))
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)
        .unwrap();

    chart1.configure_mesh()
        .x_desc("False Positive Rate")
        .y_desc("True Positive Rate")
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    // Calculate ROC curve
    let mut roc_points = Vec::new();
    let mut thresholds = anomaly_scores.to_vec();
    thresholds.sort_by(|a, b| b.partial_cmp(a).unwrap());
    thresholds.dedup();

    for &threshold in &thresholds {
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut false_negatives = 0;

        for (i, &score) in anomaly_scores.iter().enumerate() {
            let predicted_anomaly = score > threshold;
            let is_anomaly = !true_labels[i]; // true_labels[i] = true means normal

            if predicted_anomaly && is_anomaly {
                tp += 1;
            } else if predicted_anomaly && !is_anomaly {
                fp += 1;
            } else if !predicted_anomaly && !is_anomaly {
                tn += 1;
            } else {
                false_negatives += 1;
            }
        }

        let tpr = if tp + false_negatives > 0 { tp as f64 / (tp + false_negatives) as f64 } else { 0.0 };
        let fpr = if fp + tn > 0 { fp as f64 / (fp + tn) as f64 } else { 0.0 };
        roc_points.push((fpr, tpr));
    }

    // Add diagonal line for random classifier
    chart1.draw_series(LineSeries::new(
        vec![(0.0, 0.0), (1.0, 1.0)],
        RGBColor(200, 200, 200).stroke_width(1)
    )).unwrap();

    chart1.draw_series(LineSeries::new(roc_points, BLUE.stroke_width(2))).unwrap();

    // 2. Precision-Recall Curve
    let mut chart2 = ChartBuilder::on(&areas[1])
        .margin(15)
        .set_left_and_bottom_label_area_size(40)
        .caption("Precision-Recall Curve", ("sans-serif", 16))
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)
        .unwrap();

    chart2.configure_mesh()
        .x_desc("Recall")
        .y_desc("Precision")
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    // Calculate PR curve
    let mut pr_points = Vec::new();
    for &threshold in &thresholds {
        let mut tp = 0;
        let mut fp = 0;
        let mut false_negatives = 0;

        for (i, &score) in anomaly_scores.iter().enumerate() {
            let predicted_anomaly = score > threshold;
            let is_anomaly = !true_labels[i];

            if predicted_anomaly && is_anomaly {
                tp += 1;
            } else if predicted_anomaly && !is_anomaly {
                fp += 1;
            } else if !predicted_anomaly && is_anomaly {
                false_negatives += 1;
            }
        }

        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + false_negatives > 0 { tp as f64 / (tp + false_negatives) as f64 } else { 0.0 };
        pr_points.push((recall, precision));
    }

    chart2.draw_series(LineSeries::new(pr_points, RED.stroke_width(2))).unwrap();

    let _ = root.titled("Anomaly Detection Performance", ("sans-serif", 20));
}

/// Plot function reconstruction with wavelet shrinkage
pub fn plot_function_reconstruction(
    original_curves: &[Array1<f64>], 
    reconstructed_curves: &[Array1<f64>],
    _wavelet_coeffs: &[Array1<f64>],
    out_path: &str
) {
    plot_function_reconstruction_with_title(original_curves, reconstructed_curves, _wavelet_coeffs, out_path, "Function Reconstruction");
}

pub fn plot_function_reconstruction_with_title(
    original_curves: &[Array1<f64>], 
    reconstructed_curves: &[Array1<f64>],
    _wavelet_coeffs: &[Array1<f64>],
    out_path: &str,
    title: &str
) {
    if let Some(parent) = std::path::Path::new(out_path).parent() {
        let _ = create_dir_all(parent);
    }

    let root = BitMapBackend::new(out_path, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    
    // Split into 2x3 grid
    let areas = root.split_evenly((3, 2));

    // Sample a few curves for visualization
    let n_curves_to_show = original_curves.len().min(6);
    let curve_indices: Vec<usize> = (0..n_curves_to_show).collect();

    for (plot_idx, &curve_idx) in curve_indices.iter().enumerate() {
        let mut chart = ChartBuilder::on(&areas[plot_idx])
            .margin(15)
            .set_left_and_bottom_label_area_size(40)
            .caption(
                format!("{} - Curve {}", title, curve_idx),
                ("sans-serif", 14)
            )
            .build_cartesian_2d(0.0..1.0, -3.0..3.0) // Adjust ranges as needed
            .unwrap();

        chart.configure_mesh()
            .x_desc("Time")
            .y_desc("Value")
            .label_style(("sans-serif", 10))
            .draw()
            .unwrap();

        // Plot original curve (solid blue line)
        let orig_points: Vec<(f64, f64)> = (0..original_curves[curve_idx].len())
            .map(|j| (j as f64 / (original_curves[curve_idx].len() - 1) as f64, original_curves[curve_idx][j]))
            .collect();
        chart.draw_series(LineSeries::new(orig_points, BLUE.stroke_width(3))).unwrap();

        // Plot reconstructed curve (dashed red line)
        let recon_points: Vec<(f64, f64)> = (0..reconstructed_curves[curve_idx].len())
            .map(|j| (j as f64 / (reconstructed_curves[curve_idx].len() - 1) as f64, reconstructed_curves[curve_idx][j]))
            .collect();
        chart.draw_series(LineSeries::new(recon_points, RED.stroke_width(2))).unwrap();

        // Add text labels for legend
        chart.draw_series([
            Text::new("Original", (0.05, 0.95), ("sans-serif", 12).into_font().color(&BLUE)),
            Text::new("Reconstructed", (0.05, 0.90), ("sans-serif", 12).into_font().color(&RED)),
        ]).unwrap();
    }

    let _ = root.titled("Function Reconstruction with Wavelet Shrinkage", ("sans-serif", 20));
}

/// Plot detailed wavelet reconstruction analysis
pub fn plot_wavelet_reconstruction_analysis(
    original_curves: &[Array1<f64>], 
    reconstructed_curves: &[Array1<f64>],
    wavelet_coeffs: &[Array1<f64>],
    _shrinkage_factors: &[f64],
    out_path: &str
) {
    if let Some(parent) = std::path::Path::new(out_path).parent() {
        let _ = create_dir_all(parent);
    }

    let root = BitMapBackend::new(out_path, (1800, 1400)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    
    // Split into 3x3 grid
    let areas = root.split_evenly((3, 3));

    // Sample a few curves for detailed analysis
    let n_curves_to_show = original_curves.len().min(3);
    let curve_indices: Vec<usize> = (0..n_curves_to_show).collect();

    for (plot_idx, &curve_idx) in curve_indices.iter().enumerate() {
        let start_area = plot_idx * 3;
        
        // 1. Original vs Reconstructed
        let mut chart1 = ChartBuilder::on(&areas[start_area])
            .margin(15)
            .set_left_and_bottom_label_area_size(40)
            .caption(
                format!("Curve {}: Original vs Reconstructed", curve_idx),
                ("sans-serif", 12)
            )
            .build_cartesian_2d(0.0..1.0, -3.0..3.0)
            .unwrap();

        chart1.configure_mesh()
            .x_desc("Time")
            .y_desc("Value")
            .label_style(("sans-serif", 10))
            .draw()
            .unwrap();

        let orig_points: Vec<(f64, f64)> = (0..original_curves[curve_idx].len())
            .map(|j| (j as f64 / (original_curves[curve_idx].len() - 1) as f64, original_curves[curve_idx][j]))
            .collect();
        chart1.draw_series(LineSeries::new(orig_points, BLUE.stroke_width(3))).unwrap();

        let recon_points: Vec<(f64, f64)> = (0..reconstructed_curves[curve_idx].len())
            .map(|j| (j as f64 / (reconstructed_curves[curve_idx].len() - 1) as f64, reconstructed_curves[curve_idx][j]))
            .collect();
        chart1.draw_series(LineSeries::new(recon_points, RED.stroke_width(2))).unwrap();

        // Add legend
        chart1.draw_series([
            Text::new("Original", (0.05, 0.95), ("sans-serif", 10).into_font().color(&BLUE)),
            Text::new("Reconstructed", (0.05, 0.90), ("sans-serif", 10).into_font().color(&RED)),
        ]).unwrap();

        // 2. Wavelet Coefficients
        let mut chart2 = ChartBuilder::on(&areas[start_area + 1])
            .margin(15)
            .set_left_and_bottom_label_area_size(40)
            .caption(
                format!("Curve {}: Wavelet Coefficients", curve_idx),
                ("sans-serif", 12)
            )
            .build_cartesian_2d(0.0..1.0, -2.0..2.0)
            .unwrap();

        chart2.configure_mesh()
            .x_desc("Coefficient Index")
            .y_desc("Coefficient Value")
            .label_style(("sans-serif", 10))
            .draw()
            .unwrap();

        let coeff_points: Vec<(f64, f64)> = (0..wavelet_coeffs[curve_idx].len())
            .map(|j| (j as f64 / (wavelet_coeffs[curve_idx].len() - 1) as f64, wavelet_coeffs[curve_idx][j]))
            .collect();
        chart2.draw_series(LineSeries::new(coeff_points, GREEN.stroke_width(2))).unwrap();

        // 3. Reconstruction Error
        let mut chart3 = ChartBuilder::on(&areas[start_area + 2])
            .margin(15)
            .set_left_and_bottom_label_area_size(40)
            .caption(
                format!("Curve {}: Reconstruction Error", curve_idx),
                ("sans-serif", 12)
            )
            .build_cartesian_2d(0.0..1.0, -1.0..1.0)
            .unwrap();

        chart3.configure_mesh()
            .x_desc("Time")
            .y_desc("Error")
            .label_style(("sans-serif", 10))
            .draw()
            .unwrap();

        let error_points: Vec<(f64, f64)> = (0..original_curves[curve_idx].len())
            .map(|j| {
                let t = j as f64 / (original_curves[curve_idx].len() - 1) as f64;
                let error = original_curves[curve_idx][j] - reconstructed_curves[curve_idx][j];
                (t, error)
            })
            .collect();
        chart3.draw_series(LineSeries::new(error_points, MAGENTA.stroke_width(2))).unwrap();
    }

    let _ = root.titled("Detailed Wavelet Reconstruction Analysis", ("sans-serif", 20));
}

/// Plot shrinkage factor analysis
pub fn plot_shrinkage_analysis(
    shrinkage_factors: &[f64],
    coefficient_magnitudes: &[f64],
    out_path: &str
) {
    if let Some(parent) = std::path::Path::new(out_path).parent() {
        let _ = create_dir_all(parent);
    }

    let root = BitMapBackend::new(out_path, (1600, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    
    // Split into 2x1 grid
    let areas = root.split_evenly((1, 2));

    // 1. Shrinkage factors distribution
    let mut chart1 = ChartBuilder::on(&areas[0])
        .margin(15)
        .set_left_and_bottom_label_area_size(40)
        .caption("Shrinkage Factors Distribution", ("sans-serif", 16))
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)
        .unwrap();

    chart1.configure_mesh()
        .x_desc("Shrinkage Factor")
        .y_desc("Density")
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    // Create histogram of shrinkage factors
    let mut bins = vec![0; 20];
    for &factor in shrinkage_factors {
        let bin_idx = ((factor * 20.0).min(19.0) as usize).min(19);
        bins[bin_idx] += 1;
    }

    let max_count = bins.iter().max().unwrap_or(&1);
    let hist_points: Vec<(f64, f64)> = bins.iter().enumerate()
        .map(|(i, &count)| (i as f64 / 20.0, count as f64 / *max_count as f64))
        .collect();

    chart1.draw_series(LineSeries::new(hist_points, BLUE.stroke_width(2))).unwrap();

    // 2. Shrinkage vs Coefficient Magnitude
    let mut chart2 = ChartBuilder::on(&areas[1])
        .margin(15)
        .set_left_and_bottom_label_area_size(40)
        .caption("Shrinkage vs Coefficient Magnitude", ("sans-serif", 16))
        .build_cartesian_2d(0.0..2.0, 0.0..1.0)
        .unwrap();

    chart2.configure_mesh()
        .x_desc("Coefficient Magnitude")
        .y_desc("Shrinkage Factor")
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    let scatter_points: Vec<(f64, f64)> = shrinkage_factors.iter()
        .zip(coefficient_magnitudes.iter())
        .map(|(&shrink, &mag)| (mag, shrink))
        .collect();

    chart2.draw_series(LineSeries::new(scatter_points, RED.stroke_width(1))).unwrap();

    let _ = root.titled("Shrinkage Analysis", ("sans-serif", 20));
}