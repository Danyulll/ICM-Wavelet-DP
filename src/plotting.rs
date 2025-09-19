use plotters::prelude::*;
use std::fs::create_dir_all;
use crate::data_structures::{DatasetM, AnomType};

pub fn plot_dataset_icm(ds: &DatasetM, labels: Option<&[bool]>, out_path: &str) {
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

pub fn plot_dataset_colored(ds: &DatasetM, labels: &[AnomType], out_path: &str) {
    plot_dataset_colored_with_title(ds, labels, out_path, "Dataset - Before Clustering");
}

pub fn plot_dataset_colored_with_title(ds: &DatasetM, labels: &[AnomType], out_path: &str, title: &str) {
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
                format!("{} • Output {}  |  {}", title, a,
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

pub fn plot_by_cluster(ds: &DatasetM, z: &[usize], out_path: &str) {
    plot_by_cluster_with_title(ds, z, out_path, "Dataset - After Clustering");
}

pub fn plot_by_cluster_with_title(ds: &DatasetM, z: &[usize], out_path: &str, title: &str) {
    if let Some(parent) = std::path::Path::new(out_path).parent() { let _ = create_dir_all(parent); }

    let n = ds.t.len();
    let m_out = ds.m_out;
    let kmax = 1 + z.iter().copied().max().unwrap_or(0);

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

    // counts and find main cluster (largest)
    let mut counts = vec![0usize; kmax];
    for &zi in z { counts[zi] += 1; }
    let main_cluster = (0..kmax).max_by_key(|&k| counts[k]).unwrap_or(0);

    // Define distinct colors for clusters
    let cluster_colors = vec![
        RGBColor(100, 100, 100),    // k0: light gray (main cluster - semi-transparent)
        RGBColor(255, 0, 0),        // k1: red
        RGBColor(0, 0, 255),         // k2: blue  
        RGBColor(0, 255, 0),        // k3: green
        RGBColor(255, 0, 255),      // k4: magenta
        RGBColor(0, 255, 255),      // k5: cyan
        RGBColor(255, 165, 0),      // k6: orange
        RGBColor(128, 0, 128),      // k7: purple
        RGBColor(255, 192, 203),    // k8: pink
    ];

    let root = BitMapBackend::new(out_path, (1200, 320 * m_out as u32)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let areas = root.split_evenly((m_out, 1));
    let x0 = ds.t[0]; let x1 = ds.t[n-1];

    for a in 0..m_out {
        let mut chart = ChartBuilder::on(&areas[a])
            .margin(15).set_left_and_bottom_label_area_size(40)
            .caption(
                format!("{} • Output {}  |  {}", title, a,
                    (0..kmax).map(|k| format!("k{}:{}",k,counts[k])).collect::<Vec<_>>().join("  ")),
                ("sans-serif", 18))
            .build_cartesian_2d(x0..x1, y_min[a]..y_max[a]).unwrap();

        chart.configure_mesh().x_desc("t").y_desc("y").label_style(("sans-serif", 12)).draw().unwrap();

        // First pass: draw main cluster (semi-transparent)
        for (i, c) in ds.curves.iter().enumerate() {
            if z[i] == main_cluster {
                let color = &cluster_colors[z[i] % cluster_colors.len()];
                let style = color.stroke_width(1); // thin line for main cluster
                let pts = (0..n).map(|ix| (ds.t[ix], c.y[(ix, a)]));
                chart.draw_series(LineSeries::new(pts, style)).unwrap();
            }
        }

        // Second pass: draw anomaly clusters (thick, opaque)
        for (i, c) in ds.curves.iter().enumerate() {
            if z[i] != main_cluster {
                let color = &cluster_colors[z[i] % cluster_colors.len()];
                let style = color.stroke_width(3); // thick line for anomalies
                let pts = (0..n).map(|ix| (ds.t[ix], c.y[(ix, a)]));
                chart.draw_series(LineSeries::new(pts, style)).unwrap();
            }
        }
    }
    let _ = root.titled("Curves colored by assigned cluster (anomalies on top)", ("sans-serif", 14));
}
