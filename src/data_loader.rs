use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader};
use rand::prelude::*;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct RealDataset {
    pub curves: Vec<Array2<f64>>,  // Each curve is n_timepoints x m_channels
    pub labels: Vec<bool>,        // true = normal, false = anomaly
    pub time_points: Array1<f64>, // Time points for the curves
}

impl RealDataset {
    pub fn new(curves: Vec<Array2<f64>>, labels: Vec<bool>, time_points: Array1<f64>) -> Self {
        Self { curves, labels, time_points }
    }
    
    pub fn n_curves(&self) -> usize {
        self.curves.len()
    }
    
    pub fn n_timepoints(&self) -> usize {
        self.time_points.len()
    }
    
    pub fn n_channels(&self) -> usize {
        if self.curves.is_empty() { 0 } else { self.curves[0].shape()[1] }
    }
}

/// Load ECG5000 dataset and convert to bivariate functional data
pub fn load_ecg5000_dataset(
    train_path: &str,
    test_path: &str,
) -> Result<RealDataset, Box<dyn std::error::Error>> {
    println!("Loading ECG5000 dataset...");
    
    // Load training data
    let train_data = load_ecg5000_file(train_path)?;
    println!("Loaded {} training samples", train_data.len());
    
    // Load test data
    let test_data = load_ecg5000_file(test_path)?;
    println!("Loaded {} test samples", test_data.len());
    
    // Combine train and test
    let mut all_data = train_data;
    all_data.extend(test_data);
    
    println!("Total samples: {}", all_data.len());
    
    // Convert to bivariate functional data
    let (curves, labels, time_points) = convert_to_bivariate_functional_data(all_data)?;
    
    Ok(RealDataset::new(curves, labels, time_points))
}

/// Load a single ECG5000 file
fn load_ecg5000_file(file_path: &str) -> Result<Vec<(f64, Vec<f64>)>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        let values: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        
        if values.len() < 2 {
            continue; // Skip invalid lines
        }
        
        let label = values[0];
        let time_series = values[1..].to_vec();
        
        data.push((label, time_series));
    }
    
    Ok(data)
}

/// Convert univariate time series to bivariate functional data using derivative estimation
fn convert_to_bivariate_functional_data(
    data: Vec<(f64, Vec<f64>)>,
) -> Result<(Vec<Array2<f64>>, Vec<bool>, Array1<f64>), Box<dyn std::error::Error>> {
    if data.is_empty() {
        return Err("No data to process".into());
    }
    
    let original_length = data[0].1.len();
    let n_curves = data.len();
    
    // Interpolate to largest power of 2 for wavelet transform
    let n_timepoints = largest_power_of_two(original_length);
    println!("Converting {} curves of length {} to bivariate functional data (interpolated to {})", n_curves, original_length, n_timepoints);
    
    // Create time points (normalized to [0, 1])
    let time_points = Array1::linspace(0.0, 1.0, n_timepoints);
    
    let mut curves = Vec::new();
    let mut labels = Vec::new();
    
    for (label, time_series) in data {
        // Interpolate time series to power of 2
        let original_length = time_series.len();
        let original_times = Array1::linspace(0.0, 1.0, original_length);
        let target_times = Array1::linspace(0.0, 1.0, n_timepoints);
        
        let interpolated_series = interpolate_signal(&original_times, &time_series, &target_times);
        
        // Estimate first derivative using finite differences
        let derivative = estimate_derivative(&interpolated_series);
        
        // Create bivariate functional data: [original, derivative]
        let mut curve = Array2::<f64>::zeros((n_timepoints, 2));
        
        for i in 0..n_timepoints {
            curve[(i, 0)] = interpolated_series[i];  // Original signal (interpolated)
            curve[(i, 1)] = derivative[i];            // First derivative
        }
        
        curves.push(curve);
        
        // Convert label: 1.0 = normal, -1.0 = anomaly
        labels.push(label > 0.0);
    }
    
    println!("Converted to bivariate functional data with {} channels", 2);
    
    Ok((curves, labels, time_points))
}

/// Find the largest power of 2 less than or equal to n
fn largest_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut power = 1;
    while (power << 1) <= n {
        power <<= 1;
    }
    power
}

/// Estimate first derivative using finite differences
fn estimate_derivative(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let mut derivative = vec![0.0; n];
    
    if n < 2 {
        return derivative;
    }
    
    // Forward difference for first point
    derivative[0] = signal[1] - signal[0];
    
    // Central differences for interior points
    for i in 1..n-1 {
        derivative[i] = (signal[i+1] - signal[i-1]) / 2.0;
    }
    
    // Backward difference for last point
    derivative[n-1] = signal[n-1] - signal[n-2];
    
    derivative
}

/// Convert real dataset to the format expected by the ICM-Wavelet-DP algorithm
pub fn convert_to_icm_format(
    dataset: &RealDataset,
    _rng: &mut StdRng,
) -> (Vec<Array2<f64>>, Vec<bool>, Array1<f64>) {
    let n_curves = dataset.n_curves();
    let n_timepoints = dataset.n_timepoints();
    let n_channels = dataset.n_channels();
    
    println!("Converting real dataset to ICM format:");
    println!("  Curves: {}", n_curves);
    println!("  Time points: {}", n_timepoints);
    println!("  Channels: {}", n_channels);
    
    // Create time points
    let time_points = dataset.time_points.clone();
    
    // Convert curves to the expected format
    let mut formatted_curves = Vec::new();
    for curve in &dataset.curves {
        // Transpose to get (channels, timepoints) format
        let mut formatted_curve = Array2::<f64>::zeros((n_channels, n_timepoints));
        for i in 0..n_timepoints {
            for j in 0..n_channels {
                formatted_curve[(j, i)] = curve[(i, j)];
            }
        }
        formatted_curves.push(formatted_curve);
    }
    
    // Labels are already in the correct format
    let labels = dataset.labels.clone();
    
    (formatted_curves, labels, time_points)
}

/// Load 12-lead ECG dataset (St Petersburg INCART Database)
pub fn load_12lead_ecg_dataset(
    data_dir: &str,
    max_records: Option<usize>,
) -> Result<RealDataset, Box<dyn std::error::Error>> {
    println!("Loading 12-lead ECG dataset from: {}", data_dir);
    
    // Read RECORDS file to get list of record names
    let records_path = Path::new(data_dir).join("RECORDS");
    let records = load_records_file(&records_path)?;
    
    let records_to_process = if let Some(max) = max_records {
        records.into_iter().take(max).collect()
    } else {
        records
    };
    
    println!("Processing {} records", records_to_process.len());
    
    let mut all_curves = Vec::new();
    let mut all_labels = Vec::new();
    let mut time_points = None;
    
    for (i, record_name) in records_to_process.iter().enumerate() {
        if i % 10 == 0 {
            println!("Processing record {}/{}: {}", i + 1, records_to_process.len(), record_name);
        }
        
        match load_single_12lead_record(data_dir, record_name) {
            Ok((curves, labels, t_points)) => {
                if time_points.is_none() {
                    time_points = Some(t_points);
                }
                all_curves.extend(curves);
                all_labels.extend(labels);
            },
            Err(e) => {
                eprintln!("Warning: Failed to load record {}: {}", record_name, e);
                continue;
            }
        }
    }
    
    if all_curves.is_empty() {
        return Err("No valid records found".into());
    }
    
    let time_points = time_points.unwrap();
    println!("Successfully loaded 12-lead ECG dataset:");
    println!("  Records processed: {}", records_to_process.len());
    println!("  Total curves: {}", all_curves.len());
    println!("  Time points: {}", time_points.len());
    println!("  Channels: {}", if all_curves.is_empty() { 0 } else { all_curves[0].shape()[1] });
    
    Ok(RealDataset::new(all_curves, all_labels, time_points))
}

/// Load the RECORDS file to get list of record names
fn load_records_file(records_path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let file = File::open(records_path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        let record_name = line.trim().to_string();
        if !record_name.is_empty() {
            records.push(record_name);
        }
    }
    
    Ok(records)
}

/// Load a single 12-lead ECG record
fn load_single_12lead_record(
    data_dir: &str,
    record_name: &str,
) -> Result<(Vec<Array2<f64>>, Vec<bool>, Array1<f64>), Box<dyn std::error::Error>> {
    let base_path = Path::new(data_dir).join(record_name);
    
    // Load header file to get signal parameters
    let header_path = base_path.with_extension("hea");
    let (_fs_hz, n_sig, n_samp, gains, adc_zeros, _sig_names) = parse_header_file(&header_path)?;
    
    // Load signal data
    let dat_path = base_path.with_extension("dat");
    let raw_signals = load_signal_data(&dat_path, n_sig, n_samp)?;
    
    // Convert raw signals to physical units (mV)
    let physical_signals = convert_to_physical_units(&raw_signals, &gains, &adc_zeros);
    
    // Load annotations
    let atr_path = base_path.with_extension("atr");
    let annotations = load_annotations(&atr_path)?;
    
    // Interpolate to power of 2 for wavelet analysis, but limit to reasonable size
    let target_length = largest_power_of_two(n_samp).min(4096); // Max 4096 samples (~16 seconds at 257 Hz)
    println!("Interpolating signal from {} to {} samples", n_samp, target_length);
    
    // Create time points (normalized to [0, 1]) for target length
    let time_points = Array1::linspace(0.0, 1.0, target_length);
    
    // Create single curve with all 12 leads using interpolation
    let mut curve_data = Array2::<f64>::zeros((target_length, n_sig));
    
    // Original time points (normalized to [0, 1])
    let original_time_points = Array1::linspace(0.0, 1.0, n_samp);
    
    for j in 0..n_sig {
        let interpolated_signal = interpolate_signal(&original_time_points, &physical_signals[j], &time_points);
        for i in 0..target_length {
            curve_data[(i, j)] = interpolated_signal[i];
        }
    }
    
    // Determine if the entire record contains any anomalous beats
    let has_anomalies = annotations.iter().any(|(_, symbol)| symbol != "N");
    
    let curves = vec![curve_data];
    let labels = vec![!has_anomalies]; // true = normal, false = anomaly
    
    Ok((curves, labels, time_points))
}

/// Parse header file to extract signal parameters
fn parse_header_file(header_path: &Path) -> Result<(usize, usize, usize, Vec<f64>, Vec<i16>, Vec<String>), Box<dyn std::error::Error>> {
    let file = File::open(header_path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    // Parse global line: <rec> <nsig> <fs> <nsamp>
    let global_line = lines.next().ok_or("Empty header file")??;
    let global_parts: Vec<&str> = global_line.split_whitespace().collect();
    if global_parts.len() < 4 {
        return Err("Invalid global header line".into());
    }
    
    let n_sig: usize = global_parts[1].parse()?;
    let fs_hz: usize = global_parts[2].parse()?;
    let n_samp: usize = global_parts[3].parse()?;
    
    // Parse signal specification lines
    let mut gains = Vec::new();
    let mut adc_zeros = Vec::new();
    let mut sig_names = Vec::new();
    
    for _ in 0..n_sig {
        let line = lines.next().ok_or("Insufficient signal specification lines")??;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 9 {
            return Err("Invalid signal specification line".into());
        }
        
        let gain: f64 = parts[2].parse()?;
        let adc_zero: i16 = parts[4].parse()?;
        let sig_name = parts[8].to_string();
        
        gains.push(gain);
        adc_zeros.push(adc_zero);
        sig_names.push(sig_name);
    }
    
    Ok((fs_hz, n_sig, n_samp, gains, adc_zeros, sig_names))
}

/// Load raw signal data from .dat file
fn load_signal_data(dat_path: &Path, n_sig: usize, n_samp: usize) -> Result<Vec<Vec<i16>>, Box<dyn std::error::Error>> {
    let mut file = File::open(dat_path)?;
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer)?;
    
    // Convert bytes to i16 values (little-endian)
    let mut signals = vec![vec![0i16; n_samp]; n_sig];
    let mut byte_idx = 0;
    
    for sample_idx in 0..n_samp {
        for sig_idx in 0..n_sig {
            if byte_idx + 1 < buffer.len() {
                let value = i16::from_le_bytes([buffer[byte_idx], buffer[byte_idx + 1]]);
                signals[sig_idx][sample_idx] = value;
                byte_idx += 2;
            }
        }
    }
    
    Ok(signals)
}

/// Convert raw signals to physical units (mV)
fn convert_to_physical_units(
    raw_signals: &[Vec<i16>],
    gains: &[f64],
    adc_zeros: &[i16],
) -> Vec<Vec<f64>> {
    let mut physical_signals = Vec::new();
    
    for (sig_idx, raw_signal) in raw_signals.iter().enumerate() {
        let gain = gains[sig_idx];
        let adc_zero = adc_zeros[sig_idx];
        let mut physical_signal = Vec::new();
        
        for &raw_value in raw_signal {
            let physical_value = (raw_value - adc_zero) as f64 / gain;
            physical_signal.push(physical_value);
        }
        
        physical_signals.push(physical_signal);
    }
    
    physical_signals
}

/// Load annotations from .atr file (binary format)
fn load_annotations(atr_path: &Path) -> Result<Vec<(usize, String)>, Box<dyn std::error::Error>> {
    let mut file = File::open(atr_path)?;
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer)?;
    
    let mut annotations = Vec::new();
    let mut i = 0;
    
    // Parse binary annotation format
    while i + 4 < buffer.len() {
        // Read sample number (4 bytes, little-endian)
        let sample_bytes = [buffer[i], buffer[i+1], buffer[i+2], buffer[i+3]];
        let sample_idx = u32::from_le_bytes(sample_bytes) as usize;
        
        // Skip annotation type (1 byte)
        i += 5;
        
        // Read symbol (1 byte)
        if i < buffer.len() {
            let symbol_byte = buffer[i];
            let symbol = match symbol_byte {
                0 => "N".to_string(),      // Normal
                1 => "L".to_string(),      // Left bundle branch block
                2 => "R".to_string(),      // Right bundle branch block
                3 => "A".to_string(),      // Atrial premature
                4 => "a".to_string(),      // Aberrated atrial premature
                5 => "J".to_string(),     // Nodal (junctional) premature
                6 => "S".to_string(),      // Supraventricular premature
                7 => "V".to_string(),      // Premature ventricular contraction
                8 => "E".to_string(),      // Ventricular escape
                9 => "F".to_string(),      // Fusion of ventricular and normal
                10 => "Q".to_string(),     // Unclassifiable
                11 => "/".to_string(),     // Paced
                12 => "f".to_string(),     // Fusion of paced and normal
                13 => "x".to_string(),     // Non-conducted P-wave
                14 => "|".to_string(),     // Isolated QRS-like artifact
                _ => format!("{}", symbol_byte), // Unknown symbol
            };
            
            annotations.push((sample_idx, symbol));
        }
        
        i += 1;
    }
    
    Ok(annotations)
}

/// Interpolate a signal from original time points to new time points using linear interpolation
fn interpolate_signal(
    original_times: &Array1<f64>,
    signal: &[f64],
    new_times: &Array1<f64>,
) -> Vec<f64> {
    let mut interpolated = Vec::new();
    
    for &new_time in new_times.iter() {
        // Find the two nearest points for linear interpolation
        let mut left_idx = 0;
        let mut right_idx = original_times.len() - 1;
        
        // Find the interval containing new_time
        for i in 0..original_times.len() - 1 {
            if original_times[i] <= new_time && new_time <= original_times[i + 1] {
                left_idx = i;
                right_idx = i + 1;
                break;
            }
        }
        
        // Handle edge cases
        if new_time <= original_times[0] {
            interpolated.push(signal[0]);
        } else if new_time >= original_times[original_times.len() - 1] {
            interpolated.push(signal[signal.len() - 1]);
        } else {
            // Linear interpolation
            let t1 = original_times[left_idx];
            let t2 = original_times[right_idx];
            let y1 = signal[left_idx];
            let y2 = signal[right_idx];
            
            let alpha = (new_time - t1) / (t2 - t1);
            let interpolated_value = (1.0 - alpha) * y1 + alpha * y2;
            interpolated.push(interpolated_value);
        }
    }
    
    interpolated
}

