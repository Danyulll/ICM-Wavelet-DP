# ICM-Wavelet-DP: Semi-Supervised Anomaly Detection

A Rust implementation of Dirichlet Process (DP) clustering with Intrinsic Coregionalization Model (ICM) and Gaussian Process (GP) for semi-supervised anomaly detection in multivariate time series data.

## 🎯 Overview

This project implements a sophisticated Bayesian approach to anomaly detection that combines:

- **Dirichlet Process (DP)** clustering for automatic cluster discovery
- **Intrinsic Coregionalization Model (ICM)** for multivariate time series modeling
- **Gaussian Process (GP)** with multiple kernel families (SE, Matern32, Matern52)
- **Wavelet-based mean functions** with shrinkage priors
- **Semi-supervised learning** with revealed normal observations
- **Carlin-Chib model switching** for kernel selection
- **Kalli-Walker slice sampling** for efficient DP inference

## 🚀 Features

### Core Algorithm
- **Semi-supervised anomaly detection** with 5% contamination rate
- **Multiple anomaly types**: Shift, Amplitude, Shape, Trend, Phase, Decouple, Smoothness, NoiseBurst
- **Automatic cluster discovery** using Dirichlet Process
- **Multivariate time series modeling** with cross-output correlations
- **Wavelet shrinkage** for smooth mean functions
- **Model selection** via Carlin-Chib switching

### Visualization
- **Before/after plots** showing ground truth vs. clustering results
- **Color-coded anomaly types** for easy interpretation
- **Cluster assignment visualization** with distinct colors
- **Semi-supervised performance metrics** (F1 score, precision, recall)

## 📊 Datasets

The code generates two types of datasets:

### 1. Single Anomaly Type Datasets
- **8 datasets** (one per anomaly type)
- **200 curves per dataset** (190 normal + 10 anomalies)
- **5% contamination rate**
- **5% of normal observations revealed** (semi-supervised)

### 2. Mixed Anomaly Type Datasets
- **4 different combinations** of anomaly types
- **200 curves per dataset** (190 normal + 10 anomalies)
- **5% contamination rate**
- **5% of normal observations revealed** (semi-supervised)

## 🛠️ Installation

### Prerequisites
- Rust 1.70+ (2021 edition)
- Cargo package manager

### Build
```bash
# Clone the repository
git clone <repository-url>
cd ICM-Wavelet-DP

# Build in release mode (recommended for performance)
cargo build --release

# Run the analysis
cargo run --release
```

## ⚙️ Configuration

### MCMC Settings
```rust
let iters = 1200usize;        // Total MCMC iterations
let burnin = 600usize;        // Burn-in iterations (discarded)
let thin = 5usize;            // Thinning factor (keep every 5th sample)
```

### Data Dimensionality
```rust
let n = 64;               // Time points per curve
let m_out = 3;           // Number of outputs/channels
let p = 16;              // Wavelet basis functions
```

### DP Settings
```rust
let kmax = 8usize;           // Maximum number of clusters
let alpha0 = 8.0;            // DP concentration parameter
```

### Dataset Settings
```rust
// Single anomaly datasets
for (slug, pack) in anomaly::make_all_single_anomaly_datasets(&mut rng, 200) {
    // 200 curves per dataset
}

// Mixed anomaly datasets
let mixed_ds = create_mixed_anomaly_dataset(&mut rng, 200, 0.05, combo);
// 200 curves, 5% contamination
```

## 📁 Project Structure

```
src/
├── main.rs              # Main orchestration and dataset generation
├── lib.rs               # Library exports
├── data_structures.rs   # Core data types (AnomType, CurveM, DatasetM, etc.)
├── kernels.rs           # Kernel functions (SE, Matern32, Matern52)
├── icm.rs              # Intrinsic Coregionalization Model implementation
├── wavelets.rs         # Wavelet design and shrinkage functions
├── dp.rs               # Dirichlet Process clustering logic
├── anomaly.rs          # Anomaly detection and data generation
├── plotting.rs         # Visualization functions
└── utils.rs            # Utility functions and helpers

plots/                  # Generated visualization outputs
├── single_*_before.png # Single anomaly type datasets (before clustering)
├── single_*_after.png # Single anomaly type datasets (after clustering)
├── mixed_combo_*_before.png # Mixed anomaly datasets (before clustering)
└── mixed_combo_*_after.png # Mixed anomaly datasets (after clustering)
```

## 🎨 Anomaly Types

The system supports 8 different anomaly types:

| **Type** | **Description** | **Effect** |
|----------|-----------------|------------|
| **Shift** | Constant offset | `y ← y + c` |
| **Amplitude** | Amplitude scaling | `y ← γ ⊙ y` |
| **Shape** | High-frequency bump | `y ← y + bump(t)` |
| **Trend** | Linear drift | `y ← y + a*t + b` |
| **Phase** | Time lag/phase shift | `y(t) ← y(t - Δ)` |
| **Decouple** | Break correlations | `y ← y + ε` (independent) |
| **Smoothness** | Smoothness change | `y ← (1-ρ)y + ρ*filt(y)` |
| **NoiseBurst** | High-variance noise | `y ← y + σ*N(0,I)` |

## 📈 Output

### Console Output
```
=== Single Anomaly Type Datasets (5% contamination each) ===
Wrote plot: plots/single_shift_before.png
[single_anom_shift] it    1 | active  8 | occupied  8 | kept    0
[single_anom_shift] it  200 | active  3 | occupied  3 | kept    0
...
[single_anom_shift] Final cluster sizes (nonzero):
  k00: 190
  k01: 10
[single_anom_shift] Semi-supervised (revealed normals) F1 = 0.950 (tp 9, fp 1, fn 1)
```

### Generated Plots
- **Before plots**: Show ground truth with color-coded anomaly types
- **After plots**: Show clustering results with distinct cluster colors
- **Performance metrics**: F1 score, precision, recall for revealed normals

## 🔬 Algorithm Details

### 1. Data Generation
- Generate multivariate time series from ICM with cross-output correlations
- Apply anomaly transformations to create contaminated datasets
- Shuffle data while preserving label alignment

### 2. Semi-Supervised Learning
- Reveal 5% of normal observations to the algorithm
- Use revealed normals to guide cluster assignment
- Reserve cluster 0 for normal observations

### 3. DP-ICM Clustering
- Initialize DP with random cluster assignments
- Use slice sampling for efficient DP inference
- Update ICM parameters via Metropolis-Hastings
- Switch kernel families via Carlin-Chib method

### 4. Evaluation
- Calculate F1 score on revealed normal observations
- Generate before/after visualization plots
- Report final cluster sizes and assignments

## 🎛️ Customization

### Modify Dataset Size
```rust
// In main.rs, change the number of curves
for (slug, pack) in anomaly::make_all_single_anomaly_datasets(&mut rng, 500) {
    // 500 curves per dataset
}
```

### Adjust Contamination Rate
```rust
// In anomaly.rs, change contamination rate
let contam = 0.10;  // 10% contamination instead of 5%
```

### Modify MCMC Settings
```rust
// In main.rs, adjust MCMC parameters
let iters = 2000usize;    // More iterations
let burnin = 1000usize;   // Longer burn-in
let thin = 10usize;       // More aggressive thinning
```

### Change Data Dimensionality
```rust
// In anomaly.rs, modify data dimensions
let n = 128;              // More time points
let m_out = 5;            // More outputs
let p = 32;               # More wavelet functions
```

## 📚 Dependencies

- **ndarray**: N-dimensional arrays
- **ndarray-linalg**: Linear algebra operations
- **plotters**: Plotting and visualization
- **rand**: Random number generation
- **rand_distr**: Random distributions
- **statrs**: Statistical functions

## 🚀 Performance

- **Release build recommended** for optimal performance
- **MCMC sampling** typically takes 1-5 minutes per dataset
- **Memory usage** scales with dataset size and dimensionality
- **Parallel execution** not implemented (single-threaded)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{icm_wavelet_dp,
  title={ICM-Wavelet-DP: Semi-Supervised Anomaly Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ICM-Wavelet-DP}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Troubleshooting

### Common Issues

1. **Compilation errors**: Ensure you have Rust 1.70+ installed
2. **Plot generation fails**: Check that the `plots/` directory exists
3. **Memory issues**: Reduce dataset size or dimensionality
4. **Slow performance**: Use `cargo run --release` for optimized builds

### Getting Help

- Check the console output for error messages
- Verify all dependencies are installed
- Ensure sufficient disk space for plot generation
- Check that the `plots/` directory is writable

## 🔮 Future Enhancements

- [ ] Parallel MCMC sampling
- [ ] GPU acceleration for large datasets
- [ ] Interactive visualization
- [ ] Real-time anomaly detection
- [ ] Additional anomaly types
- [ ] Hyperparameter optimization
- [ ] Cross-validation framework
