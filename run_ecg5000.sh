#!/bin/bash
#SBATCH --job-name=ecg5000
#SBATCH --account=def-dsteph
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# 0) Clean environment, load Rust and MKL (module names can vary by site)
module --force purge
# If your site uses a StdEnv, uncomment the next line and pick the right year:
# module load StdEnv/2023

module load StdEnv/2023

module load rust/1.85.0

# Prefer oneAPI MKL; fall back to older IMKL name if that's what your cluster has
module load intel-oneapi-mkl || module load imkl || true

# 1) Configure threading for real dataset analysis
# Allow multi-threaded operations since we're processing one dataset at a time
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export BLIS_NUM_THREADS=12
export VECLIB_MAXIMUM_THREADS=12

# 2) Help pkg-config find MKL when using ndarray-linalg's intel-mkl backend
if [[ -n "${MKLROOT:-}" ]]; then
  export PKG_CONFIG_PATH="${MKLROOT}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

# 3) Build in fast node-local storage; run from submit dir
cd "${SLURM_SUBMIT_DIR}"
export CARGO_TARGET_DIR="${SLURM_TMPDIR}/target"

# (Rust 2024 edition needs rustc >= 1.85)
rustc --version
cargo --version

# 4) Compile & run ECG5000 dataset
echo "Running ECG5000 dataset analysis..."
cargo run --release -- --mode data --dataset ecg5000

echo "ECG5000 analysis completed!"
