# AFPM Thomas Algorithm Analysis

**Quick analysis of Approximate Fixed-Point Multipliers (AFPM) applied to Thomas algorithm for tridiagonal linear systems.**

## Quick Start

```bash
# 1. Generate test matrices
python3 matrix_generator.py

# 2. Run analysis 
python3 enhanced_afpm_analysis.py

# 3. Create plots
python3 plot_afpm_results.py

# 4. View results
ls results/
```

## What it does

- **Tests 6 AFPM configurations** from exact to maximum approximation
- **Analyzes 2 matrix types**: Tridiagonal Toeplitz and FEM Poisson  
- **Scales from 4×4 to 65536×65536** matrices
- **Compares against double precision** ground truth
- **Shows theoretical error bounds** prominently in plots

## Key Files

- `config.yaml` - Configuration (matrix sizes, AFPM settings, etc.)
- `enhanced_afpm_analysis.py` - Analysis engine (generates data)
- `plot_afpm_results.py` - Visualization engine (creates plots)
- `results/afpm_analysis_logarithmic.png` - Main results plot

## Configuration

Edit `config.yaml` to customize:
- Matrix sizes and types
- Number of matrices per size  
- AFPM configurations
- Plot settings

## Output

- **Configuration table**: Shows 6 AFPM chromosome configurations
- **Log-scale plots**: Error vs matrix size and condition number
- **Theoretical bounds**: Float32 theoretical bound displayed prominently
- **Application thresholds**: Scientific, engineering, real-time, IoT error levels

## Requirements

```bash
pip install numpy matplotlib pandas scipy
```

**Time**: ~15-30 minutes total  
**Storage**: ~1GB for matrices

## Documentation

See `DOCUMENTATION.md` for comprehensive technical details.