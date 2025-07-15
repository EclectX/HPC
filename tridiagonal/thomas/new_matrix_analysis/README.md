# AFMP Analysis Framework

Parametric analysis of Approximate Fixed-Point Multipliers (AFMP) on tridiagonal linear systems.

## Quick Start

```bash
# 1. Generate test matrices
python matrix_generators.py

# 2. Run AFMP analysis
python parametric_afmp_analysis.py

# 3. Create plots
python create_custom_plots.py
```

## Configuration

Edit `config.yaml` to customize:

```yaml
matrix_generation:
  sizes: [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
  dorr:
    alpha_values: [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    plot_sizes: [4, 8, 16, 32]  # Sizes for alpha plots
```

## Matrix Types

- **FEM**: 1D Poisson finite element matrices
- **Dorr**: Ill-conditioned test matrices with parameter α

## AFMP Configurations

- **Exact**: Full precision reference
- **High/Low/Medium/Very Low Approximation**: Different precision levels
- **Worst**: Minimal precision (chromo8=0)

## Test

Run individual components:
```bash
python matrix_generators.py  # Generate matrices
python parametric_afmp_analysis.py  # Analyze performance
python create_custom_plots.py  # Generate plots
```

Results saved to `results_parametric/`.