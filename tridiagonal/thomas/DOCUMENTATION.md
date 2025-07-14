# AFPM Analysis Framework

## Overview

This repository contains a comprehensive analysis framework for **Approximate Fixed-Point Multipliers (AFPM)** applied to the Thomas algorithm for solving tridiagonal linear systems. The framework evaluates the trade-off between computational efficiency and numerical accuracy in approximate computing applications.

## Table of Contents

1. [Background](#background)
2. [Framework Architecture](#framework-architecture)
3. [Core Components](#core-components)
4. [Usage Guide](#usage-guide)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Results and Analysis](#results-and-analysis)
7. [File Structure](#file-structure)
8. [Dependencies](#dependencies)

## Background

### Approximate Fixed-Point Multipliers (AFPM)

AFPM is a hardware acceleration technique that trades numerical precision for:
- **Energy efficiency**: Reduced power consumption
- **Speed improvement**: Faster multiplication operations
- **Area reduction**: Smaller hardware footprint

The key innovation is using **chromosome-based configuration** to control the approximation level of individual mantissa bits in IEEE 754 floating-point multiplication.

### Thomas Algorithm

The Thomas algorithm is an O(n) method for solving tridiagonal linear systems of the form:
```
Ax = d
```
where A is a tridiagonal matrix. It's widely used in:
- **Finite Element Methods (FEM)**
- **Computational Fluid Dynamics (CFD)**
- **Heat transfer simulations**
- **Signal processing applications**

## Framework Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Matrix         │    │  AFPM Analysis   │    │  Results &      │
│  Generation     │───▶│  Engine          │───▶│  Visualization  │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   test_matrices/          Enhanced Analysis        results/
   ├── toeplitz/           ├── 6 Configurations    ├── Plots
   └── fem_poisson/        ├── Error Analysis      ├── Tables
                          └── Condition Study      └── Data
                          
### Modular Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  config.yaml    │    │  Data Processing │    │  Visualization  │
│  Configuration  │───▶│  & Analysis      │───▶│  Engine         │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   AFPM configs            enhanced_afpm_          plot_afmp_
   Matrix sizes            analysis.py             results.py
   Error settings          (Analysis only)         (Plotting only)
```

## Core Components

### 1. AFPM.py
**Core AFPM Implementation**

```python
def AFPM(a_binary, b_binary, chromosome):
    """
    Approximate Fixed-Point Multiplier
    
    Args:
        a_binary: 32-bit binary representation of first operand
        b_binary: 32-bit binary representation of second operand  
        chromosome: 9-element list controlling approximation levels
                   [0-7]: Mantissa bits 0-7 approximation (0-99)
                   [8]: Mantissa bit 8 approximation (0-99)
    
    Returns:
        32-bit binary result of approximate multiplication
    """
```

**Chromosome Configuration:**
- **Values 0-99**: Approximation percentage (0=exact, 99=maximum approximation)
- **Value -1**: Exact hardware multiplication
- **Index [8]**: Controls most significant mantissa bit (critical for accuracy)

### 2. matrix_generator.py
**Test Matrix Generation Engine**

**Purpose:** Generate comprehensive test datasets for robust statistical analysis.

**Key Features:**
- **Matrix sizes:** 4 to 2^16 (65,536) elements
- **Matrix types:** 2 types × 15 sizes × 10 instances = 300 total matrices
- **Reproducible:** Consistent seeding for identical results across runs
- **Auto-cleanup:** Removes existing matrices before generation

**Matrix Types:**

1. **Tridiagonal Toeplitz** `[-1, 2, -1]`
   - **Application:** Heat diffusion, wave equations
   - **Condition number:** κ(A) ≈ n²/π²
   - **Properties:** Well-structured, predictable conditioning

2. **FEM Poisson Discretization** `[-1/h², 2/h², -1/h²]`
   - **Application:** Elliptic PDEs, structural mechanics
   - **Condition number:** κ(A) ≈ (n+1)²/π²
   - **Properties:** Physically meaningful, ill-conditioned for large n

**Directory Structure:**
```
test_matrices/
├── tridiagonal_toeplitz/
│   ├── matrix_n00004_id00.pkl  # Size 4, instance 0
│   ├── matrix_n00004_id01.pkl  # Size 4, instance 1
│   └── ...
├── fem_poisson/
│   ├── matrix_n00004_id00.pkl
│   └── ...
└── matrix_summary.pkl          # Metadata and statistics
```

### 3. config.yaml
**Centralized Configuration Management**

**Purpose:** Single configuration file controlling all analysis parameters.

**Key Sections:**
```yaml
matrix_generation:
  size_range:
    min_size: 4
    max_size: 65536          # 2^16
  matrix_types:
    - "tridiagonal_toeplitz"
    - "fem_poisson"
  matrices_per_size: 100     # Statistical robustness

afpm_analysis:
  configurations:            # 6 AFPM configurations
    "Exact": [-1, -1, ...]  # Hardware exact
    "Very Low Approx": [20, 15, ...]
    # ... 4 more configurations
    
plotting:
  figure_size: [16, 12]
  plot_types: ["logarithmic", "configuration_table"]
```

### 4. enhanced_afpm_analysis.py
**Analysis-Only Engine (No Plotting)**

**Purpose:** Generate statistical results and save to pickle files.

**Key Features:**
- **Modular design:** Analysis separated from visualization
- **Config-driven:** All parameters from config.yaml
- **Robust statistics:** 100 matrices per size for error bars
- **Double precision ground truth:** Library-quality baseline

**Analysis Pipeline:**
```python
# 1. Load configuration
config = load_config("config.yaml")

# 2. Process matrices
for matrix_type in ["tridiagonal_toeplitz", "fem_poisson"]:
    for size in [4, 8, 16, ..., 65536]:
        # Load 100 matrices per size
        size_errors = []
        for matrix_data in size_matrices:
            # Ground truth (double precision)
            x_double = thomas_solve_double_precision(...)
            
            # AFPM approximation 
            x_afpm = thomas_solve_afpm(..., chromosome)
            
            # Relative L2 Error (primary metric)
            error = np.linalg.norm(x_afpm - x_double) / np.linalg.norm(x_double)
            size_errors.append(error)
        
        # Statistical aggregation
        results.append({
            'mean_relative_l2_error': np.mean(size_errors),
            'std_relative_l2_error': np.std(size_errors),
            'float32_theoretical_bound': compute_theoretical_bound(...)
        })

# 3. Save results only (no plotting)
pickle.dump(results, open("results/afpm_analysis_results.pkl", "wb"))
```

### 5. plot_afpm_results.py  
**Visualization-Only Engine**

**Purpose:** Create plots from saved analysis results.

**Key Features:**
- **Independent operation:** Reads from pickle files
- **Multiple plot types:** Logarithmic, linear, configuration tables
- **Theoretical bounds:** Float32 bound prominently displayed
- **Application thresholds:** Scientific, engineering, real-time, IoT error levels

**Configuration Space:**

| Configuration | Chromosome | Description | Use Case |
|---------------|------------|-------------|----------|
| Exact | [-1, -1, -1, -1, -1, -1, -1, -1, -1] | Hardware exact | Reference baseline |
| Very Low Approx | [5, 5, 5, 5, 5, 5, 5, 5, 0] | Minimal approximation | High-precision applications |
| Low Approximation | [20, 20, 20, 20, 20, 20, 20, 20, 0] | Light efficiency gain | Scientific computing |
| Medium Approximation | [40, 40, 40, 40, 40, 40, 40, 40, 0] | Balanced trade-off | Engineering simulations |
| High Approximation | [70, 70, 70, 70, 70, 70, 70, 70, 0] | High efficiency | Real-time processing |
| Worst (chromo8=0) | [99, 99, 99, 99, 99, 99, 99, 99, 0] | Maximum approximation | IoT/embedded systems |

## Usage Guide

### Step 1: Configure Analysis (Optional)

```bash
# Edit config.yaml to customize:
# - Matrix sizes and types
# - AFPM configurations  
# - Plot settings
# - Number of matrices per size
nano config.yaml
```

### Step 2: Generate Test Matrices

```bash
python3 matrix_generator.py
```

**Output:**
- Creates `test_matrices/` directory
- Generates 3000 matrices (2 types × 15 sizes × 100 instances)
- Saves metadata and condition numbers
- Estimated time: 5-15 minutes (depending on matrices_per_size)

### Step 3: Run AFPM Analysis

```bash
python3 enhanced_afpm_analysis.py
```

**Output:**
- Creates `results/` directory  
- Saves statistical results to pickle file
- **No plots generated** (analysis only)
- Estimated time: 10-30 minutes

### Step 4: Generate Visualizations

```bash
python3 plot_afpm_results.py
```

**Output:**
- Reads from `results/afpm_analysis_results.pkl`
- Creates comprehensive plots and tables
- Estimated time: 1-2 minutes

### Step 5: Examine Results

```bash
ls results/
# afpm_config_table.png              - Configuration specifications
# afpm_analysis_logarithmic.png      - Log-scale analysis plots
# afpm_analysis_results.pkl          - Raw statistical data (180 data points)
```

### Advanced Usage

**Quick plotting without re-analysis:**
```bash
# If analysis results exist, just regenerate plots
python3 plot_afpm_results.py
```

**Custom configuration:**
```bash
# Modify config.yaml, then re-run analysis
python3 enhanced_afpm_analysis.py
python3 plot_afpm_results.py
```

## Mathematical Foundation

### Forward Error Analysis

The theoretical framework is based on forward error bounds for linear systems:

**Primary Error Metric (Relative L2 Error):**
```
E_RelL2 = ||x_afpm - x_double||₂ / ||x_double||₂
```

**Theoretical Upper Bound:**
```
E_RelL2 ≤ C × ε_float32 × κ(A)
```

Where:
- **E_RelL2**: Relative L2 error vs double precision ground truth  
- **C**: Algorithm constant (≈ 3 for Thomas algorithm)
- **ε_float32**: Float32 machine epsilon = 2^(-23) ≈ 1.19×10^(-7)
- **κ(A)**: Matrix condition number (SVD-based calculation)

**Theoretical Bound Behavior:**
- **Linear growth:** E ∝ κ(A) for well-conditioned systems
- **Plateau region:** Bound becomes loose for highly ill-conditioned systems  
- **Can exceed 100%:** No mathematical upper limit on relative error

### AFPM Error Model

The approximation error ε_AFPM depends on chromosome configuration:

```python
# IEEE 754 mantissa bit weights
bit_weights = [2^(-23+i) for i in range(9)]  # Bits 0-8

# AFPM
if chromosome[i] == -1:
    bit_error[i] = 0                    # Exact

ε_AFPM = sum(bit_error)
```

### Condition Number Scaling

**Theoretical scaling laws:**

1. **Tridiagonal Toeplitz:** κ(A) ≈ n²/π² ≈ 0.101 × n²
2. **FEM Poisson:** κ(A) ≈ (n+1)²/π² ≈ 0.101 × (n+1)²

**Real-world implications:**
- **n = 1024**: κ(A) ≈ 10⁶ (moderately ill-conditioned)
- **n = 4096**: κ(A) ≈ 10⁷ (ill-conditioned)  
- **n = 32768**: κ(A) ≈ 10⁹ (severely ill-conditioned)

## Results and Analysis

### Key Findings

1. **Error Convergence at Large Scales**
   - All AFPM configurations converge to similar error levels for n > 2^14
   - Indicates condition number dominance over approximation quality
   - Thomas algorithm becomes unstable for κ(A) > 10^8

2. **Sweet Spot for AFPM**
   - **Optimal range:** 64 ≤ n ≤ 4096
   - **Condition numbers:** 10⁴ ≤ κ(A) ≤ 10⁸
   - **Applications:** Embedded systems, real-time processing

3. **Theoretical Bound Validation**
   - **Float32 theoretical bound:** Prominently displayed in all plots
   - **Plateau behavior:** Theoretical bound shows non-linear behavior at high κ(A)
   - **Ground truth:** Double precision Thomas algorithm as baseline
   - **Error scaling:** Measured errors follow theoretical predictions for well-conditioned systems

4. **Configuration Performance**
   - **6 AFPM configurations:** From exact to maximum approximation
   - **Chromosome [8] = 0:** Most significant bit kept exact in all configurations
   - **Progressive approximation:** Lower configurations show better accuracy
   - **Convergence at high κ(A):** All configurations approach similar error levels

### Performance Metrics

**Accuracy Levels (Relative L2 Error):**
- **Exact:** ~10^(-7) (machine precision baseline)
- **Very Low Approx:** ~10^(-6) to 10^(-5)
- **Medium Approx:** ~10^(-4) to 10^(-3)
- **High Approx:** ~10^(-2) to 10^(-1)

**Real-World Tolerance:**
- **Engineering:** 1-5% error acceptable → Medium Approximation
- **Scientific:** 0.1-1% error typical → Low Approximation
- **Embedded/IoT:** 10% error tolerable → High Approximation

## File Structure

```
tridiagonal/thomas/
├── README.md                     # This documentation
├── config.yaml                   # Centralized configuration
├── config_loader.py              # Configuration management
├── AFPM.py                      # Core AFPM implementation
├── matrix_generator.py          # Test matrix generation
├── enhanced_afpm_analysis.py    # Analysis engine (data only)
├── plot_afpm_results.py         # Visualization engine (plots only)
├── test_matrices/               # Generated matrices (3000 total)
│   ├── tridiagonal_toeplitz/    # 1500 matrices (15 sizes × 100 instances)
│   ├── fem_poisson/             # 1500 matrices (15 sizes × 100 instances)
│   └── matrix_summary.pkl       # Metadata and statistics
└── results/                     # Analysis outputs
    ├── afpm_config_table.png            # Configuration table
    ├── afpm_analysis_logarithmic.png    # Log-scale plots with theoretical bounds
    └── afmp_analysis_results.pkl        # Statistical data (180 data points)
```

## Dependencies

**Required Python packages:**
```bash
pip install numpy matplotlib pandas scipy pathlib pickle struct
```

**System requirements:**
- **Python:** 3.7+
- **Memory:** 4GB RAM (for large matrices)
- **Storage:** 1GB (for matrix datasets)
- **Time:** 10-20 minutes total runtime

## Technical Notes

### Numerical Stability Considerations

1. **Thomas Algorithm Limitations**
   - No pivoting → sensitive to ill-conditioning
   - Breakdown for κ(A) > 10^12
   - Alternative: Use pivoted LU or iterative methods

2. **Ground Truth Selection**
   - **Current approach:** Double precision Thomas algorithm
   - **Rationale:** Same algorithm, different precision (fair comparison)
   - **Issue:** Shares same instability as AFPM version for high κ(A)
   - **Alternative:** Library solver with pivoting for true ground truth

3. **Theoretical Bound Behavior**
   - **Plateau regions:** Normal behavior for ill-conditioned systems
   - **Error > 100%:** Mathematically valid, no upper bound on relative error
   - **Condition number limits:** SVD calculation precision affects κ(A) estimates

<!-- 3. **AFPM Implementation**
   - Based on IEEE 754 binary manipulation
   - Assumes little-endian bit ordering
   - Error handling for edge cases -->

<!-- ### Performance Optimization

1. **Matrix Generation**
   - Vectorized numpy operations
   - Efficient condition number computation
   - Parallel generation possible

2. **Analysis Engine**
   - Batch processing of matrix instances
   - Statistical aggregation for robustness
   - Memory-efficient data structures

### Future Enhancements

1. **Extended Analysis**
   - More matrix types (sparse, structured)
   - Additional error metrics (element-wise, infinity norm)
   - Performance profiling (timing, energy)

2. **Advanced AFPM**
   - Dynamic chromosome adaptation
   - Machine learning optimization
   - Hardware synthesis evaluation

3. **Applications**
   - Real-world benchmark problems
   - Domain-specific error tolerance studies
   - Comparison with other approximate methods -->

## Citation
<!-- 
If you use this framework in your research, please cite:

```bibtex
@software{afpm_analysis_framework,
  title = {AFPM Analysis Framework for Tridiagonal Linear Systems},
  author = {[Your Name]},
  year = {2024},
  description = {Comprehensive analysis of Approximate Fixed-Point Multipliers 
                applied to Thomas algorithm for tridiagonal systems}
}
``` -->

## License

This project is released under the MIT License. See LICENSE file for details.

## Contact

For questions, issues, or contributions, please contact [your contact information].