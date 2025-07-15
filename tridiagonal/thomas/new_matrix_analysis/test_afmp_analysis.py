#!/usr/bin/env python3
"""
Test script to diagnose AFMP analysis issues
Tests AFMP processing for matrices that should exist but might fail analysis
"""

import numpy as np
import pickle
from pathlib import Path
from parametric_afmp_analysis import ParametricAFPMAnalysis

def test_afmp_on_existing_matrices():
    """Test AFMP analysis on all existing matrix files"""
    
    print("=== AFMP ANALYSIS TEST ===")
    print("Testing AFMP processing on existing matrix files...\n")
    
    # Create AFMP analyzer instance
    analyzer = ParametricAFPMAnalysis()
    
    # AFMP configurations to test
    configurations = {
        "Exact": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        "Very Low Approx": [20, 15, 10, 5, 0, 0, 0, 0, 0],
        "Low Approximation": [40, 30, 20, 10, 5, 0, 0, 0, 0],
        "Medium Approximation": [60, 50, 40, 30, 20, 10, 5, 0, 0],
        "High Approximation": [80, 70, 60, 50, 40, 30, 20, 10, 0],
        "Worst (chromo8=0)": [99, 99, 99, 99, 99, 99, 99, 99, 0]
    }
    
    matrix_dir = Path("test_matrices_parametric")
    success_count = 0
    failure_count = 0
    non_finite_count = 0
    
    # Test Dorr matrices specifically
    dorr_dir = matrix_dir / "dorr"
    matrix_files = sorted(list(dorr_dir.glob("matrix_*.pkl")))
    
    print(f"Found {len(matrix_files)} Dorr matrix files to test")
    
    # Group by size to check for missing combinations
    size_alpha_combinations = {}
    
    for filepath in matrix_files:
        try:
            # Load matrix
            with open(filepath, 'rb') as f:
                matrix_data = pickle.load(f)
            
            size = matrix_data['size']
            alpha = matrix_data['parameters']['alpha']
            condition_number = matrix_data['condition_number']
            
            # Track combinations
            if size not in size_alpha_combinations:
                size_alpha_combinations[size] = set()
            size_alpha_combinations[size].add(alpha)
            
            # Test each AFMP configuration
            for config_name, chromosome in configurations.items():
                try:
                    # Get Thomas format data
                    a = matrix_data['a'] 
                    b = matrix_data['b']
                    c = matrix_data['c'] 
                    d = matrix_data['d']
                    
                    # Compute exact solution first
                    n = len(d)
                    x_exact = analyzer.thomas_solve_double_precision(n, a.copy(), b.copy(), c.copy(), d.copy())
                    
                    # Check if exact solution is valid
                    if np.any(np.isnan(x_exact)) or np.any(np.isinf(x_exact)):
                        print(f"  Size {size}, α={alpha:.1e}, {config_name}: FAILED - Exact solution non-finite")
                        failure_count += 1
                        continue
                    
                    # Compute AFMP solution
                    x_afmp = analyzer.thomas_solve_afpm(n, a.copy(), b.copy(), c.copy(), d.copy(), chromosome)
                    
                    # Check if AFMP solution is valid
                    if np.any(np.isnan(x_afmp)) or np.any(np.isinf(x_afmp)):
                        print(f"  Size {size}, α={alpha:.1e}, {config_name}: FAILED - AFMP solution non-finite")
                        non_finite_count += 1
                        continue
                    
                    # Compute relative error
                    norm_exact = np.linalg.norm(x_exact)
                    if norm_exact == 0:
                        print(f"  Size {size}, α={alpha:.1e}, {config_name}: FAILED - Zero exact solution norm")
                        failure_count += 1
                        continue
                    
                    error_norm = np.linalg.norm(x_afmp - x_exact)
                    relative_error = error_norm / norm_exact
                    
                    # Check if error is finite
                    if np.isnan(relative_error) or np.isinf(relative_error):
                        print(f"  Size {size}, α={alpha:.1e}, {config_name}: FAILED - Non-finite error")
                        non_finite_count += 1
                        continue
                    
                    # Success case
                    if condition_number > 1e15:
                        print(f"  Size {size}, α={alpha:.1e}, {config_name}: SUCCESS - Error: {relative_error:.2e} (Extreme κ={condition_number:.1e})")
                    else:
                        success_count += 1
                    
                except Exception as analysis_e:
                    print(f"  Size {size}, α={alpha:.1e}, {config_name}: FAILED - Analysis error: {analysis_e}")
                    failure_count += 1
                    
        except Exception as file_e:
            print(f"  Error loading {filepath.name}: {file_e}")
            failure_count += 1
    
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Successful AFMP analyses: {success_count}")
    print(f"Failed analyses: {failure_count}")
    print(f"Non-finite results: {non_finite_count}")
    print(f"Total tests: {success_count + failure_count + non_finite_count}")
    
    # Check for missing size-alpha combinations
    print(f"\n=== SIZE-ALPHA COMBINATION CHECK ===")
    expected_alphas = {0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06}
    expected_sizes = {4, 8, 16, 32, 64, 128}
    
    for size in sorted(expected_sizes):
        if size in size_alpha_combinations:
            actual_alphas = size_alpha_combinations[size]
            missing_alphas = expected_alphas - actual_alphas
            if missing_alphas:
                print(f"Size {size}: Missing alphas {missing_alphas}")
            else:
                print(f"Size {size}: All 6 alphas present ✓")
        else:
            print(f"Size {size}: COMPLETELY MISSING!")
    
    return success_count, failure_count, non_finite_count

if __name__ == "__main__":
    test_afmp_on_existing_matrices()