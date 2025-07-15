#!/usr/bin/env python3
"""
Debug script to find missing data points in AFPM analysis results
Compares expected vs actual combinations and identifies filtering issues
"""

import pickle
import pandas as pd
from pathlib import Path

def debug_missing_data_points():
    """Compare expected vs actual data points in analysis results"""
    
    print("=== DEBUGGING MISSING DATA POINTS ===")
    print("Comparing matrix files vs analysis results...\n")
    
    # Load analysis results
    results_file = Path("results_parametric/parametric_afpm_results.pkl")
    if not results_file.exists():
        print("❌ Results file not found! Run parametric analysis first.")
        return
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    df = pd.DataFrame(results)
    print(f"Analysis results: {len(df)} total records")
    
    # Check Dorr data specifically
    dorr_data = df[df['matrix_type'] == 'dorr']
    print(f"Dorr analysis results: {len(dorr_data)} records")
    
    # Count matrix files
    matrix_dir = Path("test_matrices_parametric/dorr")
    matrix_files = list(matrix_dir.glob("matrix_*.pkl"))
    print(f"Dorr matrix files: {len(matrix_files)} files")
    
    # Expected combinations
    expected_sizes = {4, 8, 16, 32, 64, 128}
    expected_alphas = {0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06}
    expected_configs = {"Exact", "Very Low Approx", "Low Approximation", 
                       "Medium Approximation", "High Approximation", "Worst (chromo8=0)"}
    
    expected_total = len(expected_sizes) * len(expected_alphas) * len(expected_configs)
    print(f"Expected total Dorr combinations: {expected_total}")
    print()
    
    # Check which combinations are missing from analysis results
    print("=== MISSING COMBINATIONS IN ANALYSIS RESULTS ===")
    missing_count = 0
    
    for size in sorted(expected_sizes):
        for alpha in sorted(expected_alphas):
            for config in sorted(expected_configs):
                # Check if this combination exists in results
                combo_data = dorr_data[
                    (dorr_data['matrix_size'] == size) &
                    (dorr_data['alpha'] == alpha) &
                    (dorr_data['config_name'] == config)
                ]
                
                if len(combo_data) == 0:
                    print(f"  Missing: Size {size}, α={alpha:.1e}, {config}")
                    missing_count += 1
                elif len(combo_data) > 1:
                    print(f"  Duplicate: Size {size}, α={alpha:.1e}, {config} ({len(combo_data)} entries)")
    
    print(f"\nTotal missing combinations: {missing_count}")
    
    # Check which matrix files don't have corresponding analysis results
    print("\n=== MATRIX FILES WITHOUT ANALYSIS RESULTS ===")
    unprocessed_count = 0
    
    for matrix_file in sorted(matrix_files):
        filename = matrix_file.name
        
        # Parse filename: matrix_n00128_alpha1e-06_id00.pkl
        parts = filename.replace('.pkl', '').split('_')
        size_str = parts[1][1:]  # Remove 'n' prefix
        size = int(size_str)
        alpha_str = parts[2].replace('alpha', '')  # Remove 'alpha' prefix
        alpha = float(alpha_str.replace('e-0', 'e-').replace('e-', 'e-'))
        
        # Check if any analysis results exist for this matrix
        matrix_results = dorr_data[
            (dorr_data['matrix_size'] == size) &
            (dorr_data['alpha'] == alpha)
        ]
        
        if len(matrix_results) == 0:
            print(f"  No analysis results for: {filename}")
            unprocessed_count += 1
        elif len(matrix_results) != len(expected_configs):
            print(f"  Partial results for: {filename} ({len(matrix_results)}/{len(expected_configs)} configs)")
    
    print(f"\nMatrix files without complete analysis: {unprocessed_count}")
    
    # Check for extreme errors that might be filtered
    print("\n=== EXTREME ERROR ANALYSIS ===")
    
    if len(dorr_data) > 0:
        # Check error distribution
        error_stats = dorr_data['mean_relative_l2_error'].describe()
        print("Error statistics:")
        print(f"  Min: {error_stats['min']:.2e}")
        print(f"  Max: {error_stats['max']:.2e}")
        print(f"  Mean: {error_stats['mean']:.2e}")
        print(f"  Std: {error_stats['std']:.2e}")
        
        # Count extreme errors
        extreme_errors = dorr_data[dorr_data['mean_relative_l2_error'] >= 1.0]
        print(f"\nCombinations with ≥100% error: {len(extreme_errors)}")
        
        if len(extreme_errors) > 0:
            print("Extreme error combinations:")
            for _, row in extreme_errors.iterrows():
                print(f"  Size {row['matrix_size']}, α={row['alpha']:.1e}, "
                      f"{row['config_name']}: Error={row['mean_relative_l2_error']:.1e}")
        
        # Check condition number correlation
        extreme_condition = dorr_data[dorr_data['mean_condition_number'] > 1e15]
        print(f"\nCombinations with extreme condition numbers (>1e15): {len(extreme_condition)}")
        
        if len(extreme_condition) > 0:
            print("Extreme condition number combinations:")
            for _, row in extreme_condition.iterrows():
                print(f"  Size {row['matrix_size']}, α={row['alpha']:.1e}, "
                      f"{row['config_name']}: κ={row['mean_condition_number']:.1e}, "
                      f"Error={row['mean_relative_l2_error']:.1e}")

if __name__ == "__main__":
    debug_missing_data_points()