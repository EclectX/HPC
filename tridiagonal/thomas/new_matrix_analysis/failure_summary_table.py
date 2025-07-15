#!/usr/bin/env python3
"""
Simple Failure Summary Table: Clean table showing average failure rates by configuration
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def create_failure_summary():
    """Create simple failure summary table"""
    
    # Load results
    results_file = Path("results_parametric/parametric_afpm_results.pkl")
    if not results_file.exists():
        print("❌ Results file not found. Run parametric_afmp_analysis.py first.")
        return
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    df = pd.DataFrame(results)
    
    print("=== AFPM FAILURE SUMMARY TABLE ===\n")
    
    # Calculate summary statistics by configuration
    summary_data = []
    
    for config in sorted(df['config_name'].unique()):
        config_data = df[df['config_name'] == config]
        
        # Overall statistics
        total_attempted = config_data['num_matrices_attempted'].sum()
        total_successful = config_data['num_matrices_successful'].sum()
        total_failed = config_data['num_matrices_failed'].sum()
        
        avg_success_rate = config_data['success_rate'].mean() * 100
        
        # Count different failure types
        all_failure_reasons = []
        for reasons in config_data['failure_reasons']:
            all_failure_reasons.extend(reasons)
        
        condition_failures = sum(1 for r in all_failure_reasons if 'condition_number_too_high' in r)
        exact_failures = sum(1 for r in all_failure_reasons if 'double_precision_failed' in r)
        afpm_failures = sum(1 for r in all_failure_reasons if 'afpm_solution_failed' in r)
        other_failures = len(all_failure_reasons) - condition_failures - exact_failures - afpm_failures
        
        # Matrix size range
        size_range = f"{config_data['matrix_size'].min()}-{config_data['matrix_size'].max()}"
        
        summary_data.append({
            'Configuration': config,
            'Avg Success Rate (%)': f"{avg_success_rate:.1f}",
            'Total Attempted': total_attempted,
            'Total Successful': total_successful,
            'Total Failed': total_failed,
            'Condition Failures': condition_failures,
            'Exact Failures': exact_failures,
            'AFPM Failures': afpm_failures,
            'Other Failures': other_failures,
            'Size Range': size_range
        })
    
    # Create and display table
    summary_df = pd.DataFrame(summary_data)
    
    # Print formatted table
    print("Configuration Performance Summary:")
    print("=" * 120)
    print(f"{'Config':<20} {'Success Rate':<12} {'Attempted':<10} {'Success':<8} {'Failed':<7} {'Cond':<5} {'Exact':<6} {'AFMP':<5} {'Other':<6} {'Sizes':<12}")
    print("-" * 120)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Configuration']:<20} "
              f"{row['Avg Success Rate (%)']:<12} "
              f"{row['Total Attempted']:<10} "
              f"{row['Total Successful']:<8} "
              f"{row['Total Failed']:<7} "
              f"{row['Condition Failures']:<5} "
              f"{row['Exact Failures']:<6} "
              f"{row['AFMP Failures']:<5} "
              f"{row['Other Failures']:<6} "
              f"{row['Size Range']:<12}")
    
    print("-" * 120)
    print("Legend: Cond=Condition too high, Exact=Double precision failed, AFMP=AFMP failed, Other=Other reasons")
    print()
    
    # Key insights
    print("🔍 KEY INSIGHTS:")
    
    # Find most robust configuration
    best_config = summary_df.loc[summary_df['Avg Success Rate (%)'].astype(float).idxmax(), 'Configuration']
    best_rate = summary_df.loc[summary_df['Avg Success Rate (%)'].astype(float).idxmax(), 'Avg Success Rate (%)']
    print(f"   Most robust configuration: {best_config} ({best_rate}% success)")
    
    # Check AFMP failures
    total_afmp_failures = summary_df['AFMP Failures'].sum()
    total_all_failures = summary_df['Total Failed'].sum()
    print(f"   AFMP-caused failures: {total_afmp_failures}/{total_all_failures} ({total_afmp_failures/total_all_failures*100 if total_all_failures > 0 else 0:.1f}%)")
    
    # Most common failure type
    total_condition = summary_df['Condition Failures'].sum()
    total_exact = summary_df['Exact Failures'].sum()
    print(f"   Most common failure: {'Condition too high' if total_condition > total_exact else 'Exact algorithm limit'}")
    
    print(f"   Matrix sizes tested: {df['matrix_size'].min()} to {df['matrix_size'].max()}")
    
    # Save detailed CSV for further analysis
    output_file = Path("results_parametric/failure_summary_table.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"\n📄 Detailed table saved to: {output_file}")

def create_size_breakdown_table():
    """Create breakdown by matrix size"""
    
    results_file = Path("results_parametric/parametric_afpm_results.pkl")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    df = pd.DataFrame(results)
    
    print("\n=== SUCCESS RATE BY MATRIX SIZE ===\n")
    
    # Group by matrix size
    size_summary = df.groupby('matrix_size').agg({
        'success_rate': 'mean',
        'num_matrices_attempted': 'sum',
        'num_matrices_successful': 'sum',
        'num_matrices_failed': 'sum'
    }).round(3)
    
    print("Matrix Size Analysis:")
    print("=" * 80)
    print(f"{'Size':<8} {'Success Rate (%)':<15} {'Attempted':<12} {'Successful':<12} {'Failed':<8}")
    print("-" * 80)
    
    for size, row in size_summary.iterrows():
        success_pct = row['success_rate'] * 100
        print(f"{size:<8} {success_pct:<15.1f} {row['num_matrices_attempted']:<12.0f} "
              f"{row['num_matrices_successful']:<12.0f} {row['num_matrices_failed']:<8.0f}")
    
    print("-" * 80)
    print()

def main():
    """Main function"""
    try:
        create_failure_summary()
        create_size_breakdown_table()
        
        print("✅ Analysis complete!")
        print("\nTo run analysis for larger matrices:")
        print("1. python3 matrix_generators.py  (generates matrices up to 8192)")
        print("2. python3 parametric_afmp_analysis.py  (runs analysis)")
        print("3. python3 failure_summary_table.py  (creates this summary)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()