#!/usr/bin/env python3
"""
Custom Plotting Script: Creates specific plots as requested
1. FEM: Matrix size vs error (all configs + theoretical bound)
2. Dorr: Alpha vs error (4 plots for 4 sizes)  
3. Condition vs error: FEM (1 plot) + Dorr (4 plots for 4 sizes)
4. Success rate table by configuration
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import pickle
from pathlib import Path
from config_loader import load_config

def load_results_and_config():
    """Load results and configuration"""
    # Load config
    config = load_config("config.yaml")
    
    # Load results
    results_file = Path("results_parametric/parametric_afpm_results.pkl")
    if not results_file.exists():
        print("❌ Results file not found. Run parametric_afmp_analysis.py first.")
        return None, None
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    return results, config

def get_config_colors():
    """Get configuration colors"""
    try:
        config = load_config("config.yaml")
        config_dict = config.get_afpm_configurations()
        color_map = {}
        for name, config_data in config_dict.items():
            color_map[name] = config_data.get('color', '#cccccc')
    except:
        # Fallback colors
        color_map = {
            "Exact": "#1f77b4",
            "Very Low Approx": "#ff7f0e", 
            "Low Approximation": "#2ca02c",
            "Medium Approximation": "#d62728",
            "High Approximation": "#9467bd",
            "Worst (chromo8=0)": "#8c564b"
        }
    return color_map

def plot_fem_size_vs_error(df, color_map):
    """Plot 1: FEM Matrix Size vs Error (all configs + theoretical bound)"""
    
    fem_data = df[df['matrix_type'] == 'fem_poisson'].copy()
    if fem_data.empty:
        print("No FEM data found")
        return
    
    # Convert to percentage
    fem_data['mean_relative_l2_error_pct'] = fem_data['mean_relative_l2_error'] * 100
    fem_data['std_relative_l2_error_pct'] = fem_data['std_relative_l2_error'] * 100
    fem_data['float32_theoretical_bound_pct'] = fem_data['float32_theoretical_bound'] * 100
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot each configuration
    for config in sorted(fem_data['config_name'].unique()):
        config_data = fem_data[fem_data['config_name'] == config]
        if not config_data.empty:
            ax.errorbar(config_data['matrix_size'], config_data['mean_relative_l2_error_pct'],
                       yerr=config_data['std_relative_l2_error_pct'],
                       label=config, color=color_map.get(config, '#cccccc'), 
                       linewidth=2, markersize=8, capsize=4, 
                       marker='o', alpha=0.8)
    
    # Add Float32 theoretical bound
    sizes = sorted(fem_data['matrix_size'].unique())
    float32_bounds = []
    for size in sizes:
        bound_data = fem_data[fem_data['matrix_size'] == size]
        if not bound_data.empty:
            float32_bounds.append(bound_data['float32_theoretical_bound_pct'].iloc[0])
    
    if float32_bounds:
        ax.loglog(sizes, float32_bounds, 'k--', linewidth=4, alpha=0.9, 
                 label='Float32 Theory Bound', zorder=10)
    
    ax.set_xlabel('Matrix Size', fontsize=12)
    ax.set_ylabel('Relative L2 Error (%)', fontsize=12)
    ax.set_title('FEM 1D Poisson: Matrix Size vs Error\n(All AFPM Configurations + Theoretical Bound)', 
                 fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    results_dir = Path("results_parametric")
    plot_file = results_dir / "fem_size_vs_error.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ FEM size vs error plot saved to: {plot_file}")

def plot_dorr_alpha_vs_error(df, color_map):
    """Plot 2: Dorr Alpha vs Error (4 plots for 4 sizes from config)"""
    
    dorr_data = df[df['matrix_type'] == 'dorr'].copy()
    if dorr_data.empty or 'alpha' not in dorr_data.columns:
        print("No Dorr data found")
        return
    
    # Convert to percentage
    dorr_data['mean_relative_l2_error_pct'] = dorr_data['mean_relative_l2_error'] * 100
    dorr_data['std_relative_l2_error_pct'] = dorr_data['std_relative_l2_error'] * 100
    
    # Get specific sizes from config file
    try:
        config = load_config("config.yaml")
        selected_sizes = config.config.get('matrix_generation', {}).get('dorr', {}).get('plot_sizes', [4, 16, 64, 128])
        print(f"Using Dorr plot sizes from config: {selected_sizes}")
    except:
        # Fallback to automatic selection
        available_sizes = sorted(dorr_data['matrix_size'].unique())
        if len(available_sizes) >= 4:
            selected_sizes = [available_sizes[0], available_sizes[len(available_sizes)//3], 
                             available_sizes[2*len(available_sizes)//3], available_sizes[-1]]
        else:
            selected_sizes = available_sizes
        print(f"Config read failed, using auto-selected sizes: {selected_sizes}")
    
    # Verify sizes exist in data
    available_sizes = sorted(dorr_data['matrix_size'].unique())
    selected_sizes = [size for size in selected_sizes if size in available_sizes][:4]
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, size in enumerate(selected_sizes[:4]):
        ax = axes[idx]
        size_data = dorr_data[dorr_data['matrix_size'] == size]
        
        # Plot each configuration
        for config in sorted(size_data['config_name'].unique()):
            config_data = size_data[size_data['config_name'] == config]
            if not config_data.empty:
                ax.errorbar(config_data['alpha'], config_data['mean_relative_l2_error_pct'],
                           yerr=config_data['std_relative_l2_error_pct'],
                           label=config, color=color_map.get(config, '#cccccc'), 
                           linewidth=2, markersize=6, capsize=3, 
                           marker='s', alpha=0.8)
        
        ax.set_xlabel('Alpha Parameter', fontsize=11)
        ax.set_ylabel('Relative L2 Error (%)', fontsize=11)
        ax.set_title(f'Dorr Matrix {size}×{size}', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        if idx == 0:  # Only show legend on first subplot
            ax.legend(fontsize=8)
    
    # Hide unused subplots if less than 4 sizes
    for idx in range(len(selected_sizes), 4):
        axes[idx].set_visible(False)
    
    plt.suptitle('Dorr Matrices: Alpha Parameter vs Error\n(4 Different Matrix Sizes)', 
                 fontweight='bold', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    results_dir = Path("results_parametric")
    plot_file = results_dir / "dorr_alpha_vs_error.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Dorr alpha vs error plots saved to: {plot_file}")

def plot_condition_vs_error(df, color_map):
    """Plot 3: Condition Number vs Error (FEM + Dorr for 4 sizes)"""
    
    # Create figure with 1 FEM plot + 4 Dorr plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: FEM Condition vs Error
    ax_fem = axes[0, 0]
    fem_data = df[df['matrix_type'] == 'fem_poisson'].copy()
    if not fem_data.empty:
        fem_data['mean_relative_l2_error_pct'] = fem_data['mean_relative_l2_error'] * 100
        
        # Plot each configuration
        for config in sorted(fem_data['config_name'].unique()):
            config_data = fem_data[fem_data['config_name'] == config].sort_values('mean_condition_number')
            if not config_data.empty:
                ax_fem.plot(config_data['mean_condition_number'], 
                           config_data['mean_relative_l2_error_pct'],
                           label=config, color=color_map.get(config, '#cccccc'), 
                           linewidth=2, marker='o', markersize=6, alpha=0.8)
        
        # Add theoretical bound
        conditions = fem_data['mean_condition_number'].values
        if len(conditions) > 0:
            cond_range = np.logspace(np.log10(min(conditions)), np.log10(max(conditions)), 100)
            theoretical_line = 3 * 1.19e-7 * cond_range * 100
            ax_fem.loglog(cond_range, theoretical_line, 'k--', linewidth=3, alpha=0.9, 
                         label='Theory: E ∝ κ(A)')
        
        ax_fem.set_xlabel('Condition Number κ(A)', fontsize=11)
        ax_fem.set_ylabel('Relative L2 Error (%)', fontsize=11)
        ax_fem.set_title('FEM: Condition vs Error', fontweight='bold', fontsize=12)
        ax_fem.legend(fontsize=8)
        ax_fem.grid(True, alpha=0.3)
        ax_fem.set_xscale('log')
        ax_fem.set_yscale('log')
    
    # Plots 2-5: Dorr Condition vs Error for 4 sizes from config
    dorr_data = df[df['matrix_type'] == 'dorr'].copy()
    if not dorr_data.empty:
        dorr_data['mean_relative_l2_error_pct'] = dorr_data['mean_relative_l2_error'] * 100
        
        # Get specific sizes from config file
        try:
            config = load_config("config.yaml")
            selected_sizes = config.config.get('matrix_generation', {}).get('dorr', {}).get('plot_sizes', [4, 16, 64, 128])
        except:
            # Fallback to automatic selection
            available_sizes = sorted(dorr_data['matrix_size'].unique())
            if len(available_sizes) >= 4:
                selected_sizes = [available_sizes[0], available_sizes[len(available_sizes)//3], 
                                 available_sizes[2*len(available_sizes)//3], available_sizes[-1]]
            else:
                selected_sizes = available_sizes
        
        # Verify sizes exist in data
        available_sizes = sorted(dorr_data['matrix_size'].unique())
        selected_sizes = [size for size in selected_sizes if size in available_sizes][:4]
        
        # Plot positions
        plot_positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
        
        for idx, size in enumerate(selected_sizes[:4]):
            if idx < len(plot_positions):
                row, col = plot_positions[idx]
                ax = axes[row, col]
                
                size_data = dorr_data[dorr_data['matrix_size'] == size]
                
                # Plot each configuration
                for config in sorted(size_data['config_name'].unique()):
                    config_data = size_data[size_data['config_name'] == config].sort_values('mean_condition_number')
                    if not config_data.empty:
                        ax.plot(config_data['mean_condition_number'], 
                               config_data['mean_relative_l2_error_pct'],
                               label=config, color=color_map.get(config, '#cccccc'), 
                               linewidth=2, marker='s', markersize=5, alpha=0.8)
                
                ax.set_xlabel('Condition Number κ(A)', fontsize=11)
                ax.set_ylabel('Relative L2 Error (%)', fontsize=11)
                ax.set_title(f'Dorr {size}×{size}: Condition vs Error', fontweight='bold', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
                ax.set_yscale('log')
                
                if idx == 0:
                    ax.legend(fontsize=7)
    
    # Hide unused subplot
    axes[1, 2].set_visible(False)
    
    plt.suptitle('Condition Number vs Error Analysis\n(FEM + Dorr Matrices)', 
                 fontweight='bold', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    results_dir = Path("results_parametric")
    plot_file = results_dir / "condition_vs_error.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Condition vs error plots saved to: {plot_file}")

def create_success_rate_table(df, color_map):
    """Create visual success rate table figure by configuration"""
    
    print("\n=== CONFIGURATION SUMMARY TABLE ===\n")
    
    # Calculate summary by configuration
    summary_data = []
    
    for config in sorted(df['config_name'].unique()):
        config_data = df[df['config_name'] == config]
        
        # Calculate success rate using new failure tracking
        if 'total_attempted' in config_data.columns and 'num_matrices' in config_data.columns:
            # Use new detailed tracking
            total_attempted = config_data['total_attempted'].sum()
            successful_analyses = config_data['num_matrices'].sum()
            success_rate = (successful_analyses / total_attempted * 100) if total_attempted > 0 else 0
            valid_results = successful_analyses
            total_combinations = total_attempted
        else:
            # Fallback to old method
            total_combinations = len(config_data)
            valid_results = len(config_data[np.isfinite(config_data['mean_relative_l2_error'])])
            success_rate = (valid_results / total_combinations * 100) if total_combinations > 0 else 0
        
        # By matrix type
        fem_data = config_data[config_data['matrix_type'] == 'fem_poisson']
        dorr_data = config_data[config_data['matrix_type'] == 'dorr']
        
        # FEM success rate
        if 'total_attempted' in fem_data.columns and 'num_matrices' in fem_data.columns:
            fem_total = fem_data['total_attempted'].sum()
            fem_valid = fem_data['num_matrices'].sum()
            fem_success = (fem_valid / fem_total * 100) if fem_total > 0 else 0
        else:
            fem_total = len(fem_data)
            fem_valid = len(fem_data[np.isfinite(fem_data['mean_relative_l2_error'])])
            fem_success = (fem_valid / fem_total * 100) if fem_total > 0 else 0
        
        # Dorr success rate
        if 'total_attempted' in dorr_data.columns and 'num_matrices' in dorr_data.columns:
            dorr_total = dorr_data['total_attempted'].sum()
            dorr_valid = dorr_data['num_matrices'].sum()
            dorr_success = (dorr_valid / dorr_total * 100) if dorr_total > 0 else 0
        else:
            dorr_total = len(dorr_data)
            dorr_valid = len(dorr_data[np.isfinite(dorr_data['mean_relative_l2_error'])])
            dorr_success = (dorr_valid / dorr_total * 100) if dorr_total > 0 else 0
        
        # Average error for valid results
        valid_errors = config_data[np.isfinite(config_data['mean_relative_l2_error'])]['mean_relative_l2_error']
        avg_error = valid_errors.mean() * 100 if len(valid_errors) > 0 else np.nan
        
        summary_data.append({
            'Configuration': config,
            # 'Overall Success (%)': f"{success_rate:.1f}%",
            'FEM Success (%)': f"{fem_success:.1f}%",
            'Dorr Success (%)': f"{dorr_success:.1f}%",
            # 'Avg Error (%)': f"{avg_error:.2f}%" if not np.isnan(avg_error) else "N/A",
            'Valid/Total': f"{valid_results}/{total_combinations}"
        })
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Create visual table figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    # Color code configurations using same colors as plots
    config_names = summary_df['Configuration'].tolist()
    
    for i in range(len(summary_df)):
        config_name = config_names[i]
        config_color = color_map.get(config_name, '#cccccc')
        
        # Color the configuration name cell and make it bold
        for j in range(len(summary_df.columns)):
            if j == 0:  # Configuration name column
                table[(i+1, j)].set_facecolor(config_color)
                table[(i+1, j)].set_alpha(0.4)
                table[(i+1, j)].set_text_props(weight='bold')
            else:
                table[(i+1, j)].set_facecolor(config_color)
                table[(i+1, j)].set_alpha(0.1)
    
    # Style header
    for j in range(len(summary_df.columns)):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white')
        table[(0, j)].set_alpha(0.8)
    
    # Add title
    total_combinations = len(df)
    matrix_sizes = sorted(df['matrix_size'].unique())
    plt.title(f'AFMP Configuration Performance Summary\n'
             f'Matrix Sizes: {min(matrix_sizes)} to {max(matrix_sizes)} | '
             f'Total Combinations: {total_combinations} | Matrix Types: FEM & Dorr', 
             fontsize=16, fontweight='bold', pad=30)
    
    plt.tight_layout()
    
    # Save figure
    results_dir = Path("results_parametric")
    table_file = results_dir / "success_rate_table.png"
    plt.savefig(table_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save CSV
    csv_file = results_dir / "success_rate_table.csv"
    summary_df.to_csv(csv_file, index=False)
    
    # Print summary to console
    print("Configuration Performance Summary:")
    print("=" * 95)
    for _, row in summary_df.iterrows():
        print(
              f"{row['FEM Success (%)']:<11} {row['Dorr Success (%)']:<12} "
              f"{row['Valid/Total']:<12}")
            #   f"{row['Configuration']:<20} {row['Overall Success (%)']:<12} ")
    print("=" * 95)
    
    print(f"\n📊 Visual success rate table saved to: {table_file}")
    print(f"📄 CSV data saved to: {csv_file}")

def main():
    """Create all custom plots"""
    
    print("=== CUSTOM PLOT GENERATOR ===")
    print("Creating requested plots from AFPM analysis results...")
    print()
    
    # Load data
    results, config = load_results_and_config()
    if results is None:
        return
    
    df = pd.DataFrame(results)
    color_map = get_config_colors()
    
    print(f"Loaded {len(results)} analysis results")
    print(f"Matrix types: {df['matrix_type'].unique()}")
    print(f"Configurations: {df['config_name'].unique()}")
    print()
    
    # Create all plots
    print("Creating plots...")
    
    # 1. FEM: Size vs Error
    plot_fem_size_vs_error(df, color_map)
    
    # 2. Dorr: Alpha vs Error (4 sizes)
    plot_dorr_alpha_vs_error(df, color_map)
    
    # 3. Condition vs Error (FEM + 4 Dorr)
    plot_condition_vs_error(df, color_map)
    
    # 4. Success rate table
    create_success_rate_table(df, color_map)
    
    print("\n✅ All custom plots created!")
    print("\nGenerated files:")
    print("  📊 fem_size_vs_error.png - FEM matrix size vs error")
    print("  📊 dorr_alpha_vs_error.png - Dorr alpha vs error (4 sizes)")
    print("  📊 condition_vs_error.png - Condition number analysis")
    print("  📊 success_rate_table.png - Visual success rate table")
    print("  📄 success_rate_table.csv - Success rate data")

if __name__ == "__main__":
    main()