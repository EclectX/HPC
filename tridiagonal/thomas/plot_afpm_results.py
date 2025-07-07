#!/usr/bin/env python3
"""
AFPM Results Plotter: Read saved results and create visualizations
Separate plotting functionality for better modularity and debugging
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import pickle
from pathlib import Path
from config_loader import load_config

class AFPMResultsPlotter:
    """Plot AFPM analysis results from saved data"""
    
    def __init__(self, config_file="config.yaml"):
        # Load configuration
        self.config = load_config(config_file)
        self.results_dir = self.config.get_results_dir()
        self.results_file = self.results_dir / "afpm_analysis_results.pkl"
        
    def load_results(self):
        """Load results from pickle file"""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file, 'rb') as f:
            results = pickle.load(f)
        
        print(f"Loaded {len(results)} results from {self.results_file}")
        return results
    
    def create_configuration_table(self):
        """Create configuration table"""
        config_dict = self.config.get_afpm_configurations()
        configurations = {}
        for name, config_data in config_dict.items():
            configurations[name] = config_data['chromosome']
        
        # Create configuration table
        table_data = []
        for name, chrom in configurations.items():
            table_data.append({
                'Configuration': name,
                'Multiplier [0-7]': f"[{', '.join(map(str, chrom[:8]))}]",
                'Multiplier [8]': chrom[8],
                'Chromosome': str(chrom),
                'Description': self.config.get_configuration_description(name)
            })
        
        df_config = pd.DataFrame(table_data)
        
        # Create table visualization
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df_config.values,
                        colLabels=df_config.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Color code configurations using config colors
        config_names = list(configurations.keys())
        
        for i in range(len(df_config)):
            config_name = config_names[i]
            color = self.config.get_configuration_color(config_name)
            for j in range(len(df_config.columns)):
                table[(i+1, j)].set_facecolor(color)
                table[(i+1, j)].set_alpha(0.3)
        
        # Style header
        for j in range(len(df_config.columns)):
            table[(0, j)].set_facecolor('#40466e')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        matrix_sizes = self.config.get_matrix_sizes()
        num_matrices = self.config.get_matrices_per_size()
        plt.title(f'AFPM Configuration Table: {len(configurations)} Selected Configurations\n'
                 f'Matrix sizes: {min(matrix_sizes)} to {max(matrix_sizes)}, Averaged over {num_matrices} matrices per size', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Use subplots_adjust for table layout
        plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05)
        
        config_table_file = self.results_dir / "afpm_config_table.png"
        plt.savefig(config_table_file, dpi=self.config.get_dpi(), bbox_inches='tight')
        plt.close()
        
        print(f"Configuration table saved to: {config_table_file}")
    
    
    def create_logarithmic_plots(self, df):
        """Create logarithmic scale plots"""
        
        # Convert to percentage error
        df = df.copy()
        df['mean_relative_l2_error_pct'] = df['mean_relative_l2_error'] * 100
        df['std_relative_l2_error_pct'] = df['std_relative_l2_error'] * 100
        df['float32_theoretical_bound_pct'] = df['float32_theoretical_bound'] * 100
        
        matrix_types = ['tridiagonal_toeplitz', 'fem_poisson']
        titles = ['Tridiagonal Toeplitz Matrix', 'FEM Poisson Discretization']
        
        # Color map for configurations from config file
        config_names = list(self.config.get_configuration_names())
        color_map = {}
        for name in config_names:
            color_map[name] = self.config.get_configuration_color(name)
        
        # Create Figure: Logarithmic plots (2x2 grid)
        figsize = self.config.get_figure_size()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Top row: Error vs Matrix Size (Log scale)
        for matrix_type, ax, title in zip(matrix_types, [ax1, ax2], titles):
            type_data = df[df['matrix_type'] == matrix_type]
            
            for config in config_names:
                config_data = type_data[type_data['config_name'] == config]
                if not config_data.empty:
                    ax.errorbar(config_data['matrix_size'], config_data['mean_relative_l2_error_pct'],
                               yerr=config_data['std_relative_l2_error_pct'],
                               label=config, color=color_map[config], 
                               linewidth=2, markersize=6, capsize=3, 
                               marker='o', alpha=0.8)
            
            # Add Float32 theoretical bound
            sizes = sorted(type_data['matrix_size'].unique())
            float32_bounds = []
            for size in sizes:
                bound_data = type_data[type_data['matrix_size'] == size]
                if not bound_data.empty:
                    float32_bounds.append(bound_data['float32_theoretical_bound_pct'].iloc[0])
            
            if float32_bounds:
                ax.loglog(sizes, float32_bounds, 'k--', linewidth=3, alpha=0.9, 
                         label='Float32 Theory Bound', zorder=10)
            
            # Add acceptable error threshold lines for different applications
            x_range = [min(sizes), max(sizes)]
            
            # Application tolerance thresholds (in percentage) - dashed, thinner, more opaque
            thresh_1 = ax.loglog(x_range, [0.01, 0.01], 'g--', linewidth=1.5, alpha=1.0, zorder=5, label='Scientific: 0.01%')
            thresh_2 = ax.loglog(x_range, [0.1, 0.1], 'b--', linewidth=1.5, alpha=1.0, zorder=5, label='Engineering: 0.1%')
            thresh_3 = ax.loglog(x_range, [1.0, 1.0], '--', color='orange', linewidth=1.5, alpha=1.0, zorder=5, label='Real-time: 1%')
            thresh_4 = ax.loglog(x_range, [10.0, 10.0], 'r--', linewidth=1.5, alpha=1.0, zorder=5, label='IoT/Embedded: 10%')
            
            ax.set_xlabel('Matrix Size')
            ax.set_ylabel('Relative L2 Error (%)')
            ax.set_title(f'{title}\nError vs Matrix Size (Log Scale)', fontweight='bold')
            
            # Create two separate legends
            # First legend: AFPM configurations (excluding threshold lines)
            handles, labels = ax.get_legend_handles_labels()
            config_handles = [h for h, l in zip(handles, labels) if 'Scientific:' not in l and 'Engineering:' not in l and 'Real-time:' not in l and 'IoT/Embedded:' not in l]
            config_labels = [l for l in labels if 'Scientific:' not in l and 'Engineering:' not in l and 'Real-time:' not in l and 'IoT/Embedded:' not in l]
            
            legend1 = ax.legend(config_handles, config_labels, fontsize=6, loc='lower right', 
                               title='AFPM Configurations', framealpha=0.9)
            ax.add_artist(legend1)  # Keep the first legend
            
            # Create second legend for thresholds only
            threshold_lines = [thresh_1[0], thresh_2[0], thresh_3[0], thresh_4[0]]
            threshold_labels = ['Scientific: 0.01%', 'Engineering: 0.1%', 'Real-time: 1%', 'IoT/Embedded: 10%']
            ax.legend(threshold_lines, threshold_labels, fontsize=6, loc='upper right', 
                     title='Application Thresholds', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        # Bottom row: Error vs Condition Number (Log scale)
        for matrix_type, ax, title in zip(matrix_types, [ax3, ax4], titles):
            type_data = df[df['matrix_type'] == matrix_type]
            
            for config in config_names:
                config_data = type_data[type_data['config_name'] == config]
                if not config_data.empty:
                    ax.errorbar(config_data['mean_condition_number'], config_data['mean_relative_l2_error_pct'],
                               yerr=config_data['std_relative_l2_error_pct'],
                               label=config, color=color_map[config], 
                               linewidth=2, markersize=6, capsize=3, 
                               marker='o', alpha=0.8)
            
            # Add theoretical bound line
            conditions = sorted(type_data['mean_condition_number'].unique())
            if conditions:
                theoretical_line = 3 * 1.19e-7 * np.array(conditions) * 100
                ax.loglog(conditions, theoretical_line, 'k--', linewidth=3, alpha=0.9, 
                         label='Theory: E ∝ κ(A)', zorder=10)
            
            # Add acceptable error threshold lines for different applications
            x_range = [min(conditions), max(conditions)]
            
            # Application tolerance thresholds (in percentage) - dashed, thinner, more opaque
            thresh_1 = ax.loglog(x_range, [0.01, 0.01], 'g--', linewidth=1.5, alpha=1.0, zorder=5, label='Scientific: 0.01%')
            thresh_2 = ax.loglog(x_range, [0.1, 0.1], 'b--', linewidth=1.5, alpha=1.0, zorder=5, label='Engineering: 0.1%')
            thresh_3 = ax.loglog(x_range, [1.0, 1.0], '--', color='orange', linewidth=1.5, alpha=1.0, zorder=5, label='Real-time: 1%')
            thresh_4 = ax.loglog(x_range, [10.0, 10.0], 'r--', linewidth=1.5, alpha=1.0, zorder=5, label='IoT/Embedded: 10%')
            
            ax.set_xlabel('Condition Number κ(A)')
            ax.set_ylabel('Relative L2 Error (%)')
            ax.set_title(f'{title}\nError vs Condition Number (Log Scale)', fontweight='bold')
            
            # Create two separate legends
            # First legend: AFPM configurations (excluding threshold lines)
            handles, labels = ax.get_legend_handles_labels()
            config_handles = [h for h, l in zip(handles, labels) if 'Scientific:' not in l and 'Engineering:' not in l and 'Real-time:' not in l and 'IoT/Embedded:' not in l]
            config_labels = [l for l in labels if 'Scientific:' not in l and 'Engineering:' not in l and 'Real-time:' not in l and 'IoT/Embedded:' not in l]
            
            legend1 = ax.legend(config_handles, config_labels, fontsize=6, loc='lower right', 
                               title='AFPM Configurations', framealpha=0.9)
            ax.add_artist(legend1)  # Keep the first legend
            
            # Create second legend for thresholds only
            threshold_lines = [thresh_1[0], thresh_2[0], thresh_3[0], thresh_4[0]]
            threshold_labels = ['Scientific: 0.01%', 'Engineering: 0.1%', 'Real-time: 1%', 'IoT/Embedded: 10%']
            ax.legend(threshold_lines, threshold_labels, fontsize=6, loc='upper right', 
                     title='Application Thresholds', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Add conditioning regions
            ax.axvspan(1, 1e3, alpha=0.1, color='green')
            ax.axvspan(1e3, 1e6, alpha=0.1, color='yellow')
            ax.axvspan(1e6, 1e12, alpha=0.1, color='orange')
            ax.axvspan(1e12, 1e20, alpha=0.1, color='red')
        
        # Use subplots_adjust instead of tight_layout for better control
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, wspace=0.3, hspace=0.4)
        
        log_plot_file = self.results_dir / "afpm_analysis_logarithmic.png"
        plt.savefig(log_plot_file, dpi=self.config.get_dpi(), bbox_inches='tight')
        plt.close()
        
        print(f"Logarithmic plots saved to: {log_plot_file}")
    
    def create_linear_plots(self, df):
        """Create simplified linear scale plots (optimized for performance)"""
        
        print("Creating simplified linear plots...")
        
        # Filter to smaller matrix sizes only for linear plots (better visualization)
        max_size_linear = 512  # Only show smaller matrices in linear scale
        df_small = df[df['matrix_size'] <= max_size_linear].copy()
        
        if df_small.empty:
            print("No data for linear plots (all matrices too large)")
            return
        
        # Convert to percentage error with reasonable caps
        df_small['mean_relative_l2_error_pct'] = (df_small['mean_relative_l2_error'] * 100).clip(upper=50)
        
        matrix_types = ['tridiagonal_toeplitz', 'fem_poisson']
        
        # Color map for configurations from config file
        config_names = list(self.config.get_configuration_names())
        color_map = {}
        for name in config_names:
            color_map[name] = self.config.get_configuration_color(name)
        
        # Create simpler 1x2 figure (only matrix size plots)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Simple linear plots: Error vs Matrix Size only
        titles = ['Tridiagonal Toeplitz Matrix', 'FEM Poisson Discretization']
        for matrix_type, ax, title in zip(matrix_types, [ax1, ax2], titles):
            type_data = df[df['matrix_type'] == matrix_type]
            
            for config in config_names:
                config_data = type_data[type_data['config_name'] == config]
                if not config_data.empty:
                    ax.errorbar(config_data['matrix_size'], config_data['mean_relative_l2_error_pct'],
                               yerr=config_data['std_relative_l2_error_pct'],
                               label=config, color=color_map[config], 
                               linewidth=2, markersize=6, capsize=3, 
                               marker='o', alpha=0.8)
            
            # Add Float32 theoretical bound
            sizes = sorted(type_data['matrix_size'].unique())
            float32_bounds = []
            for size in sizes:
                bound_data = type_data[type_data['matrix_size'] == size]
                if not bound_data.empty:
                    float32_bounds.append(bound_data['float32_theoretical_bound_pct'].iloc[0])
            
            if float32_bounds:
                ax.plot(sizes, float32_bounds, 'k--', linewidth=3, alpha=0.9, 
                       label='Float32 Theory Bound', zorder=10)
            
            ax.set_xlabel('Matrix Size')
            ax.set_ylabel('Relative L2 Error (%)')
            ax.set_title(f'{title}\nError vs Matrix Size (Linear Scale)', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Bottom row: Error vs Condition Number (Linear scale)
        for matrix_type, ax, title in zip(matrix_types, [ax3, ax4], titles):
            type_data = df[df['matrix_type'] == matrix_type]
            
            for config in config_names:
                config_data = type_data[type_data['config_name'] == config]
                if not config_data.empty:
                    ax.errorbar(config_data['mean_condition_number'], config_data['mean_relative_l2_error_pct'],
                               yerr=config_data['std_relative_l2_error_pct'],
                               label=config, color=color_map[config], 
                               linewidth=2, markersize=6, capsize=3, 
                               marker='o', alpha=0.8)
            
            # Add theoretical bound line
            conditions = sorted(type_data['mean_condition_number'].unique())
            if conditions:
                theoretical_line = 3 * 1.19e-7 * np.array(conditions) * 100
                ax.plot(conditions, theoretical_line, 'k--', linewidth=3, alpha=0.9, 
                       label='Theory: E ∝ κ(A)', zorder=10)
            
            ax.set_xlabel('Condition Number κ(A)')
            ax.set_ylabel('Relative L2 Error (%)')
            ax.set_title(f'{title}\nError vs Condition Number (Linear Scale)', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add conditioning region labels with better positioning
            y_max = ax.get_ylim()[1]
            if y_max > 0:
                ax.text(500, y_max*0.95, 'Well-Conditioned', fontsize=8, 
                       bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
                ax.text(5e4, y_max*0.85, 'Moderately Ill-Conditioned', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
                ax.text(5e8, y_max*0.75, 'Ill-Conditioned', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
        
        # Use subplots_adjust instead of tight_layout for better control
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, wspace=0.3, hspace=0.4)
        
        linear_plot_file = self.results_dir / "afpm_analysis_linear.png"
        plt.savefig(linear_plot_file, dpi=self.config.get_dpi(), bbox_inches='tight')
        plt.close()
        
        print(f"Linear plots saved to: {linear_plot_file}")
    
    def print_summary_statistics(self, df):
        """Print summary statistics"""
        print("\n=== SUMMARY STATISTICS ===")
        for matrix_type in ['tridiagonal_toeplitz', 'fem_poisson']:
            print(f"\n{matrix_type.upper().replace('_', ' ')}:")
            type_data = df[df['matrix_type'] == matrix_type]
            
            print("  Size Range | Condition Numbers | Error Range (%)")
            print("  -----------|-------------------|------------------")
            
            for size in sorted(type_data['matrix_size'].unique()):
                size_data = type_data[type_data['matrix_size'] == size]
                if not size_data.empty:
                    condition_num = size_data['mean_condition_number'].iloc[0]
                    min_error = size_data['mean_relative_l2_error'].min() * 100
                    max_error = size_data['mean_relative_l2_error'].max() * 100
                    
                    print(f"  {size:10d} | {condition_num:12.2e} | {min_error:.3f}% - {max_error:.2f}%")
    
    def plot_all(self):
        """Create all plots from saved results"""
        
        # Load results
        results = self.load_results()
        df = pd.DataFrame(results)
        
        print(f"Data summary:")
        print(f"- Configurations: {df['config_name'].unique()}")
        print(f"- Matrix types: {df['matrix_type'].unique()}")
        print(f"- Matrix sizes: {sorted(df['matrix_size'].unique())}")
        print()
        
        # Create all plots
        print("Creating configuration table...")
        self.create_configuration_table()
        
        print("Creating logarithmic plots...")
        self.create_logarithmic_plots(df)
        
        # Check if linear plots are enabled in config
        plot_types = self.config.get_plot_types()
        if "linear" in plot_types:
            print("Creating linear plots...")
            self.create_linear_plots(df)
        else:
            print("Linear plots disabled in configuration")
            print("To create simple linear plots, run: python3 simple_linear_plot.py")
        
        # Print summary statistics
        self.print_summary_statistics(df)
        
        print(f"\nPlots created successfully!")
        print(f"- Configuration table: {self.results_dir}/afpm_config_table.png")
        print(f"- Logarithmic plots: {self.results_dir}/afpm_analysis_logarithmic.png")
        
        plot_types = self.config.get_plot_types()
        if "linear" in plot_types:
            print(f"- Linear plots: {self.results_dir}/afpm_analysis_linear.png")
        else:
            print("- Linear plots: disabled (run simple_linear_plot.py for fast linear plots)")

def main():
    """Main plotting function"""
    
    # Load configuration to get correct results directory
    config = load_config()
    results_dir = config.get_results_dir()
    
    # Check if results directory exists
    if not results_dir.exists():
        print(f"Results directory {results_dir} not found!")
        print("Please run enhanced_afpm_analysis.py first to generate results.")
        return
    
    # Create plotter and generate all plots
    plotter = AFPMResultsPlotter()
    
    try:
        plotter.plot_all()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run enhanced_afpm_analysis.py first to generate results.")
    except Exception as e:
        print(f"Plotting error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()