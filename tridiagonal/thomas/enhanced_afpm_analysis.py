#!/usr/bin/env python3
"""
Enhanced AFPM Analysis: Reads matrices from directory, averages over 10 matrices per size
Includes condition number analysis and saves results to directory
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import pickle
import struct
import sys
from pathlib import Path
import time
sys.path.append('.')

from AFPM import AFPM, genes_list
from config_loader import load_config

class EnhancedAFPMAnalysis:
    """Enhanced AFPM analysis reading matrices from directory"""
    
    def __init__(self, config_file="config.yaml"):
        # Load configuration
        self.config = load_config(config_file)
        
        self.matrix_dir = self.config.get_output_dir()
        self.results_dir = self.config.get_results_dir()
        self.results_dir.mkdir(exist_ok=True)
        
        # Create matrix type directories dynamically
        self.type_dirs = {}
        for matrix_type in self.config.get_matrix_types():
            self.type_dirs[matrix_type] = self.matrix_dir / matrix_type
        
        self.results = []
        
    def float_to_32bit_binary(self, f):
        """Convert float to 32-bit IEEE 754 binary representation (MSB at index 31)"""
        packed = struct.pack('>f', f)
        bits = struct.unpack('>I', packed)[0]
        binary = [0] * 32
        
        # Fill array with MSB at index 31, LSB at index 0
        for i in range(32):
            binary[31-i] = (bits >> i) & 1
        
        return binary
    
    def binary_to_float(self, binary):
        """Convert 32-bit binary array (MSB at index 31) back to float"""
        bits = 0
        for i in range(32):
            if binary[31-i]:  # MSB is at index 31
                bits |= (1 << i)
        packed = struct.pack('>I', bits)
        return struct.unpack('>f', packed)[0]
    
    def afpm_multiply(self, a, b, chromosome):
        """Perform approximate multiplication using AFPM"""
        # Check if all chromosome values are -1 (exact multiplication)
        if all(c == -1 for c in chromosome):
            return float(a) * float(b)
        
        a_binary = self.float_to_32bit_binary(float(a))
        b_binary = self.float_to_32bit_binary(float(b))
        result_binary = AFPM(a_binary, b_binary, chromosome)
        
        if len(result_binary) != 32:
            raise ValueError(f"AFPM returned invalid result length: {len(result_binary)} (expected 32). "
                           f"Input: a={a}, b={b}, chromosome={chromosome}")
        
        return self.binary_to_float(result_binary)
    
    def thomas_solve_afpm(self, n, a, b, c, d, chromosome):
        """Solve tridiagonal system using Thomas algorithm with AFPM"""
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32) 
        c = np.array(c, dtype=np.float32)
        d = np.array(d, dtype=np.float32)
        x = np.zeros(n, dtype=np.float32)
        
        # Forward elimination
        for i in range(1, n):
            if abs(b[i-1]) < 1e-10:
                raise ValueError(f"Zero pivot at position {i-1}")
            
            m = self.afpm_multiply(a[i], 1.0/b[i-1], chromosome)
            b[i] = b[i] - self.afpm_multiply(m, c[i-1], chromosome)
            d[i] = d[i] - self.afpm_multiply(m, d[i-1], chromosome)
        
        # Back substitution
        if abs(b[n-1]) < 1e-10:
            raise ValueError(f"Zero pivot at position {n-1}")
        
        x[n-1] = d[n-1] / b[n-1]
        
        for i in range(n-2, -1, -1):
            if abs(b[i]) < 1e-10:
                raise ValueError(f"Zero pivot at position {i}")
            x[i] = (d[i] - self.afpm_multiply(c[i], x[i+1], chromosome)) / b[i]
        
        return x
    
    def thomas_solve_double_precision(self, n, a, b, c, d):
        """Solve tridiagonal system using double precision as ground truth"""
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64) 
        c = np.array(c, dtype=np.float64)
        d = np.array(d, dtype=np.float64)
        x = np.zeros(n, dtype=np.float64)
        
        # Forward elimination
        for i in range(1, n):
            if abs(b[i-1]) < 1e-15:
                raise ValueError(f"Zero pivot at position {i-1}")
            
            m = a[i] / b[i-1]
            b[i] = b[i] - m * c[i-1]
            d[i] = d[i] - m * d[i-1]
        
        # Back substitution
        if abs(b[n-1]) < 1e-15:
            raise ValueError(f"Zero pivot at position {n-1}")
        
        x[n-1] = d[n-1] / b[n-1]
        
        for i in range(n-2, -1, -1):
            if abs(b[i]) < 1e-15:
                raise ValueError(f"Zero pivot at position {i}")
            x[i] = (d[i] - c[i] * x[i+1]) / b[i]
        
        return x
    
    def load_matrix(self, filepath):
        """Load matrix from pickle file"""
        with open(filepath, 'rb') as f:
            matrix_data = pickle.load(f)
        return matrix_data
    
    def compute_float32_theoretical_bound(self, condition_number):
        """Compute theoretical bound for exact 32-bit float arithmetic"""
        epsilon_float32 = self.config.get_float32_epsilon()
        C_thomas = self.config.get_thomas_constant()
        float32_bound = C_thomas * epsilon_float32 * condition_number
        return float32_bound
    
    def run_comprehensive_analysis(self):
        """Run analysis reading matrices from directory"""
        
        print("=== ENHANCED AFPM ANALYSIS: READING FROM DIRECTORY ===")
        print("Using configuration from config.yaml")
        print(f"Matrix directory: {self.matrix_dir}")
        print(f"Results directory: {self.results_dir}")
        print()
        
        # Get configurations from config file
        config_dict = self.config.get_afpm_configurations()
        configurations = {}
        for name, config_data in config_dict.items():
            configurations[name] = config_data['chromosome']
        
        matrix_types = self.config.get_matrix_types()
        results = []
        
        print("Configurations:")
        for name, chrom in configurations.items():
            print(f"  {name}: {chrom}")
        print()
        
        for matrix_type in matrix_types:
            print(f"\n--- Processing {matrix_type} ---")
            
            matrix_dir = self.type_dirs[matrix_type]
            
            # Get all matrix files for this type
            matrix_files = sorted(list(matrix_dir.glob("matrix_n*_id*.pkl")))
            
            # Group by size
            size_groups = {}
            for filepath in matrix_files:
                matrix_data = self.load_matrix(filepath)
                size = matrix_data['size']
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(matrix_data)
            
            print(f"  Found matrices for sizes: {sorted(size_groups.keys())}")
            
            for config_name, chromosome in configurations.items():
                print(f"    Processing {config_name}...")
                
                for size in sorted(size_groups.keys()):
                    size_matrices = size_groups[size]
                    
                    # Collect errors from all 10 matrices for this size
                    size_errors = []
                    size_conditions = []
                    
                    for matrix_data in size_matrices:
                        try:
                            a = matrix_data['a']
                            b = matrix_data['b'] 
                            c = matrix_data['c']
                            d = matrix_data['d']
                            condition_number = matrix_data['condition_number']
                            
                            # Solve with double precision (ground truth)
                            x_double = self.thomas_solve_double_precision(size, a.copy(), b.copy(), c.copy(), d.copy())
                            
                            # Solve with AFPM
                            x_afpm = self.thomas_solve_afpm(size, a.copy(), b.copy(), c.copy(), d.copy(), chromosome)
                            
                            # Compute Relative L2 Error vs double precision
                            relative_l2_error = np.linalg.norm(x_afpm - x_double) / np.linalg.norm(x_double)
                            
                            size_errors.append(relative_l2_error)
                            size_conditions.append(condition_number)
                            
                        except Exception as e:
                            print(f"      Error in {config_name}, size {size}, matrix {matrix_data['matrix_id']}: {e}")
                    
                    if size_errors:
                        # Compute statistics over the 10 matrices
                        mean_error = np.mean(size_errors)
                        std_error = np.std(size_errors)
                        min_error = np.min(size_errors)
                        max_error = np.max(size_errors)
                        mean_condition = np.mean(size_conditions)
                        
                        # Compute Float32 theoretical bound
                        float32_theoretical_bound = self.compute_float32_theoretical_bound(mean_condition)
                        
                        # Store results
                        result = {
                            'config_name': config_name,
                            'chromosome': chromosome.copy(),
                            'matrix_type': matrix_type,
                            'matrix_size': size,
                            'num_matrices': len(size_errors),
                            'mean_relative_l2_error': mean_error,
                            'std_relative_l2_error': std_error,
                            'min_relative_l2_error': min_error,
                            'max_relative_l2_error': max_error,
                            'mean_condition_number': mean_condition,
                            'float32_theoretical_bound': float32_theoretical_bound
                        }
                        results.append(result)
        
        self.results = results
        
        # Save results
        results_file = self.results_dir / "afpm_analysis_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to: {results_file}")
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
        
        plt.tight_layout()
        config_table_file = self.results_dir / "afpm_config_table.png"
        plt.savefig(config_table_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Configuration table saved to: {config_table_file}")
        return df_config
    
    
    def save_results_only(self):
        """Save results without creating plots"""
        if not self.results:
            print("No results available to save.")
            return
        
        # Save results to pickle file
        results_file = self.results_dir / "afpm_analysis_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Results saved to: {results_file}")
        print(f"Total data points: {len(self.results)}")
        print("\nTo create plots, run: python3 plot_afpm_results.py")
    
    def print_summary_statistics(self, df):
        """Print summary statistics"""
        print("\n=== SUMMARY STATISTICS ===")
        for matrix_type in ['tridiagonal_toeplitz', 'fem_poisson']:
            print(f"\n{matrix_type.upper().replace('_', ' ')}:")
            type_data = df[df['matrix_type'] == matrix_type]
            
            print("  Size Range | Condition Numbers | Error Range")
            print("  -----------|-------------------|-------------")
            
            for size in sorted(type_data['matrix_size'].unique()):
                size_data = type_data[type_data['matrix_size'] == size]
                if not size_data.empty:
                    condition_num = size_data['mean_condition_number'].iloc[0]
                    min_error = size_data['mean_relative_l2_error'].min()
                    max_error = size_data['mean_relative_l2_error'].max()
                    
                    print(f"  {size:10d} | {condition_num:12.2e} | {min_error:.10We} - {max_error:.2e}")

def main():
    """Run enhanced analysis"""
    try:
        analysis = EnhancedAFPMAnalysis()
        
        # Check if matrix directory exists
        if not analysis.matrix_dir.exists():
            print(f"Matrix directory {analysis.matrix_dir} not found!")
            print("Please run matrix_generator.py first to generate test matrices.")
            return
        
        # Create configuration table
        print("Creating configuration table...")
        analysis.create_configuration_table()
        
        # Run comprehensive analysis
        print("\nRunning comprehensive analysis...")
        results = analysis.run_comprehensive_analysis()
        
        # Save results (plotting moved to separate script)
        print("\nSaving results...")
        analysis.save_results_only()
        
        print(f"\nAnalysis complete!")
        print(f"- Results data: {analysis.results_dir}/afpm_analysis_results.pkl")
        print(f"- Total results: {len(results)} data points")
        print(f"\nTo create plots:")
        print(f"  python3 plot_afpm_results.py")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()