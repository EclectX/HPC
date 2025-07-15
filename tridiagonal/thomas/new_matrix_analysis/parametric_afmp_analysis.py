#!/usr/bin/env python3
"""
Parametric AFPM Analysis: FEM 1D Poisson and Dorr matrices with parameter sweeps
Analyzes AFPM performance across different matrix types and parameters
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

class ParametricAFPMAnalysis:
    """AFPM analysis for parametric matrices (FEM and Dorr)"""
    
    def __init__(self, config_file="config.yaml"):
        # Load configuration
        self.config = load_config(config_file)
        
        # Use Path directly since config may not have all sections defined
        try:
            # Try to get from config first
            self.matrix_dir = Path(self.config.config.get('matrix_generation', {}).get('output_dir', 'test_matrices_parametric'))
            self.results_dir = Path(self.config.config.get('afmp_analysis', {}).get('results_dir', 'results_parametric'))
        except:
            # Fallback to default values
            self.matrix_dir = Path('test_matrices_parametric')
            self.results_dir = Path('results_parametric')
        
        self.results_dir.mkdir(exist_ok=True)
        
        # Create matrix type directories dynamically
        self.type_dirs = {
            'fem_poisson': self.matrix_dir / 'fem_poisson',
            'dorr': self.matrix_dir / 'dorr'
        }
        
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
        try:
            epsilon_float32 = self.config.get_float32_epsilon()
            C_thomas = self.config.get_thomas_constant()
        except:
            # Fallback values
            epsilon_float32 = 1.19e-7
            C_thomas = 3.0
        float32_bound = C_thomas * epsilon_float32 * condition_number
        return float32_bound
    
    def run_parametric_analysis(self):
        """Run analysis on parametric matrices"""
        
        print("=== PARAMETRIC AFPM ANALYSIS ===")
        print("Analyzing FEM 1D Poisson and Dorr matrices")
        print(f"Matrix directory: {self.matrix_dir}")
        print(f"Results directory: {self.results_dir}")
        print()
        
        # Get configurations from config file
        try:
            config_dict = self.config.get_afpm_configurations()
            configurations = {}
            for name, config_data in config_dict.items():
                configurations[name] = config_data['chromosome']
        except:
            # Fallback configurations
            configurations = {
                "Exact": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                "Very Low Approx": [20, 15, 10, 5, 0, 0, 0, 0, 0],
                "Low Approximation": [40, 30, 20, 10, 5, 0, 0, 0, 0],
                "Medium Approximation": [60, 50, 40, 30, 20, 10, 5, 0, 0],
                "High Approximation": [80, 70, 60, 50, 40, 30, 20, 10, 0],
                "Worst (chromo8=0)": [99, 99, 99, 99, 99, 99, 99, 99, 0]
            }
        
        matrix_types = ['fem_poisson', 'dorr']
        results = []
        
        print("Configurations:")
        for name, chrom in configurations.items():
            print(f"  {name}: {chrom}")
        print()
        
        for matrix_type in matrix_types:
            print(f"\n--- Processing {matrix_type} ---")
            
            matrix_dir = self.type_dirs[matrix_type]
            
            # Create expected combinations for this matrix type
            expected_groups = {}
            if matrix_type == 'fem_poisson':
                # Expected FEM sizes
                expected_sizes = [4, 8, 16, 32, 64, 128]
                for size in expected_sizes:
                    group_key = (size,)
                    expected_groups[group_key] = []
            elif matrix_type == 'dorr':
                # Expected Dorr combinations  
                expected_sizes = [4, 8, 16, 32, 64, 128]
                expected_alphas = [0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06]
                for size in expected_sizes:
                    for alpha in expected_alphas:
                        group_key = (size, alpha)
                        expected_groups[group_key] = []
            
            # Load actual matrix files if directory exists
            matrix_groups = expected_groups.copy()
            if matrix_dir.exists():
                matrix_files = sorted(list(matrix_dir.glob("matrix_*.pkl")))
                print(f"  Found {len(matrix_files)} matrix files in {matrix_dir}")
                
                for filepath in matrix_files:
                    try:
                        matrix_data = self.load_matrix(filepath)
                        size = matrix_data['size']
                        
                        # Create grouping key based on matrix type
                        if matrix_type == 'fem_poisson':
                            group_key = (size,)
                        elif matrix_type == 'dorr':
                            alpha = matrix_data['parameters']['alpha']
                            group_key = (size, alpha)
                        
                        if group_key in matrix_groups:
                            matrix_groups[group_key].append(matrix_data)
                        else:
                            print(f"    WARNING: Unexpected combination {group_key} found")
                            matrix_groups[group_key] = [matrix_data]
                            
                    except Exception as e:
                        print(f"    ERROR: Failed to load {filepath.name}: {e}")
            else:
                print(f"  Directory {matrix_dir} not found!")
            
            # Count loaded matrices
            total_loaded = sum(len(matrices) for matrices in matrix_groups.values())
            total_expected = len(expected_groups)
            print(f"  Expected {total_expected} parameter groups, loaded matrices for {len([k for k, v in matrix_groups.items() if v])} groups")
            print(f"  Total matrices loaded: {total_loaded}")
            
            for config_name, chromosome in configurations.items():
                print(f"    Processing {config_name}...")
                
                for group_key in sorted(matrix_groups.keys()):
                    group_matrices = matrix_groups[group_key]
                    
                    # Collect errors from all matrices in this group
                    group_errors = []
                    group_conditions = []
                    group_failures = 0
                    
                    # Check if we have any matrices for this group
                    if not group_matrices:
                        # No matrix file exists for this combination
                        print(f"      No matrix file for {config_name}, group {group_key}")
                        group_failures = 1  # Count as one failure
                        
                        # Create entry for missing matrix file
                        result = {
                            'config_name': config_name,
                            'chromosome': chromosome.copy(),
                            'matrix_type': matrix_type,
                            'group_key': group_key,
                            'num_matrices': 0,
                            'num_failures': 1,
                            'total_attempted': 1,
                            'success_rate': 0.0,
                            'mean_relative_l2_error': np.nan,
                            'std_relative_l2_error': np.nan,
                            'min_relative_l2_error': np.nan,
                            'max_relative_l2_error': np.nan,
                            'mean_condition_number': np.nan,
                            'float32_theoretical_bound': np.nan
                        }
                        
                        # Add matrix-type specific parameters
                        if matrix_type == 'fem_poisson':
                            result['matrix_size'] = group_key[0]
                        elif matrix_type == 'dorr':
                            result['matrix_size'] = group_key[0]
                            result['alpha'] = group_key[1]
                        
                        results.append(result)
                        continue
                    
                    for matrix_data in group_matrices:
                        try:
                            size = matrix_data['size']
                            a = matrix_data['a']
                            b = matrix_data['b'] 
                            c = matrix_data['c']
                            d = matrix_data['d']
                            condition_number = matrix_data['condition_number']
                            
                            # Skip matrices with very high condition numbers to avoid numerical issues
                            if condition_number > 1e12:
                                continue
                            
                            # Solve with double precision (ground truth)
                            x_double = self.thomas_solve_double_precision(size, a.copy(), b.copy(), c.copy(), d.copy())
                            
                            # Check if solution is valid
                            if np.any(np.isnan(x_double)) or np.any(np.isinf(x_double)):
                                continue
                            
                            # Solve with AFPM
                            x_afpm = self.thomas_solve_afpm(size, a.copy(), b.copy(), c.copy(), d.copy(), chromosome)
                            
                            # Check for NaN/Inf in solutions
                            if (np.any(np.isnan(x_double)) or np.any(np.isinf(x_double)) or
                                np.any(np.isnan(x_afpm)) or np.any(np.isinf(x_afpm))):
                                print(f"      WARNING: Non-finite solution for {config_name}, group {group_key}")
                                group_failures += 1
                                continue
                            
                            # Compute Relative L2 Error vs double precision
                            x_double_norm = np.linalg.norm(x_double)
                            if x_double_norm < 1e-15:  # Avoid division by very small numbers
                                print(f"      WARNING: Zero exact solution norm for {config_name}, group {group_key}")
                                group_failures += 1
                                continue
                            
                            relative_l2_error = np.linalg.norm(x_afpm - x_double) / x_double_norm
                            
                            # Check for non-finite error
                            if np.isnan(relative_l2_error) or np.isinf(relative_l2_error):
                                print(f"      WARNING: Non-finite error for {config_name}, group {group_key}")
                                group_failures += 1
                                continue
                            
                            group_errors.append(relative_l2_error)
                            group_conditions.append(condition_number)
                            
                        except Exception as e:
                            print(f"      Error in {config_name}, group {group_key}, matrix {matrix_data['matrix_id']}: {e}")
                            group_failures += 1
                    
                    if group_errors:
                        # Compute statistics over the group
                        mean_error = np.mean(group_errors)
                        std_error = np.std(group_errors)
                        min_error = np.min(group_errors)
                        max_error = np.max(group_errors)
                        mean_condition = np.mean(group_conditions)
                        
                        # Compute Float32 theoretical bound
                        float32_theoretical_bound = self.compute_float32_theoretical_bound(mean_condition)
                        
                        # Create result entry for successful analyses
                        result = {
                            'config_name': config_name,
                            'chromosome': chromosome.copy(),
                            'matrix_type': matrix_type,
                            'group_key': group_key,
                            'num_matrices': len(group_errors),
                            'num_failures': group_failures,
                            'total_attempted': len(group_matrices),
                            'success_rate': len(group_errors) / len(group_matrices),
                            'mean_relative_l2_error': mean_error,
                            'std_relative_l2_error': std_error,
                            'min_relative_l2_error': min_error,
                            'max_relative_l2_error': max_error,
                            'mean_condition_number': mean_condition,
                            'float32_theoretical_bound': float32_theoretical_bound
                        }
                        
                        # Add matrix-type specific parameters
                        if matrix_type == 'fem_poisson':
                            result['matrix_size'] = group_key[0]
                        elif matrix_type == 'dorr':
                            result['matrix_size'] = group_key[0]
                            result['alpha'] = group_key[1]
                        
                        results.append(result)
                    
                    else:
                        # This group had matrices but all processing failed
                        # Create entry for completely failed group
                        result = {
                            'config_name': config_name,
                            'chromosome': chromosome.copy(),
                            'matrix_type': matrix_type,
                            'group_key': group_key,
                            'num_matrices': 0,
                            'num_failures': len(group_matrices),
                            'total_attempted': len(group_matrices),
                            'success_rate': 0.0,
                            'mean_relative_l2_error': np.nan,
                            'std_relative_l2_error': np.nan,
                            'min_relative_l2_error': np.nan,
                            'max_relative_l2_error': np.nan,
                            'mean_condition_number': np.nan,
                            'float32_theoretical_bound': np.nan
                        }
                        
                        # Add matrix-type specific parameters
                        if matrix_type == 'fem_poisson':
                            result['matrix_size'] = group_key[0]
                        elif matrix_type == 'dorr':
                            result['matrix_size'] = group_key[0]
                            result['alpha'] = group_key[1]
                        
                        results.append(result)
                        print(f"      WARNING: All {len(group_matrices)} matrices failed AFMP processing for {config_name}, group {group_key}")
        
        self.results = results
        
        # Save results
        results_file = self.results_dir / "parametric_afpm_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Total results: {len(results)} data points")
        return results
    
    def create_configuration_table(self):
        """Create configuration table"""
        try:
            config_dict = self.config.get_afpm_configurations()
            configurations = {}
            for name, config_data in config_dict.items():
                configurations[name] = config_data['chromosome']
        except:
            # Fallback configurations
            configurations = {
                "Exact": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                "Very Low Approx": [20, 15, 10, 5, 0, 0, 0, 0, 0],
                "Low Approximation": [40, 30, 20, 10, 5, 0, 0, 0, 0],
                "Medium Approximation": [60, 50, 40, 30, 20, 10, 5, 0, 0],
                "High Approximation": [80, 70, 60, 50, 40, 30, 20, 10, 0],
                "Worst (chromo8=0)": [99, 99, 99, 99, 99, 99, 99, 99, 0]
            }
            config_dict = {name: {"chromosome": chrom, "description": f"Configuration {name}", "color": "#cccccc"} 
                          for name, chrom in configurations.items()}
        
        # Create configuration table
        table_data = []
        for name, chrom in configurations.items():
            table_data.append({
                'Configuration': name,
                'Multiplier [0-7]': f"[{', '.join(map(str, chrom[:8]))}]",
                'Multiplier [8]': chrom[8],
                'Chromosome': str(chrom),
                'Description': config_dict[name].get('description', 'No description')
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
        
        # Color code configurations
        config_names = list(configurations.keys())
        
        for i in range(len(df_config)):
            config_name = config_names[i]
            color = config_dict[config_name].get('color', '#cccccc')
            for j in range(len(df_config.columns)):
                table[(i+1, j)].set_facecolor(color)
                table[(i+1, j)].set_alpha(0.3)
        
        # Style header
        for j in range(len(df_config.columns)):
            table[(0, j)].set_facecolor('#40466e')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        plt.title(f'AFPM Configuration Table: {len(configurations)} Configurations\n'
                 f'Parametric Analysis: FEM 1D Poisson & Dorr Matrices', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        config_table_file = self.results_dir / "parametric_config_table.png"
        plt.savefig(config_table_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Configuration table saved to: {config_table_file}")
        return df_config

def main():
    """Run parametric analysis"""
    try:
        analysis = ParametricAFPMAnalysis()
        
        # Check if matrix directory exists
        if not analysis.matrix_dir.exists():
            print(f"Matrix directory {analysis.matrix_dir} not found!")
            print("Please run matrix_generators.py first to generate parametric matrices.")
            return
        
        # Create configuration table
        print("Creating configuration table...")
        analysis.create_configuration_table()
        
        # Run parametric analysis
        print("\nRunning parametric analysis...")
        results = analysis.run_parametric_analysis()
        
        print(f"\nAnalysis complete!")
        print(f"- Results data: {analysis.results_dir}/parametric_afpm_results.pkl")
        print(f"- Total results: {len(results)} data points")
        print(f"\nTo create plots:")
        print(f"  python3 plot_parametric_results.py")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()