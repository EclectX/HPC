#!/usr/bin/env python3

import numpy as np
import pickle
from pathlib import Path
import time
import shutil
from config_loader import load_config

class MatrixGenerator:
    def __init__(self, config_file="config.yaml"):
        # Load configuration
        self.config = load_config(config_file)
        self.output_dir = self.config.get_output_dir()
        
        # Remove existing directory if it exists
        if self.output_dir.exists():
            print(f"Removing existing matrix directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        # Create fresh directory structure
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each matrix type
        self.type_dirs = {}
        for matrix_type in self.config.get_matrix_types():
            type_dir = self.output_dir / matrix_type
            type_dir.mkdir(exist_ok=True)
            self.type_dirs[matrix_type] = type_dir
        
        print(f"Created fresh matrix directory: {self.output_dir}")
        print(f"Matrix types: {', '.join(self.config.get_matrix_types())}")
        
    def generate_tridiagonal_toeplitz(self, n, seed=None):
        if seed is None:
            seed = self.config.get_random_seed()
        np.random.seed(seed)
        
        a = np.full(n, -1.0, dtype=np.float64)
        b = np.full(n, 2.0, dtype=np.float64)
        c = np.full(n, -1.0, dtype=np.float64)
        a[0] = 0.0
        c[n-1] = 0.0
        
        d = np.random.uniform(-1, 1, n).astype(np.float64)
        
        return a, b, c, d
    
    def generate_fem_poisson(self, n, seed=None):
        if seed is None:
            seed = self.config.get_random_seed()
        np.random.seed(seed)
        
        h = 1.0 / (n + 1)
        a = np.full(n, -1.0 / (h*h), dtype=np.float64)
        b = np.full(n, 2.0 / (h*h), dtype=np.float64)
        c = np.full(n, -1.0 / (h*h), dtype=np.float64)
        a[0] = 0.0
        c[n-1] = 0.0
        
        d = np.random.uniform(-1, 1, n).astype(np.float64)
        
        return a, b, c, d
    
    def compute_condition_number(self, n, a, b, c):
        if n <= 2048:
            A_full = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                if i > 0:
                    A_full[i, i-1] = a[i]
                A_full[i, i] = b[i]
                if i < n-1:
                    A_full[i, i+1] = c[i]
            
            try:
                condition_number = np.linalg.cond(A_full)
            except:
                condition_number = np.inf
        else:
            if np.allclose(b, 2.0) and np.allclose(a[1:], -1.0):
                condition_number = (n**2) / (np.pi**2)
            else:
                condition_number = ((n+1)**2) / (np.pi**2)
                
        return condition_number
    
    def save_matrix(self, matrix_type, size, matrix_id, a, b, c, d, condition_number):
        save_dir = self.type_dirs[matrix_type]
        
        filename = f"matrix_n{size:05d}_id{matrix_id:02d}.pkl"
        filepath = save_dir / filename
        
        matrix_data = {
            'matrix_type': matrix_type,
            'size': size,
            'matrix_id': matrix_id,
            'a': a,
            'b': b, 
            'c': c,
            'd': d,
            'condition_number': condition_number,
            'generation_time': time.time()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(matrix_data, f)
        
        return filepath
    
    def generate_all_matrices(self):
        print("=== MATRIX GENERATION FOR AFPM ANALYSIS ===")
        print("Using configuration from config.yaml")
        print()
        
        # Get parameters from configuration
        matrix_sizes = self.config.get_matrix_sizes()
        matrix_types = self.config.get_matrix_types()
        num_matrices_per_size = self.config.get_matrices_per_size()
        
        print(f"Matrix sizes: {len(matrix_sizes)} sizes from {min(matrix_sizes)} to {max(matrix_sizes)}")
        print(f"Matrix types: {', '.join(matrix_types)}")
        print(f"Matrices per size: {num_matrices_per_size}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        total_matrices = len(matrix_sizes) * len(matrix_types) * num_matrices_per_size
        generated_count = 0
        
        summary_data = {
            'generation_info': {
                'total_matrices': total_matrices,
                'matrix_sizes': matrix_sizes,
                'matrix_types': matrix_types,
                'num_matrices_per_size': num_matrices_per_size,
                'generation_start': time.time()
            },
            'matrices': []
        }
        
        for matrix_type in matrix_types:
            print(f"\n--- Generating {matrix_type} matrices ---")
            
            for size in matrix_sizes:
                print(f"  Size {size}:", end=" ")
                
                for matrix_id in range(num_matrices_per_size):
                    try:
                        seed = self.config.get_random_seed() + size * 100 + matrix_id
                        
                        if matrix_type == 'tridiagonal_toeplitz':
                            a, b, c, d = self.generate_tridiagonal_toeplitz(size, seed)
                        elif matrix_type == 'fem_poisson':
                            a, b, c, d = self.generate_fem_poisson(size, seed)
                        else:
                            raise ValueError(f"Unknown matrix type: {matrix_type}")
                        
                        condition_number = self.compute_condition_number(size, a, b, c)
                        
                        filepath = self.save_matrix(matrix_type, size, matrix_id, a, b, c, d, condition_number)
                        
                        summary_data['matrices'].append({
                            'matrix_type': matrix_type,
                            'size': size,
                            'matrix_id': matrix_id,
                            'condition_number': condition_number,
                            'filepath': str(filepath.relative_to(self.output_dir))
                        })
                        
                        generated_count += 1
                        
                        if matrix_id == 0:
                            print(f"κ={condition_number:.2e}", end=" ")
                        
                    except Exception as e:
                        print(f"Error: {e}")
                
                print(f"✓ ({generated_count}/{total_matrices})")
        
        summary_data['generation_info']['generation_end'] = time.time()
        summary_data['generation_info']['total_generated'] = generated_count
        
        summary_file = self.output_dir / "matrix_summary.pkl"
        with open(summary_file, 'wb') as f:
            pickle.dump(summary_data, f)
        
        print(f"\n=== GENERATION COMPLETE ===")
        print(f"Total matrices generated: {generated_count}")
        
        return summary_data

def main():
    """Main function for matrix generation"""
    try:
        generator = MatrixGenerator()
        summary = generator.generate_all_matrices()
        
        print(f"\nMatrix generation complete!")
        print(f"Matrices saved in: {generator.output_dir}")
        
    except Exception as e:
        print(f"Error during matrix generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()