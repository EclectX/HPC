#!/usr/bin/env python3
"""
Matrix generators for FEM 1D Poisson and Dorr matrices
Supports parameter sweeps for comprehensive AFPM analysis
"""

import numpy as np
import pickle
from pathlib import Path
import sys
from config_loader import load_config

def fem_1d_poisson(n):
    """
    Assemble the FEM system for -u'' = 1 on (0,1) with u(0)=u(1)=0
    using piecewise linear elements on an equispaced mesh.

    Parameters:
    - n: number of elements (=> n+1 nodes, n-1 interior nodes)

    Returns:
    - A: (n-1)x(n-1) stiffness matrix
    - b: (n-1) load vector
    - h: mesh size
    """
    # Mesh size
    h = 1.0 / n

    # Number of interior nodes
    N = n - 1

    # Stiffness matrix: tridiagonal with 2 on diagonal and -1 on off-diagonals
    A = (1.0 / h) * (2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1))

    # right-hand side vector: f = 1, piecewise linear elements
    # For uniform f=1 and uniform mesh, each interior node gets 2*h/2 = h
    b = np.ones(N) * h

    return A, b, h

def dorr(alpha: float, n: int) -> np.ndarray:
    """
    Generate the Dorr matrix of order n with parameter alpha.

    Parameters
    ----------
    alpha : float
        Parameter controlling the strength of the perturbation. Smaller alpha leads to more ill-conditioning.
    n : int
        Size of the matrix.

    Returns
    -------
    A : ndarray of shape (n, n)
        The Dorr matrix.
    
    Reference
    ---------
    Fred Dorr, "An example of ill-conditioning in the numerical solution of singular perturbation problems",
    Mathematics of Computation, 25(114), 1971, pp. 271–283.
    """
    A = np.zeros((n, n), dtype=float)
    np1 = n + 1

    for i in range(n):
        row_idx = i + 1  # For 1-based indexing in formulas

        if row_idx <= (np1 // 2):
            # Upper half
            if i > 0:
                A[i, i - 1] = -alpha * np1**2
            A[i, i] = 2 * alpha * np1**2 + 0.5 * np1 - row_idx
            if i < n - 1:
                A[i, i + 1] = -alpha * np1**2 - 0.5 * np1 + row_idx
        else:
            # Lower half
            if i > 0:
                A[i, i - 1] = -alpha * np1**2 + 0.5 * np1 - row_idx
            A[i, i] = 2 * alpha * np1**2 - 0.5 * np1 + row_idx
            if i < n - 1:
                A[i, i + 1] = -alpha * np1**2

    return A

def extract_tridiagonal_thomas_format(A, rhs=None):
    """
    Extract tridiagonal components from full matrix for Thomas algorithm
    
    Parameters:
    - A: full matrix (n x n)
    - rhs: right-hand side vector (optional, will create ones if not provided)
    
    Returns:
    - a: lower diagonal (size n, first element unused)
    - b: main diagonal (size n)  
    - c: upper diagonal (size n, last element unused)
    - d: right-hand side vector (size n)
    """
    n = A.shape[0]
    
    # Extract diagonals
    a = np.zeros(n)  # lower diagonal
    b = np.zeros(n)  # main diagonal
    c = np.zeros(n)  # upper diagonal
    
    # Main diagonal
    b = np.diag(A)
    
    # Upper diagonal
    if n > 1:
        c[:-1] = np.diag(A, k=1)
    
    # Lower diagonal  
    if n > 1:
        a[1:] = np.diag(A, k=-1)
    
    # Right-hand side
    if rhs is None:
        d = np.ones(n)
    else:
        d = rhs.copy()
    
    return a, b, c, d

def compute_condition_number(A):
    """Compute condition number using SVD"""
    try:
        cond = np.linalg.cond(A)
        return cond
    except:
        return np.inf

class ParametricMatrixGenerator:
    """Generate matrices with parameter sweeps"""
    
    def __init__(self, config_file="config.yaml"):
        # Load configuration
        self.config = load_config(config_file)
        
        # Get output directory from config
        try:
            self.output_dir = Path(self.config.config.get('matrix_generation', {}).get('output_dir', 'test_matrices_parametric'))
        except:
            self.output_dir = Path('test_matrices_parametric')
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "fem_poisson").mkdir(exist_ok=True)
        (self.output_dir / "dorr").mkdir(exist_ok=True)
        
        self.results = []
    
    def generate_fem_matrices(self, sizes, matrices_per_size=10):
        """Generate FEM 1D Poisson matrices"""
        print(f"Generating FEM 1D Poisson matrices...")
        
        for size in sizes:
            print(f"  Size {size}...")
            
            for matrix_id in range(matrices_per_size):
                try:
                    # Generate FEM matrix
                    A_full, rhs, h = fem_1d_poisson(size + 1)  # +1 because fem function takes n_elements
                    actual_size = A_full.shape[0]
                    
                    # Extract Thomas format
                    a, b, c, d = extract_tridiagonal_thomas_format(A_full, rhs)
                    
                    # Compute condition number
                    condition_number = compute_condition_number(A_full)
                    
                    # Create matrix data
                    matrix_data = {
                        'matrix_type': 'fem_poisson',
                        'size': actual_size,
                        'matrix_id': matrix_id,
                        'parameters': {'h': h, 'n_elements': size + 1},
                        'a': a,
                        'b': b, 
                        'c': c,
                        'd': d,
                        'condition_number': condition_number,
                        'A_full': A_full,  # For verification
                        'rhs_original': rhs
                    }
                    
                    # Save matrix
                    filename = f"matrix_n{actual_size:05d}_id{matrix_id:02d}.pkl"
                    filepath = self.output_dir / "fem_poisson" / filename
                    
                    with open(filepath, 'wb') as f:
                        pickle.dump(matrix_data, f)
                    
                    self.results.append({
                        'matrix_type': 'fem_poisson',
                        'size': actual_size,
                        'matrix_id': matrix_id,
                        'condition_number': condition_number,
                        'parameters': {'h': h},
                        'filepath': str(filepath)
                    })
                    
                except Exception as e:
                    print(f"    Error generating FEM matrix size {size}, id {matrix_id}: {e}")
    
    def generate_dorr_matrices(self, sizes, alpha_values, matrices_per_config=5):
        """Generate Dorr matrices with alpha parameter sweep"""
        print(f"Generating Dorr matrices...")
        
        total_expected = len(sizes) * len(alpha_values) * matrices_per_config
        total_generated = 0
        total_failed = 0
        
        for size in sizes:
            for alpha in alpha_values:
                print(f"  Size {size}, alpha={alpha:.1e}...", end=" ")
                
                for matrix_id in range(matrices_per_config):
                    try:
                        # Generate Dorr matrix
                        A_full = dorr(alpha, size)
                        
                        # Check for NaN/Inf in matrix
                        if np.any(np.isnan(A_full)) or np.any(np.isinf(A_full)):
                            print(f"FAILED - Matrix contains NaN/Inf")
                            total_failed += 1
                            continue
                        
                        # Create synthetic RHS (ones vector)
                        rhs = np.ones(size)
                        
                        # Extract Thomas format
                        a, b, c, d = extract_tridiagonal_thomas_format(A_full, rhs)
                        
                        # Check for NaN/Inf in Thomas format
                        if (np.any(np.isnan(a)) or np.any(np.isinf(a)) or 
                            np.any(np.isnan(b)) or np.any(np.isinf(b)) or
                            np.any(np.isnan(c)) or np.any(np.isinf(c)) or
                            np.any(np.isnan(d)) or np.any(np.isinf(d))):
                            print(f"FAILED - Thomas format contains NaN/Inf")
                            total_failed += 1
                            continue
                        
                        # Compute condition number
                        condition_number = compute_condition_number(A_full)
                        
                        # Check condition number validity
                        if np.isnan(condition_number) or np.isinf(condition_number):
                            print(f"FAILED - Invalid condition number")
                            total_failed += 1
                            continue
                        
                        # Create matrix data
                        matrix_data = {
                            'matrix_type': 'dorr',
                            'size': size,
                            'matrix_id': matrix_id,
                            'parameters': {'alpha': alpha},
                            'a': a,
                            'b': b,
                            'c': c, 
                            'd': d,
                            'condition_number': condition_number,
                            'A_full': A_full,  # For verification
                            'rhs_original': rhs
                        }
                        
                        # Save matrix
                        filename = f"matrix_n{size:05d}_alpha{alpha:.0e}_id{matrix_id:02d}.pkl"
                        filepath = self.output_dir / "dorr" / filename
                        
                        with open(filepath, 'wb') as f:
                            pickle.dump(matrix_data, f)
                        
                        self.results.append({
                            'matrix_type': 'dorr',
                            'size': size,
                            'matrix_id': matrix_id,
                            'condition_number': condition_number,
                            'parameters': {'alpha': alpha},
                            'filepath': str(filepath)
                        })
                        
                        total_generated += 1
                        print(f"SUCCESS (κ={condition_number:.1e})")
                        
                    except Exception as e:
                        print(f"FAILED - Exception: {e}")
                        total_failed += 1
        
        print(f"\n=== DORR MATRIX GENERATION SUMMARY ===")
        print(f"Expected matrices: {total_expected}")
        print(f"Successfully generated: {total_generated}")
        print(f"Failed generations: {total_failed}")
        print(f"Success rate: {100*total_generated/total_expected:.1f}%")
        if total_failed > 0:
            print(f"⚠️  {total_failed} matrices failed to generate!")
    
    def save_summary(self):
        """Save summary of generated matrices"""
        summary_file = self.output_dir / "matrix_summary.pkl"
        
        summary = {
            'total_matrices': len(self.results),
            'matrix_types': list(set([r['matrix_type'] for r in self.results])),
            'sizes': list(set([r['size'] for r in self.results])),
            'results': self.results
        }
        
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        print(f"\nMatrix generation complete!")
        print(f"Total matrices generated: {len(self.results)}")
        print(f"Summary saved to: {summary_file}")
        
        # Print statistics
        print("\nGeneration summary:")
        for matrix_type in summary['matrix_types']:
            type_results = [r for r in self.results if r['matrix_type'] == matrix_type]
            print(f"  {matrix_type}: {len(type_results)} matrices")
            
            sizes = list(set([r['size'] for r in type_results]))
            print(f"    Sizes: {sorted(sizes)}")
            
            conditions = [r['condition_number'] for r in type_results if np.isfinite(r['condition_number'])]
            if conditions:
                print(f"    Condition numbers: {min(conditions):.2e} to {max(conditions):.2e}")

def main():
    """Generate parametric test matrices using configuration file"""
    
    print("=== PARAMETRIC MATRIX GENERATOR ===")
    print("Generating FEM 1D Poisson and Dorr matrices for AFPM analysis")
    print()
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Get parameters from config
    try:
        sizes = config.config.get('matrix_generation', {}).get('sizes', [4, 8, 16, 32, 64, 128, 256, 512, 1024])
        fem_config = config.config.get('matrix_generation', {}).get('fem_poisson', {})
        dorr_config = config.config.get('matrix_generation', {}).get('dorr', {})
        
        matrices_per_size = fem_config.get('matrices_per_size', 10)
        alpha_values_raw = dorr_config.get('alpha_values', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        # Convert string notation to float if needed
        alpha_values = []
        for alpha in alpha_values_raw:
            if isinstance(alpha, str):
                alpha_values.append(float(alpha))
            else:
                alpha_values.append(alpha)
        matrices_per_config = dorr_config.get('matrices_per_config', 5)
        
        print(f"Configuration loaded:")
        print(f"  Matrix sizes: {sizes}")
        print(f"  Alpha values: {alpha_values}")
        print(f"  FEM matrices per size: {matrices_per_size}")
        print(f"  Dorr matrices per config: {matrices_per_config}")
        print()
        
    except Exception as e:
        print(f"Error reading config, using defaults: {e}")
        sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        alpha_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        matrices_per_size = 10
        matrices_per_config = 5
    
    # Create generator
    generator = ParametricMatrixGenerator()
    
    # Generate FEM matrices
    generator.generate_fem_matrices(sizes, matrices_per_size)
    
    # Generate Dorr matrices  
    generator.generate_dorr_matrices(sizes, alpha_values, matrices_per_config)
    
    # Save summary
    generator.save_summary()

if __name__ == "__main__":
    main()