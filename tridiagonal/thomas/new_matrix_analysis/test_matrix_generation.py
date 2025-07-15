#!/usr/bin/env python3
"""
Test script to diagnose matrix generation issues
Tests Dorr matrix generation for extreme parameter combinations
"""

import numpy as np
import sys
from matrix_generators import dorr, compute_condition_number, extract_tridiagonal_thomas_format

def test_dorr_generation():
    """Test Dorr matrix generation for various size/alpha combinations"""
    
    # Test parameters that might cause issues
    sizes = [4, 8, 16, 32, 64, 128]
    alpha_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    
    print("=== DORR MATRIX GENERATION TEST ===")
    print("Testing combinations that might fail...\n")
    
    success_count = 0
    failure_count = 0
    extreme_condition_count = 0
    
    for size in sizes:
        print(f"--- Size {size} ---")
        
        for alpha in alpha_values:
            try:
                # Test matrix generation
                A_full = dorr(alpha, size)
                
                # Check for NaN/Inf values
                if np.any(np.isnan(A_full)) or np.any(np.isinf(A_full)):
                    print(f"  Alpha {alpha:.1e}: FAILED - Matrix contains NaN/Inf values")
                    failure_count += 1
                    continue
                
                # Test condition number computation
                try:
                    condition_number = compute_condition_number(A_full)
                    
                    if np.isnan(condition_number) or np.isinf(condition_number):
                        print(f"  Alpha {alpha:.1e}: FAILED - Condition number is NaN/Inf")
                        failure_count += 1
                        continue
                    
                    # Check for extreme condition numbers
                    if condition_number > 1e15:
                        print(f"  Alpha {alpha:.1e}: WARNING - Extreme condition number: {condition_number:.2e}")
                        extreme_condition_count += 1
                    
                except Exception as cond_e:
                    print(f"  Alpha {alpha:.1e}: FAILED - Condition number computation error: {cond_e}")
                    failure_count += 1
                    continue
                
                # Test Thomas format extraction
                try:
                    rhs = np.ones(size)
                    a, b, c, d = extract_tridiagonal_thomas_format(A_full, rhs)
                    
                    # Check for NaN/Inf in Thomas format
                    if (np.any(np.isnan(a)) or np.any(np.isinf(a)) or 
                        np.any(np.isnan(b)) or np.any(np.isinf(b)) or
                        np.any(np.isnan(c)) or np.any(np.isinf(c)) or
                        np.any(np.isnan(d)) or np.any(np.isinf(d))):
                        print(f"  Alpha {alpha:.1e}: FAILED - Thomas format contains NaN/Inf")
                        failure_count += 1
                        continue
                        
                except Exception as thomas_e:
                    print(f"  Alpha {alpha:.1e}: FAILED - Thomas format extraction error: {thomas_e}")
                    failure_count += 1
                    continue
                
                # If we get here, generation succeeded
                print(f"  Alpha {alpha:.1e}: SUCCESS - Condition: {condition_number:.2e}")
                success_count += 1
                
            except Exception as e:
                print(f"  Alpha {alpha:.1e}: FAILED - Matrix generation error: {e}")
                failure_count += 1
        
        print()
    
    print("=== SUMMARY ===")
    print(f"Successful generations: {success_count}")
    print(f"Failed generations: {failure_count}")
    print(f"Extreme condition numbers (>1e15): {extreme_condition_count}")
    print(f"Total tested: {success_count + failure_count}")
    
    if failure_count > 0:
        print(f"\n⚠️  {failure_count} combinations failed!")
        print("This explains why some data points are missing from the plots.")
    else:
        print("\n✅ All matrix generations succeeded!")

def test_specific_problematic_combinations():
    """Test specific combinations that might be problematic"""
    print("\n=== TESTING SPECIFIC PROBLEMATIC COMBINATIONS ===")
    
    # These are combinations that might fail based on theory
    problematic = [
        (64, 1e-5), (64, 1e-6),
        (128, 1e-4), (128, 1e-5), (128, 1e-6)
    ]
    
    for size, alpha in problematic:
        print(f"\nTesting Size {size}, Alpha {alpha:.1e}:")
        try:
            A_full = dorr(alpha, size)
            condition_number = compute_condition_number(A_full)
            
            print(f"  Matrix generation: SUCCESS")
            print(f"  Condition number: {condition_number:.2e}")
            
            # Check if condition number is reasonable for floating point
            if condition_number > 1e15:
                print(f"  ⚠️  Condition number exceeds 1e15 - likely numerical issues!")
            
            # Test actual matrix solve
            rhs = np.ones(size)
            try:
                x_exact = np.linalg.solve(A_full, rhs)
                print(f"  Direct solve: SUCCESS")
                
                # Check solution quality  
                residual = np.linalg.norm(A_full @ x_exact - rhs)
                print(f"  Residual: {residual:.2e}")
                
            except Exception as solve_e:
                print(f"  Direct solve: FAILED - {solve_e}")
                
        except Exception as e:
            print(f"  Matrix generation: FAILED - {e}")

if __name__ == "__main__":
    test_dorr_generation()
    test_specific_problematic_combinations()