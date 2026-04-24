#!/usr/bin/env python3
"""
Debug script to understand the Min/Max issue.
"""

import numpy as np
from sympy import symbols, lambdify
from iMCTS.src.utils.reward import sp_module, _safe_min, _safe_max

def debug_min_function():
    """Debug the _safe_min function directly."""
    print("Debugging _safe_min function...")
    
    # Test with scalar and array
    scalar = 4.0
    array = np.array([1.0, 2.0, 3.0])
    
    print(f"Scalar: {scalar}")
    print(f"Array: {array}")
    
    try:
        result = _safe_min(scalar, array)
        print(f"_safe_min(scalar, array) = {result}")
        print(f"Result type: {type(result)}")
        print(f"Result shape: {result.shape if hasattr(result, 'shape') else 'No shape'}")
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        result = _safe_min(array, scalar)
        print(f"_safe_min(array, scalar) = {result}")
        print(f"Result type: {type(result)}")
        print(f"Result shape: {result.shape if hasattr(result, 'shape') else 'No shape'}")
    except Exception as e:
        print(f"Error: {e}")

def debug_lambdify():
    """Debug the lambdify process."""
    print("\nDebugging lambdify process...")
    
    x0 = symbols('x0')
    R0 = symbols('R0')
    R1 = symbols('R1')
    
    # Create the expression that was failing: ((R0 * x0) - Min(R1,x0))
    from sympy import Min
    expr = R0 * x0 - Min(R1, x0)
    
    print(f"Expression: {expr}")
    
    # Create lambdified function using our fixed sp_module
    f = lambdify([x0, R0, R1], expr, modules=sp_module)
    
    # Test with sample data
    x_test = np.array([1.0, 2.0, 3.0])
    R0_test = 2.0
    R1_test = 4.0
    
    print(f"x_test: {x_test}")
    print(f"R0_test: {R0_test}")
    print(f"R1_test: {R1_test}")
    
    try:
        # Test the Min function directly
        print("Testing Min function directly...")
        min_result = _safe_min(R1_test, x_test)
        print(f"Min result: {min_result}")
        
        # Test the full expression
        print("Testing full expression...")
        result = f(x_test, R0_test, R1_test)
        print(f"Full expression result: {result}")
        print(f"Result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_min_function()
    debug_lambdify()
