#!/usr/bin/env python3
"""
Test script to verify the Max function fix for axis parameter support.
"""

import numpy as np
from sympy import symbols, lambdify
from iMCTS.src.utils.reward import sp_module

def test_max_with_axis():
    """Test that Max function works with axis parameter."""
    print("Testing Max function with axis parameter...")
    
    # Create test data
    x0 = symbols('x0')
    R0 = symbols('R0')
    R1 = symbols('R1')
    
    # Create the expression that was failing: Max(Pow(R0,x0),(R1 / x0))
    from sympy import Max, Pow
    expr = Max(Pow(R0, x0), R1 / x0)
    
    print(f"Expression: {expr}")
    
    # Create lambdified function using our fixed sp_module
    f = lambdify([x0, R0, R1], expr, modules=sp_module)
    
    # Test with sample data
    x_test = np.array([1.0, 2.0, 3.0])
    R0_test = 2.0
    R1_test = 4.0
    
    try:
        result = f(x_test, R0_test, R1_test)
        print(f"Test successful! Result: {result}")
        print(f"Result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

def test_min_with_axis():
    """Test that Min function works with axis parameter."""
    print("\nTesting Min function with axis parameter...")
    
    # Create test data
    x0 = symbols('x0')
    R0 = symbols('R0')
    R1 = symbols('R1')
    
    # Create a Min expression
    from sympy import Min, Pow
    expr = Min(Pow(R0, x0), R1 / x0)
    
    print(f"Expression: {expr}")
    
    # Create lambdified function using our fixed sp_module
    f = lambdify([x0, R0, R1], expr, modules=sp_module)
    
    # Test with sample data
    x_test = np.array([1.0, 2.0, 3.0])
    R0_test = 2.0
    R1_test = 4.0
    
    try:
        result = f(x_test, R0_test, R1_test)
        print(f"Test successful! Result: {result}")
        print(f"Result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

def test_simple_max():
    """Test simple Max expression that was failing."""
    print("\nTesting simple Max expression...")
    
    # Create test data
    x0 = symbols('x0')
    
    # Create the expression that was failing: Max(x0,(x0 + x0))
    from sympy import Max, Add
    expr = Max(x0, Add(x0, x0))
    
    print(f"Expression: {expr}")
    
    # Create lambdified function using our fixed sp_module
    f = lambdify([x0], expr, modules=sp_module)
    
    # Test with sample data
    x_test = np.array([1.0, 2.0, 3.0])
    
    try:
        result = f(x_test)
        print(f"Test successful! Result: {result}")
        print(f"Result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

def test_problematic_expression():
    """Test the specific problematic expression from the error."""
    print("\nTesting problematic expression: ((R0 * x0) - Min(R1,x0))")
    
    # Create test data
    x0 = symbols('x0')
    R0 = symbols('R0')
    R1 = symbols('R1')
    
    # Create the expression that was failing: ((R0 * x0) - Min(R1,x0))
    from sympy import Min, Mul, Add
    expr = Mul(R0, x0) - Min(R1, x0)
    
    print(f"Expression: {expr}")
    
    # Create lambdified function using our fixed sp_module
    f = lambdify([x0, R0, R1], expr, modules=sp_module)
    
    # Test with sample data
    x_test = np.array([1.0, 2.0, 3.0])
    R0_test = 2.0
    R1_test = 4.0
    
    try:
        result = f(x_test, R0_test, R1_test)
        print(f"Test successful! Result: {result}")
        print(f"Result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Running tests for Max/Min function axis parameter support...")
    
    success1 = test_max_with_axis()
    success2 = test_min_with_axis()
    success3 = test_simple_max()
    success4 = test_problematic_expression()
    
    if success1 and success2 and success3 and success4:
        print("\n✅ All tests passed! The fix should resolve the original error.")
    else:
        print("\n❌ Some tests failed. The fix may need further adjustments.")
