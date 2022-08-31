"""
This file is meant to test the function basis code
"""

#Common imports
import numpy as np

#My custom imports
from context import kmodel
from kmodel.function_bases import PolynomialBasis, FourierBasis


def unit_test():
    
    exponents = 6
    
    values = np.array([[2.0]])
    
    evaluated = np.array([[1,2,4,8,16,32]])
    
    poly_basis = PolynomialBasis(exponents,'phase')
    
    poly_evald = poly_basis.evaluate(values)
    
    print(evaluated)
    
    print(poly_evald)
    
    assert(np.linalg.norm(evaluated-poly_evald) < 1e-7)
    
    
    
    x = np.array([[0.435]])
    
    fourier = FourierBasis(1,'phase')
    
    expected_result = np.array([1,float(np.sin(2*np.pi*x)),float(np.cos(2*np.pi*x))])
    fourier_result = fourier.evaluate(x)

    print(f"F evaluate: {fourier_result}")
    print(f"Expected {expected_result}")

    
    assert(np.linalg.norm(fourier_result-expected_result) < 1e-7)


    
if __name__ == '__main__':
    unit_test()