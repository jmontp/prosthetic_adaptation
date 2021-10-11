import numpy as np



def unit_test():
    
    exponents = 6
    
    values = np.array([[2.0]])
    
    evaluated = np.array([[1,2,4,8,16,32]])
    
    poly_basis = PolynomialBasis(exponents,'phase')
    
    poly_evald = poly_basis.evaluate_derivative(values,0)
    
    print(evaluated)
    
    print(poly_evald)
    
    #assert(np.linalg.norm(evaluated-poly_evald) < 1e-7)
    
    derivative = np.array([[0,1,4,12,24,80]])
    
    poly_deri_evald = poly_basis.evaluate_derivative(values)
    
    print(derivative)
    
    print(poly_deri_evald)
    
    second_derivative = np.array([[0,0,2,12,48,160]])
    
    poly_second_deri = poly_basis.evaluate_derivative(values, 2)
    
    print(second_derivative)
    
    print(poly_second_deri)
    
    p3d = poly_basis.evaluate_derivative(values, 4)
    
    print(p3d)


    exponent = 3
    
    x = np.array([[0.435]])
    
    fourier = FourierBasis(exponent,'phase')
    
    print("F evaluate: {}".format(fourier.evaluate(x)))
    print("F first derivative evaluate: {}".format(fourier.evaluate_derivative(x,1)))
    print("F sec derivative evaluate: {}".format(fourier.evaluate_derivative(x,2)))
    print("F third derivative evaluate: {}".format(fourier.evaluate_derivative(x,3)))

    


    
if __name__ == '__main__':
    unit_test()