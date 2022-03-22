"""
Model Generator 

This code is meant to generate the regressor model based on a Kronecker Product of different function which each are one dimensional functions of task variables. 


"""


#TODO - It might be a good idea to implement numeric differentiation here


import numpy as np


class Basis:
    def __init__(self, n, var_name):
        self.n = n
        self.var_name = var_name

    #Need to implement with other subclasses
    def evaluate(self,x):
        pass

    #Need to implement the derivative of this also
    def evaluate_derivative(self,x):
        pass

    def evaluate_conditional(self,x,apply_derivative,num_derivatives=1):
        if(apply_derivative == True):
            return self.evaluate_derivative(x,num_derivatives)
        else:
            return self.evaluate(x)


##Define classes that will be used to calculate kronecker products in real time

#This will create a Polynomial Basis with n entries
# The variable name is also needed
class PolynomialBasis(Basis):
    def __init__(self, n, var_name):
        Basis.__init__(self,n,var_name)
        self.size = n

    #This function will evaluate the model at the given x value
    def evaluate(self,x):
       return np.polynomial.polynomial.polyvander(x, self.n-1)
         


#This will create a Polynomial Basis with n harmonic frequencies
# The variable name is also needed
class FourierBasis(Basis):
    """
    Fourier basis function

    Keyword Arguments
    n - number of basis functions
        Evaluate returns output of size 2*n + 1
        where every pair element is cos 
        and every odd element is sine
    var_name - name of the basis function
    """

    def __init__(self, n, var_name):
        """
        Fourier basis function

        Keyword Arguments
        n - number of basis functions
            Evaluate returns output of size 2*n + 1
            where every pair element is cos 
            and every odd element is sine
        var_name - name of the basis function
        """

        Basis.__init__(self, n, var_name)
        self.size = 2*n+1

    #This function will evaluate the model at the given x value
    def evaluate(self,x):
        x = x.reshape(-1,1)

        #l is used to generate the coefficients of the series
        l = np.arange(1,self.n+1).reshape(1,-1)
        
        #Initialize everything as one to get 
        result = np.ones((x.shape[0],self.size))
       
        result[:,1:self.n+1] = np.sin(2*np.pi*x @ l)
        result[:,self.n+1:] =  np.cos(2*np.pi*x @ l)
        return result


class LegendreBasis(Basis):
    "Legendre polynomials are on [-1,1]"
    def __init__(self, n, var_name):
        Basis.__init__(self, n, var_name)
        self.size = n

    def evaluate(self,x):
        return np.polynomial.legendre.legvander(x, self.n-1)


class ChebyshevBasis(Basis):

    "Chebyshev polynomials are on [-1,1]"
    def __init__(self, n, var_name):
        Basis.__init__(self, n, var_name)
        self.size = n
    
    def evaluate(self, x):
        return np.polynomial.chebyshev.chebvander(x, self.n-1)


class HermiteBasis(Basis):
    "Hermite polynomials are on [-inf,inf]"
    def __init__(self, n, var_name):
        Basis.__init__(self, n, var_name)
        self.size = n

    def evaluate(self,x):
        return np.polynomial.hermite_e.hermevander(x, self.n-1)
