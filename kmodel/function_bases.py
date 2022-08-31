"""
Model Generator 

This code is meant to generate the regressor model based on a Kronecker Product of different function which each are one dimensional functions of task variables. 


"""


#TODO - It might be a good idea to implement numeric differentiation here


from msilib.schema import Error
from matplotlib.pyplot import axis
import numpy as np


class Basis:
    """
    This class is defined to have 
    """
    def __init__(self, n, var_name):
        self.n = n
        self.var_name = var_name

    #Need to implement with other subclasses
    def evaluate(self,x):
        pass


##Define classes that will be used to calculate kronecker products in real time

#This will create a Polynomial Basis with n entries
# The variable name is also needed
class PolynomialBasis(Basis):
    def __init__(self, n, var_name):
        Basis.__init__(self,n,var_name)
        self.size = n
        self.name = "Polynomial"
        #List of powers that every element in x will be evaluated to
        self.__powers = np.arange(n)
        self.__powers_copy = np.arange(n)
        #Pre-allocate memory to 
        self.__one_array = np.ones((1,n))
        self.__one_array_copy = np.ones((1,n))
    #This function will evaluate the model at the given x value
    def evaluate(self,x,derivative=0):
        
        
        if derivative ==  0:
            return np.power(x * self.__one_array,self.__powers)
        
        elif derivative > 0:
            #Temprorarily change one array to zero out elements
            # this avoids memory allocation in run time
            self.__one_array[:derivative] = 0
            #Get the correct power values using max, out argument places the 
            # results in the powers matrix
            self.__powers = np.max(0, self.__powers - derivative,
                                    out=self.__powers)
            out = np.power(x * self.__one_array,self.__powers)
            #Reset values
            self.__one_array[:] = self.__one_array_copy
            self.__powers[:] = self.__powers_copy
            
            return out
        else:
            raise Error("Derivative must be greater than zero")
        #return np.polynomial.polynomial.polyvander(x, self.n-1)


         


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
        self.name = "Fourier"

    #This function will evaluate the model at the given x value
    def evaluate(self,x):
        x = x.reshape(-1,1)

        #l is used to generate the coefficients of the series
        l = np.arange(1,self.n+1).reshape(1,-1)
        
        #Initialize everything as empty for speed increase to get 
        result = np.empty((x.shape[0],self.size))

        result[:,0] = 1
        result[:,1:self.n+1] = np.sin(2*np.pi*x @ l)
        result[:,self.n+1:] =  np.cos(2*np.pi*x @ l)
        
        return result


class LegendreBasis(Basis):
    "Legendre polynomials are on [-1,1]"
    def __init__(self, n, var_name):
        Basis.__init__(self, n, var_name)
        self.size = n
        self.name = "Legendre"

    def evaluate(self,x):
        return np.polynomial.legendre.legvander(x, self.n-1)


class ChebyshevBasis(Basis):

    "Chebyshev polynomials are on [-1,1]"
    def __init__(self, n, var_name):
        Basis.__init__(self, n, var_name)
        self.size = n
        self.name = "Chebyshev"

    def evaluate(self, x):
        return np.polynomial.chebyshev.chebvander(x, self.n-1)


class HermiteBasis(Basis):
    "Hermite polynomials are on [-inf,inf]"
    def __init__(self, n, var_name):
        Basis.__init__(self, n, var_name)
        self.size = n
        self.name = "Hermite"

    def evaluate(self,x):
        return np.polynomial.hermite_e.hermevander(x, self.n-1)
