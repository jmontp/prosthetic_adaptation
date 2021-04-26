"""
Model Generator 

This code is meant to generate the regressor model based on a Kronecker Product of different function which each are one dimensional functions of task variables. 


"""
#H is the partial derivative of the model with respect to the state variables and the pca axis coefficient variables 
#The partial derivative with respect to the state variable is the derivative of the function row for the row function vector for that particular state variable kroneckerd with he other normal funcitons
#The partial derivative with respect to he pca axis is just the pca axis times the kronecker productl
import pandas as pd
import numpy as np
import math
import pickle
from os import path
from sklearn.decomposition import PCA
#--------------------------
#Need to create two objects:
#Basis object:
# basis(x): takes input and returns the value
# basis_name:
# basis size
# basis params
# variable name
#Dont use basis directly, its just a blueprint to what you need to implement
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
class Polynomial_Basis(Basis):
    def __init__(self, n, var_name):
        Basis.__init__(self,n,var_name)
        self.size = n

    #This function will evaluate the model at the given x value
    def evaluate(self,x):
        #result = [math.pow(x,i) for i in range(0,self.n)]
        #Power will evaluate elementwise by the power defined in the second 
        #argument. Arrange will generate a list from 0-n-1, therefore it will 
        #evaluate the power of each element
        x_array = np.repeat(x,self.n,axis=1)
        power_array = np.arange(self.n)
        output = np.power(x_array,power_array)
        return output

    #This function will evaluate the derivative of the model at the given 
    # x value
    #TODO: Unit test please, please please
    def evaluate_derivative(self,x,num_derivatives=1):
        if(num_derivatives == 0):
            return self.evaluate(x)
        
        if(num_derivatives < self.size):
            
            x_array = np.repeat(x,self.size,axis=1)
            coefficient_array = np.arange(self.n)
            temp_array = np.arange(self.n)
            # print("Coefficients: " + str(coefficient_array))
            # print("Temp array: " + str(temp_array))
            for i in range(1,num_derivatives):
                temp_array = temp_array-1
                coefficient_array = coefficient_array*(temp_array)
                # print("Coefficients: " + str(coefficient_array))
                # print("Temp array: " + str(temp_array))
            
            #Generate power array
            power_array = np.arange(-num_derivatives,self.size-num_derivatives)
            #Set negative indices to zero
            power_array = np.where(power_array<0,0,power_array)
            
            return (np.power(x_array,power_array)*coefficient_array)
        else:
            return np.repeat(0,self.size)

         


#This will create a Polynomial Basis with n harmonic frequencies
# The variable name is also needed

class Fourier_Basis(Basis):
    def __init__(self, n, var_name):
        Basis.__init__(self, n, var_name)
        self.size = 2*n+1

    #This function will evaluate the model at the given x value
    def evaluate(self,x):
        #l is used to generate the coefficients of the series
        l = np.arange(1,self.n+1).reshape(1,-1)
        
        #Initialize everything as one to get 
        result = np.ones((x.shape[0],self.size))
        #print("Size: {}".format(self.size))
        #print("n: {}".format(self.n))
        #print("Result size: {}".format(result.shape))
        #Add the sine and cos part
        result[:,1:self.n+1] = np.cos(2*np.pi*x @ l)
        result[:,self.n+1:] =  np.sin(2*np.pi*x @ l)
        return result


    #This function will evaluate the derivative of the model at the given 
    # x value
    def evaluate_derivative(self,x,num_derivatives=1):
        if (num_derivatives == 0):
            return self.evaluate(x)
        
        #l is used to generate the coefficients of the series
        l = np.arange(1,self.n+1).reshape(1,-1)
        
        #Initialize everything as one to get 
        result = np.zeros((x.shape[0],self.size))
        
        #Add the sine and cos part
        #https://www.wolframalpha.com/input/?i=d%5En+cos%282*pi*a*x%29%2Fdx%5En
        result[:,1:self.n+1] = np.power((2*np.pi*l),num_derivatives)*np.cos(0.5*np.pi*(num_derivatives + 4*x @ l)) 
        #https://www.wolframalpha.com/input/?i=d%5En+sin%282*pi*a*x%29%2Fdx%5En
        result[:,self.n+1:] =  np.power((2*np.pi*l),num_derivatives)*np.sin(0.5*np.pi*(num_derivatives + 4*x @ l))
        
        return result



#Not really using this right now so keeping in the old format
class Bernstein_Basis(Basis):

    def __init__(self, n, var_name):
        Basis.__init__(self, n, var_name)
        self.size = n

    def evaluate(self,x):
        basis = [math.comb(self.n,i)*math.pow(x,i)*math.pow((1-x),(self.n-i)) for i in range(0,self.n+1)];
        return np.array(basis)

    def evaluate_derivative(self,x):
        #raise NotImplementedError "Bernstain Basis derivative not implmented"
        pass




def unit_test():
    
    exponents = 6
    
    values = np.array([[2.0]])
    
    evaluated = np.array([[1,2,4,8,16,32]])
    
    poly_basis = Polynomial_Basis(exponents,'phase')
    
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
    
    fourier = Fourier_Basis(exponent,'phase')
    
    print("F evaluate: {}".format(fourier.evaluate(x)))
    print("F first derivative evaluate: {}".format(fourier.evaluate_derivative(x,1)))
    print("F sec derivative evaluate: {}".format(fourier.evaluate_derivative(x,2)))
    print("F third derivative evaluate: {}".format(fourier.evaluate_derivative(x,3)))

    


    
if __name__ == '__main__':
    unit_test()