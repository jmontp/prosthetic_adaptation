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
        power_arrray = np.arange(self.n)
        return np.power(x_array,power_arrray)

    #This function will evaluate the derivative of the model at the given 
    # x value
    def evaluate_derivative(self,x,num_derivatives=1):
        if(num_derivatives < self.size):
            
            x_array = np.repeat(x,self.size,axis=1)
            coefficient_array = np.arrange(self.n)
            #Generate power array
            power_array = np.arange(-num_derivatives,self.size-num_derivatives)
            #Set negative indices to zero
            power_array = np.whereis(power_array<0,0,power_array)
            
            return np.power(x_array,power_array)*coefficient_array
        else:
            return np.repeat(0,self.size)
         


#This will create a Polynomial Basis with n harmonic frequencies
# The variable name is also needed

class Fourier_Basis(Basis):
    def __init__(self, n, var_name):
        Basis.__init__(self, n, var_name)
        self.size = 2*n-1

    #This function will evaluate the model at the given x value
    def evaluate(self,x):
        #l is used to generate the coefficients of the series
        l = np.arange(1,self.n).reshape(1,-1)
        
        #Initialize everything as one to get 
        result = np.ones((x.shape[0],self.size))
        
        #Add the sine and cos part
        result[:,1:self.n] = np.cos(2*np.pi*x @ l)
        result[:,self.n:] =  np.sin(2*np.pi*x @ l)
        return result


    #This function will evaluate the derivative of the model at the given 
    # x value
    def evaluate_derivative(self,x,num_derivatives=1):
        #l is used to generate the coefficients of the series
        l = np.arange(1,self.n).reshape(1,-1)
        
        #Initialize everything as one to get 
        result = np.ones((x.shape[0],self.size))
        
        #Add the sine and cos part
        #https://www.wolframalpha.com/input/?i=d%5En+cos%282*pi*a*x%29%2Fdx%5En
        result[:,1:self.n] = np.power((2*np.pi*l),num_derivatives)*np.cos(0.5*np.pi*(num_derivatives + 4*x @ l)) 
        #https://www.wolframalpha.com/input/?i=d%5En+sin%282*pi*a*x%29%2Fdx%5En
        result[:,self.n:] =  np.power((2*np.pi*l),num_derivatives)*np.sin(0.5*np.pi*(num_derivatives + 4*x @ l))
        
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



#Model Object:
# list of basis objects
# string description
# model_size
# pca_axis - list of np array of model_size length
# coefficient_list - list of coefficients for each pca_axis
#--------------------------
class Kronecker_Model:
    def __init__(self, output_name, *funcs):
        self.funcs = funcs

        #Calculate the size of the parameter array
        #Additionally, pre-allocate arrays for kronecker products intermediaries 
        # to speed up results
        self.output_name = output_name
        self.alocation_buff = []
        self.order = []
        size = 1
        for func in funcs:
            #Since we multiply left to right, the total size will be on the left 
            #and the size for the new row will be on the right
            print((str(size), str(func.size)))
            self.alocation_buff.append(np.zeros((size, func.size)))

            size = size * func.size

            self.order.append(func.var_name)


        self.size = size
        self.num_states = len(funcs)
        self.subjects = {}
        #Todo: Add average pca coefficient
        
    
    def add_subject(self,subjects):
        for subject,filename in subjects:
            self.subjects[subject] = \
                {'filename': filename, \
                 'dataframe': pd.read_parquet(filename, columns=[self.output_name,*self.order]), \
                 'optimal_xi': [], \
                 'least_squares_info': [], \
                 'pca_axis': [], \
                 'pca_coefficients': [] \
             }

    #Evaluate the models at the function inputs that are received
    #The function inputs are expected in the same order as they where defined
    #Alternatively, you can also input a dictionary with the var_name as the key and the 
    # value you want to evaluate the function as the value
    def evaluate(self, *function_inputs,partial_derivative=None, result_buffer=None):
        
        result = np.array([1])
        
        for i in range(self.num_states):

            #Verify if we want to take the partial derivative of this function
            if(partial_derivative is not None and curr_func.var_name in partial_derivative):
                apply_derivative = True
            else: 
                apply_derivative = False


            result=fast_kronecker(result,self.funcs[i].evaluate_conditional(function_inputs[i],False))#, self.alocation_buff[i], False)
            
        return result.copy()

    def evaluate_pandas(self, dataframe, partial_derivative=None):
        #Todo: Implement partial derivatives
        #Can be done by creating a new list of functions and adding partial derivatives when needed
        func_list = []
        for func in self.funcs:
            if(func.var_name in partial_derivative):
                num_derivatives = partial_derivative.count(func.var_name)
                func_list.append(lambda x: func.evaluate_derivative(x,num_derivatives))
            else:
                func_list.append(func.evaluate)
        return numpy_kronecker(dataframe,self.funcs)
    
    def evaluate_subject(self,subject, dataframe):
        regressor = self.evaluate_pandas(dataframe)
        print(regressor.shape)
        xi = self.subjects[subject]['optimal_xi']
        print(regressor.shape)
        return regressor @ xi
    
    
    
    #Future optimizations
    #@numba.jit(nopython=True, parallel=True)
    def least_squares(self,dataframe,output,splits=100):

        RTR = np.zeros((self.size,self.size))
        yTR = np.zeros((1,self.size))
        RTy = np.zeros((self.size,1))
        yTy = 0
        for sub_dataframe in np.array_split(dataframe,splits):
            R = numpy_kronecker(sub_dataframe,self.funcs)
            #nans = sub_dataframe[output].isnull().sum()
            #print(nans)
            #print(sub_dataframe.shape[0])
            y = sub_dataframe[output].values[:,np.newaxis]
            RTR_ = R.T @ R
            
            RTR += RTR_
            yTR += y.T @ R
            RTy += R.T @ y
            yTy += y.T @ y

        return np.linalg.solve(RTR, RTy), RTR, RTy, yTR, yTy


    def fit_subjects(self):
        for name,subject_dict in self.subjects.items():
            print("Doing " + name)
            print(subject_dict['filename'])
            data = subject_dict['dataframe']
            output = self.least_squares(data,self.output_name)
            subject_dict['optimal_xi'] = output[0]
            subject_dict['least_squares_info'] = output[1:]


    def calculate_gait_fingerprint(self):
        num_subjects = len(self.subjects.values())
        #Get all the gait fingerprints into a matrix
        XI = np.array([subject['optimal_xi'] for subject in self.subjects.values()]).reshape(num_subjects,-1)
        print(XI.shape)
        pca = PCA(n_components=num_subjects)
        pca.fit(XI) 
        self.pca_result = pca
        


    #Todo: Need to add functionality for the models pca_axis list
    #Dont know if I want to have it run least squares in the initialization
    def set_pca_axis(self,pca_axis):
        self.pca_axis = pca_axis
        self.pca_coefficients = [0]*len(self.pca_axis)

    def set_pca_coefficients(self,pca_coefficients):
        self.pca_coefficients = pca_coefficients

    def sum_pca_axis(self,pca_coefficients):
        if(len(self.pca_axis) != len(pca_coefficients)):
            err_string = 'Wrong amount of inputs. Received:'  + str(len(pca_coefficients)) + ', expected:' + str(len(self.pca_axis))
            raise ValueError(err_string)

        return sum([axis*coeff for axis,coeff in zip(self.pca_axis,pca_coefficients)])

    def evaluate_scalar_output(self,*function_inputs,partial_derivative=None):
        states = function_inputs[:self.num_states]
        pca_coefficients = function_inputs[self.num_states:]
        return self.evaluate(*states,partial_derivative=partial_derivative) @ self.sum_pca_axis(pca_coefficients).T

    def __str__(self):
        output = ''
        for func in self.funcs:
            func_type = type(func).__name__
            if(func_type == 'Polynomial_Basis'):
                basis_identifier = 'P'
            elif (func_type == 'Fourier_Basis'):
                basis_identifier = 'F'
            elif (func_type == 'Bernstein_Basis'):
                basis_identifier = 'B'
            else:
                raise TypeError("This is not a basis")

            output += func.var_name + '-' + str(func.n)+ basis_identifier + '--'
        return output

    def get_order(self):
        return self.order


def optimized_least_squares(output,input,model_size,splits=10):
    RTR = np.zeros((model_size,model_size))
    yTR = np.zeros((1,model_size))
    RTy = np.zeros((model_size,1))
    yTy = 0
    for sub_dataframe in np.array_split(input,splits):
        R = numpy_kronecker(sub_dataframe,self.funcs)
        #nans = sub_dataframe[output].isnull().sum()
        #print(nans)
        #print(sub_dataframe.shape[0])
        y = sub_dataframe[output].values[:,np.newaxis]
        RTR_ = R.T @ R
        
        RTR += RTR_
        yTR += y.T @ R
        RTy += R.T @ y
        yTy += y.T @ y

    return np.linalg.solve(RTR, RTy), RTR, RTy, yTR, yTy


#Evaluate model 
def model_prediction(model,ξ,*input_list,partial_derivative=None):
    result = [model.evaluate(*function_inputs,partial_derivative=partial_derivative)@ξ for function_inputs in zip(*input_list)]
    return np.array(result)



#@vectorize(nopython=True)
def numpy_kronecker(dataframe,funcs):
    rows = dataframe.shape[0]
    output = np.array(1).reshape(1,1,1)
    for func in funcs:
        output = (output[:,np.newaxis,:]*func.evaluate(dataframe[func.var_name].values[:,np.newaxis])[:,:,np.newaxis]).reshape(rows,-1)
        print("I'm alive, size = " + str(output.shape))

    return output

####################################################################################
#//////////////////////////////////////////////////////////////////////////////////#
####################################################################################