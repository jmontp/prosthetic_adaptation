"""
Model Generator 

This code is meant to generate the regressor model based on a Kronecker Product of different function which each are one dimensional functions of task variables. 


"""
import numpy as np
from math import comb
import timeit


#This is another way that might be cleaner than the function-number tuple
def polynomial_basis(n):
	def p_func(x):
		basis = [x**i for i in range(0,n)]
		return np.array(basis)
	return (p_func, n)


#Return a polynomial basis function with n members
def polynomial_function(x,n):
	basis = [x**i for i in range(0,n)]
	return np.array(basis)

#Return Bernstein Polynomial basis function with n members
def bernstein_polynomial(x,n):
	basis = [comb(n,i)*(1-x)**(n-i) for i in range(0,n+1)];
	return np.array(basis)


#Return fourier series basis function
#We assume that x goes from 0 to 1
def fourier_series(x,n):
	basis_m = [[np.cos(2*np.pi*i*x), np.sin(2*np.pi*i*x)] for i in range(1,n)]
	basis_f = np.array(basis_m).flatten()
	basis = np.insert(basis_f,0,1)
	return basis


#This function will return a Kronecker model of all the basis functions that are inputed
#func_tuples - tuple with a function basis as the first entry and the amount of entries in that function 
def kronecker_generator(*func_tuples):
	#This will serve as the list of input array
	size = 1;
	for func in func_tuples:
		size = size * func[1]

	#This will calculate the kronecker product based on the basis functions that are passed in 
	def kronecker_builder(*function_inputs):
		result = np.array([1])
		for values in zip(function_inputs,func_tuples):
			curr_val,(curr_func,curr_n) = values
			result = np.kron(result,curr_func(curr_val,curr_n))
		return result

	return (kronecker_builder, size)



#Calculate the least squares based on the data
def least_squares(model, output, *data):

	#Calculate the regressor matrix as a python array
	regressor_list = [model(*entry) for entry in zip(*data)]
	#print(regressor_list)
	#R is the regressor matrix
	R = np.array(regressor_list)
	#print(R)
	
	output = np.array(output)

	return np.linalg.solve(R.T @ R, R.T @ output), R



def model_prediction(model, parameters, *data):
	model_output = [model(*entry) @ parameters for entry in zip(*data)]
	return np.array(model_output)


##################################

##################################
#Test everything out
def unit_test():

	phase = (fourier_series, 3)
	ramp = (polynomial_function, 2)

	model,size = kronecker_generator(phase, ramp)


	phase_sample = 0.1
	ramp_sample = 0.2
	output = model(phase_sample, ramp_sample)
	output = model(phase_sample, ramp_sample)
	output = model(phase_sample, ramp_sample)
	output = model(phase_sample, ramp_sample)

	print(output)

	#This is the expected kronecker product
	test = [ramp_sample**0*np.cos(2*np.pi*0*phase_sample), ramp_sample**1*np.cos(2*np.pi*0*phase_sample), 
			ramp_sample**0*np.cos(2*np.pi*1*phase_sample), ramp_sample**1*np.cos(2*np.pi*1*phase_sample),
			ramp_sample**0*np.cos(2*np.pi*2*phase_sample), ramp_sample**1*np.cos(2*np.pi*2*phase_sample)]
	print(test-output)



def numpy_testing():

	start = timeit.default_timer()

	phase = 2
	ramp = 1

	phase_model = (polynomial_function,20)
	ramp_model = (fourier_series, 20)

	model,size = kronecker_generator(phase_model, ramp_model)
	for i in range(1000000):
		result = model(phase,ramp).T
		print(i)

	print(result)

	stop = timeit.default_timer()

	print('Time: ', stop - start)  


def least_squares_testing():
	phase = [72,2,3]
	ramp = [4,54,6]

	output = [1,2,3]

	phase_model = (polynomial_function,2)
	ramp_model = (polynomial_function, 2)

	model, size = kronecker_generator(phase_model, ramp_model)

	xi, _ = least_squares(model, output, phase, ramp)

	print(xi.shape)

if __name__=='__main__':
	#unit_test()
	#numpy_testing()
	least_squares_testing()