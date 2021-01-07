"""
Model Generator 

This code is meant to generate the regressor model based on a Kronecker Product of different function which each are one dimensional functions of task variables. 


"""
import numpy as np
from math import comb

#Return a polynomial basis function with n members
def polynomial_basis(n,var_name):
	
	def p_func(x):
		basis = [x**i for i in range(0,n)]
		return np.array(basis)

	p_func.parent_function = 'polynomial_basis'
	p_func.size = n
	p_func.params = n
	p_func.variable_name = var_name
	
	return p_func

def berstein_basis(n,var_name):

	def b_func(x):
		basis = [comb(n,i)*(1-x)**(n-i) for i in range(0,n+1)];
		return np.array(basis)

	b_func.parent_function = 'berstein_basis'
	b_func.params = n
	b_func.size = n
	b_func.variable_name = var_name


	return b_func

def fourier_basis(n,var_name):

	def f_func(x):
		basis_m = [[np.cos(2*np.pi*i*x), np.sin(2*np.pi*i*x)] for i in range(1,n)]
		basis_f = np.array(basis_m).flatten()
		basis = np.insert(basis_f,0,1)
		return basis

	f_func.parent_function = 'fourier_basis'
	f_func.params = n
	f_func.size = 2*n-1
	f_func.variable_name = var_name

	return f_func


#This function will return a Kronecker model of all the basis functions that are inputed
#func_tuples - tuple with a function basis as the first entry and the amount of entries in that function 
def kronecker_generator(*funcs):

	model_description = model_descriptor(*funcs)
	
	#This will calculate the kronecker product based on the basis functions that are passed in 
	def kronecker_builder(*function_inputs):
		result = np.array([1])
		for values in zip(function_inputs,funcs):
			curr_val,curr_func = values
			result = np.kron(result,curr_func(curr_val))
		return result

	#This will serve as the list of input array
	size = 1;
	for func in funcs:
		size = size * func.size

	kronecker_builder.size = size
	kronecker_builder.model_description = model_description


	return kronecker_builder

def model_descriptor(*funcs):
	output = ''

	for func in funcs:
		basis_name = func.parent_function
		basis_number = func.params
		variable_name = func.variable_name
		output+=basis_name+','+str(basis_number)+',' + variable_name +'\n'

	return output

def model_saver(filename,model):
	with open('./'+filename, 'w') as file:
		file.write(model.model_description)
	
def model_loader(filename):
	func_list = [];
	with open('./'+filename, 'r') as file:
		lines = file.readlines()
		for line in lines:
			function, n, name = line.split(',')
			func_list.append(globals()[function](int(n),name))

	print(func_list)
	return kronecker_generator(*func_list)

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

	phase = polynomial_basis(2,'phase')
	ramp = polynomial_basis(2,'ramp')

	print(phase)
	print(ramp)

	model = kronecker_generator(phase, ramp)
	phase_sample = 0.1
	ramp_sample = 0.2
	output = model(phase_sample, ramp_sample)
	output = model(phase_sample, ramp_sample)
	output = model(phase_sample, ramp_sample)
	output = model(phase_sample, ramp_sample)

	print(output)

	#This is the expected kronecker product
	test = [1, ramp_sample, phase_sample, phase_sample*ramp_sample]
	print(test-output)

	model_saver('./SavedModel',model)

	model2 = model_loader('./SavedModel')

	output2 = model2(phase_sample, ramp_sample)
	print(output2)

	print(output-output2)

def numpy_testing():

	start = timeit.default_timer()

	phase = 2
	ramp = 1

	phase_model = (polynomial_basis,20)
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

	phase_model = (polynomial_basis,2)
	ramp_model = (polynomial_function, 2)

	model, size = kronecker_generator(phase_model, ramp_model)

	xi, _ = least_squares(model, output, phase, ramp)

	print(xi.shape)

if __name__=='__main__':
	unit_test()
	#numpy_testing()
	#least_squares_testing()