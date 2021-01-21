"""
Model Generator 

This code is meant to generate the regressor model based on a Kronecker Product of different function which each are one dimensional functions of task variables. 


"""

#H is the partial derivative of the model with respect to the state variables and the pca axis coefficient variables 
#The partial derivative with respect to the state variable is the derivative of the function row for the row function vector for that particular state variable kroneckerd with he other normal funcitons
#The partial derivative with respect to he pca axis is just the pca axis times the kronecker productl


import numpy as np
import math
import pickle




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

	def evaluate_conditional(self,x,apply_derivative):
		if(apply_derivative == True):
			return self.evaluate_derivative(x)
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
		result = [math.pow(x,i) for i in range(0,self.n)]
		return np.array(result)

	#This function will evaluate the derivative of the model at the given 
	# x value
	def evaluate_derivative(self,x):
		result = [i*math.pow(x,(i-1)) for i in range(0,self.n)]
		return np.array(result)


#This will create a Polynomial Basis with n harmonic frequencies
# The variable name is also needed
class Fourier_Basis(Basis):
	def __init__(self, n, var_name):
		Basis.__init__(self, n, var_name)
		self.size = 2*n-1

	#This function will evaluate the model at the given x value
	def evaluate(self,x):
		result = [1]
		result += ([np.cos(2*np.pi*i*x) for i in range(1,self.n)])
		result += ([np.sin(2*np.pi*i*x) for i in range(1,self.n)]) 
		return np.array(result)


	#This function will evaluate the derivative of the model at the given 
	# x value
	def evaluate_derivative(self,x):
		result = [0]
		result += ([-2*np.pi*i*np.sin(2*np.pi*i*x) for i in range(1,self.n)])
		result += ([2*np.pi*i*np.cos(2*np.pi*i*x) for i in range(1,self.n)]) 
		return np.array(result)



#----------------------------------------
#Not really using this right now so keeping in the old format
def berstein_basis(n,var_name):

	def b_func(x):
		basis = [math.comb(n,i)*(1-x)**(n-i) for i in range(0,n+1)];
		return np.array(basis)

	b_func.parent_function = 'berstein_basis'
	b_func.params = n
	b_func.size = n
	b_func.variable_name = var_name


	return b_func
#----------------------------------------


#Model Object:
# list of basis objects
# string description
# model_size
# pca_axis - list of np array of model_size length
# coefficient_list - list of coefficients for each pca_axis
#--------------------------
class Kronecker_Model:
	def __init__(self, *funcs):
		self.funcs = funcs

		#Calculate the size of the parameter array
		#Additionally, pre-allocate arrays for kronecker products intermediaries 
		# to speed up results
		self.alocation_buff = []
		size = 1;
		for func in funcs:
			#Since we multiply left to right, the total size will be on the left 
			#and the size for the new row will be on the right
			print((str(size), str(func.size)))
			self.alocation_buff.append(np.zeros((size, func.size)))

			size = size * func.size

		self.size = size
		self.pca_axis = []
		self.pca_coefficients = []
        #Todo: Add average pca coefficient
        
    
	#Evaluate the models at the function inputs that are received
	#The function inputs are expected in the same order as they where defined
	#Alternatively, you can also input a dictionary with the var_name as the key and the 
	# value you want to evaluate the function as the value
	def evaluate(self, *function_inputs,partial_derivative=None):
		
		#Verify that you have the correct input 
		if(len(function_inputs) != len(self.funcs)):
			err_string = 'Wrong amount of inputs. Received:'  + str(len(function_inputs)) + ', expected:' + str(len(self.funcs))
			raise ValueError(err_string)

		#if(isinstance(function_inputs,dict) == False and isinstance(function_inputs,list) == False): 
		#	raise TypeError("Only Lists and Dicts are supported, you used:" + str(type(function_inputs)))

		#There are two behaviours: one for list and one for dictionary
		#List expects the same order that you received it in
		#Dictionary has key values for the function var names

		result = np.array([1])
		#Assume that you get a list which means that everything is in order
		for values in zip(function_inputs,self.funcs,self.alocation_buff):
			curr_val, curr_func, curr_buf = values
			
			#If you get a dictionary, then get the correct input for the function
			if( isinstance(function_inputs,dict) == True):
				#Get the value from the var_name in the dictionary
				curr_val = function_inputs[curr_func.var_name]

			#Verify if we want to take the partial derivative of this function
			if(partial_derivative is not None and curr_func.var_name in partial_derivative):
				apply_derivative = True
			else: 
				apply_derivative = False

			#Since there isnt an implementation for doing kron in one shot, do it one by one
			result = fast_kronecker(result,curr_func.evaluate_conditional(curr_val,apply_derivative), curr_buf)

		return result

	#Todo: Need to add functionality for the models pca_axis list
	#Dont know if I want to have it run least squares in the initialization
	def set_pca_axis(self,pca_axis):
		self.pca_axis = pca_axis
		self.pca_coefficients = [0]*len(self.pca_axis)

	def set_pca_coefficients(self,pca_coefficients):
		self.pca_coefficients = pca_coefficients

	def sum_pca_axis(self):
		return sum([axis*coeff for axis,coeff in zip(self.pca_axis,self.pca_coefficients)])

	def evaluate_scalar_output(self,*function_inputs,partial_derivative=None):
		return self.evaluate(*function_inputs,partial_derivative=partial_derivative) @ self.sum_pca_axis().T


#Evaluate model 
def model_prediction(model,ξ,*input_list,partial_derivative=None):
	result = [model.evaluate(*function_inputs,partial_derivative=partial_derivative)@ξ for function_inputs in zip(*input_list)]
	return np.array(result)


class Measurement_Model():
	def __init__(self,*models):
		self.models = models

	def evalulate_H_func(self,*inputs):
		#get the output
		result = [model.evaluate_scalar_output(*inputs) for model in self.models]
		return np.array(result)

	def evaluate_dH_func(self,*inputs):
		result = []
		for model in self.models:
			state_derivatives = [model.evaluate_scalar_output(*inputs,partial_derivative=func.var_name) for func in model.funcs]
			gait_fingerprint_derivatives = [model.evaluate(*inputs)@axis for axis in model.pca_axis]
			total_derivatives = state_derivatives + gait_fingerprint_derivatives
			result.append(total_derivatives)

		return np.array(result)


#Calculate the least squares based on the data
def least_squares(model, output, *data):
	
	#Get data size
	rows = data[0].shape[0]
	columns = model.size
	buffer_shape = (rows, columns)

	#Initialize a buffer array
	regressor_matrix = np.zeros(buffer_shape)

	#Create the regressor matrix by evaluating the zipped data
	counter = 0
	for row in zip(*data):
		regressor_matrix[counter,:] = model.evaluate(*row)
		counter = counter + 1

	#Rename
	R = regressor_matrix

	#Only make it a np array if it isnt
	if isinstance(output,(np.ndarray)):
		output = np.array(output)

	return np.linalg.solve(R.T @ R, R.T @ output), R


def model_saver(model,filename):
	with open(filename,'wb') as file:
		pickle.dump(model,file)

def model_loader(filename):
	with open(filename,'rb') as file:
		return pickle.load(file)

def fast_kronecker(a,b,buff=None):
	#If you pass the buffer is the fast implementation
	#139 secs with 1 parameter fit
	if(buff is not None):
		return np.outer(a,b,buff).ravel().copy()

	#Else use the default implementation
	#276.738 secs with 1 param
	else:
		return np.kron(a,b)


####################################################################################
#//////////////////////////////////////////////////////////////////////////////////#
####################################################################################

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
	phase = np.array([72,2,3])
	ramp = np.array([4,54,6])

	output = np.array([1,2,3])

	phase_model = Polynomial_Basis(2,'phase')
	ramp_model = Polynomial_Basis(2,'ramp')

	model = Kronecker_Model([phase_model, ramp_model])

	xi, _ = least_squares(model, output, phase, ramp)

	print(xi.shape)

def object_testing():
	phase = Fourier_Basis(5,'phase')
	ramp = Polynomial_Basis(3,'ramp')

	test_num = 2
	phase_eval1 = phase.evaluate(test_num)

	#save basis
	save_file = 'test_file_saver.txt'
	with open(save_file,'wb') as file:
		pickle.dump(phase,file)

	#load the basis
	with open(save_file,'rb') as file:
		phase2 = pickle.load(file)

	import os
	os.remove(save_file)

	phase_eval2 = phase2.evaluate(test_num)

	print("Phase_eval1 = " + str(phase_eval1))
	print("Phase_eval2 = " + str(phase_eval2))


	model = Kronecker_Model([ramp,phase])
	inputs = [1,2]
	inputs_dict = {'phase':2,'ramp': 1}

	model_out1 = model.evaluate(inputs)
	model_out2 = model.evaluate(inputs_dict)

	print("Model with list input vs with dict: " + str(model_out1-model_out2))

	with open(save_file,'wb') as file:
		pickle.dump(model,file)

	#load the basis
	with open(save_file,'rb') as file:
		model2 = pickle.load(file)

	import os
	os.remove(save_file)

	model2_out = model2.evaluate(inputs)
	print("Model recovered vs original model: " + str(model2_out-model_out1))

	model3 = Kronecker_Model([ramp])

	model3_out1 = model3.evaluate([2])
	model3_out2 = model3.evaluate([2],['ramp'])

	print("Model without derivative:" + str(model3_out1))
	print("Model with derivative:" + str(model3_out2))

if __name__=='__main__':
	#object_testing()
	#unit_test()
	#numpy_testing()
	least_squares_testing()