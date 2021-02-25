import numpy as np
import math
import pickle

from os import path

from data_generators import get_trials_in_numpy, get_phase_from_numpy, get_phase_dot, get_step_length, get_ramp, get_subject_names


#Create a regressor matrix for the given output
def calculate_regression_matrix(model, *data):
	
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
		#model.evaluate(*row,result_buffer=regressor_matrix[counter])

		counter = counter + 1

	#Rename
	R = regressor_matrix

	return R


#This will create a regression matrix for the subjects
def generate_regression_matrices(model, subject_names, joint):

	#Filename to save the regression matrix
	#filename = "local-storage/"+str(model)+'.pickle'
	filename = str(model)+'.pickle'

	#If it already exists, then just load it in
	if(path.exists(filename)):
		return model_loader(filename)

	#Initialize dictionaries
	output_dict = {}
	regressor_dict = {}
	order_dict = {}

	#Get the input order
	order = model.get_order()

	#Else calculate it by hand
	for subject in subject_names:

		print("Subject " + subject)
		#Add the expected output
		output_dict[subject] = get_trials_in_numpy(subject,joint).ravel()

		#Get the data for the subject
		order_dict['phase'] = get_phase_from_numpy(output_dict[subject]).ravel()
		order_dict['phase_dot'] = get_phase_dot(subject).ravel()
		order_dict['step_length'] = get_step_length(subject).ravel()
		order_dict['ramp'] = get_ramp(subject).ravel()

		#Set the data in order
		data = [order_dict[x] for x in order]

		#Calculate the regression matrix
		regressor_dict[subject] = calculate_regression_matrix(model, *data)

	print("Saving Model...")

	result_dict = {'output_dict': output_dict, 'regressor_dict': regressor_dict}

	model_saver(result_dict, filename)

	return result_dict




def least_squares_r(output, R):
		return np.linalg.solve(R.T @ R, R.T @ output)



#Calculate the least squares based on the data
def least_squares(model, output, *data):
	
	R = calculate_regression_matrix(model, *data)

	#Make the output vector an np array if it isnt
	if isinstance(output,(np.ndarray)):
		output = np.array(output)

	return np.linalg.solve(R.T @ R, R.T @ output), R




#Save the model so that you can use them later
def model_saver(model,filename):
	with open(filename,'wb') as file:
		pickle.dump(model,file)

#Load the model from a file
def model_loader(filename):
	with open(filename,'rb') as file:
		return pickle.load(file)