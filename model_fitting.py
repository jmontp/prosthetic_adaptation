import numpy as np
import math
import pickle
from os import path
from data_generators import get_trials_in_numpy, get_phase_from_numpy, get_phase_dot, get_step_length, get_ramp, get_subject_names

from numba import jit, cuda

import h5py

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
	for i in range(rows):
		#row = [entry[i] for entry in data]
		regressor_matrix[i,:] = model.evaluate(data[0][i],data[1][i],data[2][i],data[3][i])
		#model.evaluate(*row,result_buffer=regressor_matrix[counter])

	#Rename
	R = regressor_matrix

	return R


#This will create a regression matrix for the subjects
def generate_regression_matrices(model, subject_names, joint):


	#Filename to save the regression matrix
	filename = "local-storage/"+str(model)+joint+'.pickle'
	#filename = str(model)+'.pickle'

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





#Calculate the personalization map 
def get_personalization_map(ξs, regressors, subject_names, subject_to_leave_out=None):
	
	parameter_list = []
	G_total = 0
	N_total = 0

	for subject in subject_names:

		if(subject == subject_to_leave_out):
		    #Skip over this person
		    continue

		#Get the regressor for the person
		R_p = regressors[subject]

		#Add to the gramiam and amount of elements
		G_total += R_p.T @ R_p
		N_total += R_p.shape[0]

		#Add to the parameter list for the personalization map
		parameter_list.append(ξs[subject])

	#This is equation eq:inner_regressor in the paper!
	G = G_total/N_total

	#Get the amount of subjects in the personalization map
	amount_of_subjects=len(parameter_list)

	#Create the parameter matrix
	Ξ = np.array(parameter_list)#.reshape(amount_of_subjects,model.size)


	#Verify we are positive semidefinite
	assert(np.linalg.norm(G-G.T)<1e-7)

	#Diagonalize the matrix G as G = OVO
	eig, O = np.linalg.eigh(G)
	V = np.diagflat(eig)

	#Additionally, all the eigenvalues are true
	for e in eig:
	    assert (e>=0)
	    assert( e>0) # pd

	# Verify that it diagonalized correctly G = O (eig) O.T
	assert(np.linalg.norm(G - O @ V @ O.T)< 1e-7 * np.linalg.norm(G)) # passes

	#This is based on the equation in eq:Qdef
	# Q G Q = I
	Q       = sum([O[:,[i]] @ O[:,[i]].T * 1/np.sqrt(eig[i]) for i in range(len(eig))])
	Qinv    = sum([O[:,[i]] @ O[:,[i]].T * np.sqrt(eig[i]) for i in range(len(eig))])

	#Change of basis conversions
	def param_to_orthonormal(ξ):
	    return Qinv @ ξ
	def param_from_orthonormal(ξ):
	    return Q @ ξ
	def matrix_to_orthonormal(Ξ):
	    return Ξ @ Qinv

	#Get the average coefficients
	ξ_avg = np.mean(Ξ, axis=0)

	#Substract the average coefficients
	Ξ0 = Ξ - ξ_avg

	##Todo: The pca axis can also be obtained with pca instead of eigenvalue 
	## decomposition
	#Calculate the coefficients in the orthonormal space
	Ξ0prime = matrix_to_orthonormal(Ξ0)

	#Get the covariance matrix for this
	Σ = Ξ0prime.T @ Ξ0prime / (Ξ0prime.shape[0]-1)

	#Calculate the eigendecomposition of the covariance matrix
	ψinverted, Uinverted = np.linalg.eigh(Σ)

	#Eigenvalues are obtained from smalles to bigger, make it bigger to smaller
	ψs = np.flip(ψinverted)
	Ψ = np.diagflat(ψs)

	#If we change the eigenvalues we also need to change the eigenvectors
	U = np.flip(Uinverted, axis=1)

	#Run tests to make sure that this is working
	assert(np.linalg.norm(Σ - U @ Ψ @ U.T)< 1e-7 * np.linalg.norm(Σ)) # passes
	for i in range(len(ψs)-1):
	    assert(ψs[i] > ψs[i+1])

	#Define the amount principles axis that we want
	#η = num_gait_fingerprints
	η=amount_of_subjects
	pca_axis_array = []

	#Convert from the new basis back to the original basis vectors
	for i in range (0,η):
	    pca_axis_array.append(param_from_orthonormal(U[:,i]*np.sqrt(ψs[i])))

	#Return the personalization map
	return np.array(pca_axis_array).T


#Calculate the gait fingerprints
def calculate_gait_fingerprints(num_gait_fingerprints, regressor_matrix, personalization_map, ξ_avg, expected_output):

	#Calculate the average estimate
	average_estimate = regressor_matrix @ ξ_avg

	#Calculate the new output
	Y = expected_output - average_estimate
	A = regressor_matrix @ personalization_map[:,:num_gait_fingerprints]

	return np.linalg.solve(A.T @ A, A.T @ Y)



#Save the model so that you can use them later
def model_saver(model,filename):
	with open(filename,'wb') as file:
		pickle.dump(model,file)

#Load the model from a file
def model_loader(filename):
	with open(filename,'rb') as file:
		return pickle.load(file)
