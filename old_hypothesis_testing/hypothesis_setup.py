from data_generators import get_subject_names
from model_framework import Fourier_Basis, Polynomial_Basis, Bernstein_Basis, Kronecker_Model
from model_fitting import generate_regression_matrices, least_squares_r, least_squares_b
from statistical_testing import calculate_f_score

import numpy as np

#Enable print statements
visualize = True

#Get the joint that we want to generate regressors for
joint = 'hip'
#Get the names of the subjects
subject_names = get_subject_names()


#Initialize the model that we are going to base the regressor on
phase_model = Fourier_Basis(5,'phase')
phase_dot_model = Polynomial_Basis(3, 'phase_dot')
ramp_model = Polynomial_Basis(3, 'ramp')
step_length_model = Polynomial_Basis(3,'step_length')
model_hip = Kronecker_Model(phase_dot_model, ramp_model, step_length_model,phase_model)

#Get the regressor matrix
pickle_dict = generate_regression_matrices(model_hip, subject_names, joint)

#Extract the output
output_dict = pickle_dict['output_dict']
regressor_dict = pickle_dict['regressor_dict']
ξ_dict = {}

for subject in subject_names:

	#Calculate the least squares

	#The model-regressor matrix for the subject
	RTR,RTY = regressor_dict[subject]

	#Store the xi for the person
	ξ_dict[subject] = least_squares_b(RTR,RTY)

#Get the average model parameter vector
ξ_avg = sum(ξ_dict.values())/len(ξ_dict.values())