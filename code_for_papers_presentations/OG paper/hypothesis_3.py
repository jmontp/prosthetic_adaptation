from data_generators import get_subject_names
from model_framework import Fourier_Basis, Polynomial_Basis, Bernstein_Basis, Kronecker_Model
from model_fitting import generate_regression_matrices, least_squares_r, get_personalization_map, calculate_gait_fingerprints
from statistical_testing import calculate_f_score

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

#Enable print statements
visualize = True

#Get the joint that we want to generate regressors for
joint = 'hip'
#Get the names of the subjects
subject_names = get_subject_names()


#Initialize the model that we are going to base the regressor on
phase_model = Fourier_Basis(4,'phase')
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

	#The expected hip value for the subject
	Y = output_dict[subject]
	#The model-regressor matrix for the subject
	R = regressor_dict[subject]

	#Store the xi for the person
	ξ_dict[subject] = least_squares_r(Y,R)

#Get the average model parameter vector
ξ_avg = sum(ξ_dict.values())/len(ξ_dict.values())


#Set the number of gait fingerprints that we are going to use
η = 3

#Initialize dict of f_scores
f_core_dict = {}

#Do leave one out validation
for subject in subject_names:
	#Get the personalization map without the subject
	personalization_map = get_personalization_map(ξ_dict, regressor_dict, subject_names, subject_to_leave_out=subject)

	#Get the gait fingerprint for the subject
	gait_fingerprint = calculate_gait_fingerprints(η,regressor_dict[subject],personalization_map, ξ_avg, output_dict[subject])

	#Calculate the variance for the restricted model (average gait model)
	residual_avg = output_dict[subject] - regressor_dict[subject]@ξ_avg
	restricted_RSS = np.sum(np.power(residual_avg, 2))

	#Calculate the variance for the specific model
	residual_ind = output_dict[subject] - regressor_dict[subject]@(ξ_avg + personalization_map[:,:η] @ gait_fingerprint)
	unrestricted_RSS = np.sum(np.power(residual_ind, 2))

	#Calculate p1 and p2 based on the f-score 
	#Not sure how to set this up tbh
	p2 = η
	p1 = 0

	#Get the number of samples that we have
	n = output_dict[subject].shape[0]

	#Calculate the f_score
	f_score, critical_f_score = calculate_f_score(unrestricted_RSS, restricted_RSS, p1, p2, n, visualize)

	#Save for plotting
	f_core_dict[subject] = (f_score, critical_f_score)



