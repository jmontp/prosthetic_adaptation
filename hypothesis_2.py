from data_generators import get_subject_names
from model_framework import Fourier_Basis, Polynomial_Basis, Bernstein_Basis, Kronecker_Model
from model_fitting import generate_regression_matrices, least_squares_r, get_personalization_map, calculate_gait_fingerprints
from statistical_testing import calculate_f_score

import numpy as np

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


#Calculate the mean ξ vector
for subject in subject_names:

	#The expected hip value for the subject
	Y = output_dict[subject]
	#The model-regressor matrix for the subject
	R = regressor_dict[subject]

	#Store the xi for the person
	ξ_dict[subject] = least_squares_r(Y,R)

#Get the average model parameter vector
ξ_avg = sum(ξ_dict.values())/len(ξ_dict.values())

#Get the personalization map
personalization_map = get_personalization_map(ξ_dict, regressor_dict, subject_names)
#Initialize the gait fingerprint dictionary 
gait_fingerprint_dict = {}


#Get the gait fingerprint for only one entry
for subject in subject_names:
	gait_fingerprint_dict[subject] = calculate_gait_fingerprints(1,regressor_dict[subject],personalization_map, ξ_avg, output_dict[subject])


#Get the expected output, regressor matrix and optimal vector
#make sure that the order is the same
output_list = [output_dict[subject] for subject in subject_names]
Y = np.concatenate(output_list, axis=0)

#calculate the estimate using the average ξ
est_avg_list  = [regressor_dict[subject]@ξ_avg for subject in subject_names]
est_avg = np.concatenate(est_avg_list, axis=0)

#calculate the estimate using the personalized ξ
est_ind_list = [regressor_dict[subject] @ (ξ_avg + personalization_map[:,:1] @ gait_fingerprint_dict[subject]) for subject in subject_names]
est_ind = np.concatenate(est_ind_list, axis=0)

#Calculate the variance for the restricted model (average gait model)
residual_n0 = Y - est_avg
restricted_RSS = np.sum(np.power(residual_n0, 2))

#Calculate the variance for the unrestricted model (individualized model)
residual_n1 = Y - est_ind
unrestricted_RSS = np.sum(np.power(residual_n1, 2))

#Calculate p1 and p2 based on the f-score 
#Not sure how to set this up tbh
p2 = 1
p1 = 0

#Amount of samples
n = Y.shape[0]

#Get the f-score and the critical f-score
f_score, critical_f_score = calculate_f_score(unrestricted_RSS, restricted_RSS, p1, p2, n, visualize)

#We must have a better f_score for the base case
assert(f_score > critical_f_score)

#Loop until you stop beating the f-test
for i in range(2,10):
	#Use the old restricted model as the unrestricted model
	residual_n0 = residual_n1
	restricted_RSS = np.sum(np.power(residual_n0, 2))

	#Get the new gait fingerprint
	for subject in subject_names:
		gait_fingerprint_dict[subject] = calculate_gait_fingerprints(i,regressor_dict[subject],personalization_map, ξ_avg, output_dict[subject])

	#calculate the estimate using the personalized ξ
	est_n1_list = [regressor_dict[subject] @ (ξ_avg + personalization_map[:,:i] @ gait_fingerprint_dict[subject]) for subject in subject_names]
	est_n1 = np.concatenate(est_n1_list, axis=0)

	#Calculate the variance for the unrestricted model (individualized model)
	residual_n1 = Y - est_n1
	unrestricted_RSS = np.sum(np.power(residual_n1, 2))

	#Calculate p1 and p2 based on the f-score 
	#Not sure how to set this up tbh
	p2 = i+1
	p1 = i

	#Get the f-score and the critical f-score
	f_score, critical_f_score = calculate_f_score(unrestricted_RSS, restricted_RSS, p1, p2, n, visualize)

	if(f_score < critical_f_score):
		print("The critical n is: " + str(i))
		break
	else:
		print("N = " + str(i) + " is statistically significant")