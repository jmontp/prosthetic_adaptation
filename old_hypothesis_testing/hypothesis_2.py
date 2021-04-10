from model_fitting import get_personalization_map, calculate_gait_fingerprints
from statistical_testing import calculate_f_score
from hypothesis_setup import *
import numpy as np




#Get the personalization map
personalization_map = get_personalization_map(ξ_dict, regressor_dict, subject_names)

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
f_score, critical_f_score = calculate_f_score(unrestricted_RSS, restricted_RSS, p1, p2, n)

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

	print("Restricted RSS: " + str(restricted_RSS) + " unrestricted RSS: " + str(unrestricted_RSS))

	#Get the f-score and the critical f-score
	f_score, critical_f_score = calculate_f_score(unrestricted_RSS, restricted_RSS, p1, p2, n)

	if(f_score < critical_f_score):
		print("The critical n is: " + str(i))
		break
	else:
		print("N = " + str(i) + " is statistically significant")