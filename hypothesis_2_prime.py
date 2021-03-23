from model_fitting import get_personalization_map, calculate_gait_fingerprints
from statistical_testing import calculate_f_score
from hypothesis_setup import *
import numpy as np
import scipy as sp




#Get the personalization map
personalization_map = get_personalization_map(ξ_dict, regressor_dict, subject_names) # 135ish x 10 matrix
ξ_avg = ξ_avg # known from hypothesis_setup # 135ish vector
# output_dict[subject]  # dict of 1-D arrays
# regressor_dict[subject] # dict of 36000ish x 135ish matricies

old_subject_names = list(subject_names)
# for q in range(10):
# 	subject_names = old_subject_names[q:q+1] # looks only at subject 1

for q in range(1):
	subject_names = old_subject_names # looks only at subject 1

	RSS_accumulator = [0.0]*12
	N_accumulator = 0
	for subject in subject_names:
		A_big = regressor_dict[subject]
		Y = output_dict[subject] - A_big@ξ_avg
		N_accumulator+= Y.shape[0]
		for n in range(12):
			if n in range(1,11):
				A = A_big @ personalization_map[:,:n]
				x = np.linalg.solve(A.T@A, A.T@Y)
				RSS_accumulator[n] += (Y-A@x).T @ (Y-A@x)
			elif n==0:
				RSS_accumulator[n] += Y.T @ Y
			else:
				assert (n==11)
				x = np.linalg.solve(A_big.T @A_big, A_big.T@output_dict[subject])
				error = output_dict[subject] - A_big@x
				RSS_accumulator[11] += error.T @ error

	# RSS_accumulator has the data necessary for any F-test
	for i in range(10):
		RSS_restricted = RSS_accumulator[i]
		RSS_unrestricted = RSS_accumulator[i+1]
		dof_restricted = len(subject_names) * i
		dof_unrestricted = len(subject_names) * (i+1)

		f_score = ((RSS_restricted-RSS_unrestricted)/(dof_unrestricted-dof_restricted)) / (RSS_unrestricted/(N_accumulator-dof_unrestricted))
		f_test_comparison = sp.stats.f.isf(0.05, dof_unrestricted-dof_restricted, N_accumulator-dof_unrestricted)
		number = 100*(1-(RSS_accumulator[i]-RSS_accumulator[11])/(RSS_accumulator[0]-RSS_accumulator[11]))

		print("subject AB%2d, n=%d: f_score=%.3e, f_test_comparison=%.2f, RSS_restricted = %.3e, percentage %.2f%%"%(
			q+1, i, f_score, f_test_comparison, RSS_restricted, number))
