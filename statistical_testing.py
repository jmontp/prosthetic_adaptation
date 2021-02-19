
from model_framework import model_loader
from data_generators import get_subject_names

from math import pow, sqrt

import numpy as np 
import matplotlib.pyplot as plt

#Calculate the standard deviation for every point in phase
def standard_deviation_at_phase():

	list_of_dicts = model_loader('gait_fingerprints.pickle')

	gait_fingerprints, expected_model_parameters, personalization_map_dict, regression_matrix_dict, output_map, average_xi_map = tuple(list_of_dicts)

	subject_names = get_subject_names()
	amount_of_subjects = len(subject_names)


	for j, name in enumerate(subject_names):
		
		#Get the information for a particular person
		gait_fingerprint = gait_fingerprints[name]
		expected_model_parameter = expected_model_parameters[name]
		personalization_map = personalization_map_dict[name]
		regression_matrix = regression_matrix_dict[name]
		output = output_map[name]
		average_xi = average_xi_map[name] 


		#Calculate the residual error based on the average xi
		average_xi_residual = output.ravel() - regression_matrix @ average_xi

		#Calcuate the residual error based on the gait figerprint
		individualized_xi = average_xi + personalization_map @ gait_fingerprint

		individialuzed_xi_residual = output.ravel() - regression_matrix @ individualized_xi


		#Bin the residuals based on phase
		
		#Initialize dicionary for bins
		avg_residual_dict = {}
		ind_residual_dict = {}

		phase = 0
		for i in range(150):

			#Create a dictionary for (Residual Squared Sum, Number of people)
			avg_residual_dict[phase] = [0,0]
			ind_residual_dict[phase] = [0,0]

			phase = phase + 1/150

		residual_size = average_xi_residual.shape[0]

		phase = 0

		for i in range(residual_size):
			avg_residual_dict[phase][0] += pow(average_xi_residual[i],2)
			avg_residual_dict[phase][1] += 1

			ind_residual_dict[phase][0] += pow(individialuzed_xi_residual[i],2)
			ind_residual_dict[phase][1] += 1

			phase += 1/150

			if(phase >= 1):
				phase = 0

		#Convert the dictionary into the standard deviation
		avg_std_dev_dic = {phase: sqrt(value[0])/value[1] for (phase, value) in avg_residual_dict.items()}
		ind_std_dev_dic = {phase: sqrt(value[0])/value[1] for (phase, value) in ind_residual_dict.items()}

		#plot?
		phase = list(avg_std_dev_dic.keys())
		up_std_dev =  np.array([avg_std_dev_dic[i] for i in phase])
		p_std_dev =  np.array([ind_std_dev_dic[i] for i in phase])
		diff = up_std_dev - p_std_dev

		# plt.subplot(3, 1, 1)
		# plt.plot(phase, up_std_dev)
		# plt.title('Un-Personalized Xi Standard Deviation')
		# plt.legend(['Phase', 'Standard Deviation'])
		# plt.subplot(3, 1, 2)
		# plt.plot(phase, p_std_dev)
		# plt.title('Personalized Xi Standard Deviation')
		# plt.legend(['Phase', 'Standard Deviation'])
		plt.subplot(amount_of_subjects, 1, j+1)
		plt.plot(phase, diff)
		plt.title('Difference in Standard Deviation')
		plt.legend(['Phase', 'Standard Deviation'])


	plt.show()



if __name__ == '__main__':
	standard_deviation_at_phase()