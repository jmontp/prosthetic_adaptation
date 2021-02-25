from data_generators import get_subject_names
from model_framework import Fourier_Basis, Polynomial_Basis, Bernstein_Basis, Kronecker_Model
from model_fitting import generate_regression_matrices, least_squares_r
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

#Get the expected output, regressor matrix and optimal vector
#make sure that the order is the same
output_list = [output_dict[subject] for subject in subject_names]
Y = np.concatenate(output_list, axis=0)

#calculate the estimate using the average ξ
est_avg_list  = [regressor_dict[subject]@ξ_avg for subject in subject_names]
est_avg = np.concatenate(est_avg_list, axis=0)
#calculate the estimate using the personalized ξ
est_ind_list = [regressor_dict[subject]@ξ_dict[subject] for subject in subject_names]
est_ind = np.concatenate(est_ind_list, axis=0)

#Calculate the variance for the restricted model (average gait model)
residual_avg = Y - est_avg
restricted_RSS = np.sum(np.power(residual_avg, 2))

#Calculate the variance for the unrestricted model (individualized model)
residual_ind = Y - est_ind
unrestricted_RSS = np.sum(np.power(residual_ind, 2))

#Calculate p1 and p2 based on the f-score 
#Not sure how to set this up tbh
p2 = ξ_avg.shape[0]
p1 = 0

#Amount of samples
n = Y.shape[0]

#Get the f-score and the critical f-score
f_score, critical_f_score = calculate_f_score(unrestricted_RSS, restricted_RSS, p1, p2, n, visualize)

x = np.linspace(0,1,150)

#Initialize the std deviation per phase dicts
phase_std_dev_avg = []
phase_std_dev_ind = []

for i in range(150):
	phase_std_dev_avg.append(np.std(residual_avg[i::150])) 
	phase_std_dev_ind.append(np.std(residual_ind[i::150])) 

phase_std_dev_avg_np = np.array(phase_std_dev_avg)
phase_std_dev_ind_np = np.array(phase_std_dev_ind)


##Plotting

plot_expected_y = Y[0:150]

plot_estimated_avg_y = est_avg[0:150]

plot_estimated_ind_y = est_ind[0:150]

#Add two subplots
fig = make_subplots(rows=2, cols=1)

#Add measured y value on both plots
fig.add_trace(go.Scatter(x=x, y=plot_expected_y, name="Measured Hip Angle", line_color='red'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=plot_expected_y, name="Measured Hip Angle", line_color='red'), row=2, col=1)

#Average Estimate
fig.add_trace(go.Scatter(x=x, y=plot_estimated_avg_y, name="Estimated Avg Hip Angle", line_color='green'), row=1, col=1)
#Std deviation fill
fig.add_trace(go.Scatter(x=x, y=plot_estimated_avg_y + 1.9*phase_std_dev_avg_np, name="Estimated Avg Hip Angle", fill='tonexty', line_color='green'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=plot_estimated_avg_y - 1.9*phase_std_dev_avg_np, name="Estimated Avg Hip Angle", fill='tonexty', line_color='green'), row=1, col=1)


#Individualized Estimate
fig.add_trace(go.Scatter(x=x, y=plot_estimated_ind_y, name="Estimated Ind Hip Angle", line_color='blue'), row=2, col=1)
#Std deviation fill
fig.add_trace(go.Scatter(x=x, y=plot_estimated_ind_y + 1.9*phase_std_dev_ind_np, name="Estimated Ind Hip Angle", fill='tonexty', line_color='blue'), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=plot_estimated_ind_y - 1.9*phase_std_dev_ind_np, name="Estimated Ind Hip Angle", fill='tonexty', line_color='blue'), row=2, col=1)




fig.show()