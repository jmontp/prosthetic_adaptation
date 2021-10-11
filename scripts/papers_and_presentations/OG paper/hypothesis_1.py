from statistical_testing import calculate_f_score
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from hypothesis_setup import *


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
rms_error_avg = np.sqrt(np.mean(np.power(residual_avg,2)))
print("RMS Error for average model: " + str(rms_error_avg))


#Calculate the variance for the unrestricted model (individualized model)
residual_ind = Y - est_ind
unrestricted_RSS = np.sum(np.power(residual_ind, 2))
rms_error_ind = np.sqrt(np.mean(np.power(residual_ind,2)))
print("RMS Error for individualized model: " + str(rms_error_ind))


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
fig.add_trace(go.Scatter(x=x, y=plot_estimated_avg_y + 1.9*phase_std_dev_avg_np, name="Estimated Avg Hip Angle + 1.9std dev", fill=None, line_color='green'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=plot_estimated_avg_y - 1.9*phase_std_dev_avg_np, name="Estimated Avg Hip Angle - 1.9std dev", fill='tonexty', line_color='green'), row=1, col=1)


#Individualized Estimate
fig.add_trace(go.Scatter(x=x, y=plot_estimated_ind_y, name="Estimated Ind Hip Angle", line_color='blue'), row=2, col=1)
#Std deviation fill
fig.add_trace(go.Scatter(x=x, y=plot_estimated_ind_y + 1.9*phase_std_dev_ind_np, name="Estimated Ind Hip Angle + 1.9std dev", fill=None, line_color='blue'), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=plot_estimated_ind_y - 1.9*phase_std_dev_ind_np, name="Estimated Ind Hip Angle - 1.9std dev", fill='tonexty', line_color='blue'), row=2, col=1)




fig.show()