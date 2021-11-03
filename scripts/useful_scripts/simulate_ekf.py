#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
#Standard Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

#Relative Imports
from context import kmodel
from context import ekf
from context import utils
from context import rtplot
from ekf.measurement_model import MeasurementModel
from ekf.dynamic_model import GaitDynamicModel
from ekf.ekf import Extended_Kalman_Filter
from ekf.ekf import default_ekf
from kmodel.kronecker_model import model_loader
from kmodel import kronecker_model
from utils.math_utils import get_rmse
from rtplot import client



def simulate_ekf():
    pass
#%%
#Real time plotting configuration

plot_1_config = {#Trace Config
                 'names': ['phase', 'phase_dot', 'stride_length','phase_a','phase_dot_a','stride_length_a'],
                 'colors' : ['r','g','b']*2,
                 'line_style' : ['']*5 + ['-']*5,
                #Titles and labels 
                 'title': "States",
                 'ylabel': "reading (unitless)",
                 'xlabel': 'Varied'}

plot_2_config = {#Trace Config
                'names': ['ramp', 'ramp_a'],
                'colors': ['r','r'],
                'line_style':['','-'],
                #Titles and labels 
                'title': "Ramp",
                'ylabel': "reading (unitless)",
                'xlabel': 'Degree Incline (Deg)'}

plot_3_config = {#Trace Config
                'names': [f"gf{i+1}" for i in range(5)] + [f"gf{i+1}_optimal" for i in range(5)],
                'colors' : ['r','g','b','c','m']*2,
                'line_style' : ['']*5 + ['-']*5,
                #Titles and labels 
                'title': "Gait Fingerprint Vs Expected Gait Fingerprint",
                'ylabel': "reading (unitless)",
                'xlabel': 'STD Deviation'}

client.initialize_plots([plot_1_config,plot_2_config,plot_3_config])
client.wait_for_response()




#Process noise
#Phase, Phase, Dot, Stride_length, ramp, gait fingerprints
gf_var = 1e-6
phase_var = 0
phase_dot_var = 8e-6
stride_length_var = 2e-6
ramp_var = 1e-5
q_diag = [phase_var,phase_dot_var,stride_length_var,ramp_var,
          gf_var,gf_var,gf_var,gf_var,gf_var]
Q = np.diag(q_diag)


#Create the instance for the ekf
ekf_instace = default_ekf(Q=Q)

#Get the diagonal entries of the Q matrix
q_diag = np.diagonal(ekf_instace.Q)

#Get the models to get some info from them to plot 
models = ekf_instace.measurement_model.models

### Load the datasets

#File relative imports
file_location = '../../data/flattened_dataport/dataport_flattened_partial_{}.parquet'
subject_name = 'AB09'
filename = file_location.format(subject_name)
print(f"Looking for {filename}")

#Read in the parquet dataframe
total_data = pd.read_parquet(filename)

#Define the joints that you want to import 
model_names = ['jointangles_hip_dot_x','jointangles_hip_x',
                'jointangles_knee_dot_x','jointangles_knee_x',
                'jointangles_thigh_dot_x','jointangles_thigh_x']


#Define the output names
output_names = ['jointmoment_hip_x', 'jointmoment_knee_x']


#Phase, Phase Dot, Ramp, Step Length, 5 gait fingerprints
state_names = ['phase', 'phase_dot', 'stride_length', 'ramp',
                'gf1', 'gf2','gf3', 'gf4', 'gf5']


#Get the joint data to play back
joint_data = total_data[model_names]


#Get the ground truth from the datasets
ground_truth_labels = ['phase','phase_dot','stride_length','ramp']
ground_truth = total_data[ground_truth_labels]

############ Setup - Data Segments
### Setup data per section constants
trials = 5                      #How many trials
steps_per_trial = 10            #Steps per trial
points_per_step = 150               
points_per_trial = points_per_step * steps_per_trial
experiment_point_gap = 107 * points_per_step
#Skip x amount of steps from the start of the experiments
skip_steps = 15
skip_points = skip_steps * points_per_step
#Make the first section very long to learn gait fingerprint
first_section_steps = 5
first_section_points = first_section_steps * points_per_step# + 75

#Calculate the total number of datapoints
datapoints = points_per_trial * trials + first_section_points
#Pre-allocate memory
multiple_step_data = np.zeros((datapoints,len(model_names)))
multiple_step_ground_truth = np.zeros((datapoints,len(ground_truth_labels)))

#Create the data array based on the setup above
for i in range(-1,trials):
    
    f = first_section_points
    
    if i == -1:
        multiple_step_data[:f, :] = \
            joint_data.iloc[:f, :]
        
        multiple_step_ground_truth[:f, :] = \
            ground_truth.iloc[:f, :]
    else:
        multiple_step_data[(i*points_per_trial) + f : \
                            (i*points_per_trial) + f + points_per_trial , :] = \
            joint_data.iloc[i*experiment_point_gap + skip_points + f:\
                        i*experiment_point_gap + skip_points + f + points_per_trial, :]
            
        multiple_step_ground_truth[i*points_per_trial + f:(i+1)*points_per_trial + f, :] = \
            ground_truth.iloc[i*experiment_point_gap + skip_points + f: i*experiment_point_gap + points_per_trial + skip_points + f, :]


#Repeat the data
repeat_dataset = 5
total_datapoints = datapoints * repeat_dataset

multiple_step_data = np.tile(multiple_step_data, (repeat_dataset,1))
multiple_step_ground_truth = np.tile(multiple_step_ground_truth, (repeat_dataset,1))


#Well really, just do all the data
# multiple_step_data = joint_data.values
# multiple_step_ground_truth = ground_truth.values
# total_datapoints = multiple_step_data.shape[0]

#Calculate the time step based on the fact that phase_dot = dphase/dt
#And that dphase = 150 from the fact that we are using the normalized dataset
# dt = dt/dphase * dphase
time_step = (np.reciprocal(multiple_step_ground_truth[:,1])*1/150).reshape(-1)

#Create storage for state history
# state_history = np.zeros((total_datapoints,len(state_names)))
# real_measurements = np.zeros((total_datapoints, len(model_names)))
# measurement_history = np.zeros((total_datapoints, len(model_names)))
# output_history = np.zeros((total_datapoints, len(output_names)))


#Get the least squared estimated gait fingerprint
ls_gf = models[0].subjects[subject_name]['cross_model_gait_coefficients_unscaled']

try:
    for i in range(total_datapoints):
        curr_data = multiple_step_data[i].reshape(-1,1)
        # real_measurements[i,:] = curr_data.T
        next_state = ekf_instace.calculate_next_estimates(time_step[i], curr_data)[0].T
        # state_history[i,:] = next_state
        # measurement_history[i,:] = ekf_instace.calculated_measurement_.T
        # output_history[i,:] = ekf_instace.get_output().T
        plot_array = np.concatenate([next_state[0,:3].reshape(-1,1),                    #phase, ramp, stride_length 
                                     multiple_step_ground_truth[i,:3].reshape(-1,1),    #phase, ramp, stride_length from dataset
                                     next_state[0,3].reshape(-1,1),                     #ramp
                                     multiple_step_ground_truth[i,3].reshape(-1,1) ,    #ramp from dataset
                                     next_state[0,4:].reshape(-1,1),                    #gait fingerprints
                                     ls_gf.reshape(-1,1)                                #gait fingerprints from least squares 
                                    ])
    
        client.send_array(plot_array)
        client.wait_for_response()
        print(f'{i} out of {total_datapoints}')


except KeyboardInterrupt:
    print(f"Interrupted at step {i}")
    # total_datapoints = i+1
    # curr_data = curr_data[:total_datapoints,:]
    # real_measurements = real_measurements[:total_datapoints,:]
    # state_history = state_history[:total_datapoints,:]
    # measurement_history = measurement_history[:total_datapoints,:]
    # multiple_step_ground_truth = multiple_step_ground_truth[:total_datapoints,:]
    # output_history = output_history[:total_datapoints,:]
    pass

#old plotting    
#print(state_history[:,0])
#plt.plot(state_history[:,:])


# rmse_state = [get_rmse(state_history[:,i],multiple_step_ground_truth[:,i]) for i in range(4)]

# rmse_joint_angle = [get_rmse(real_measurements, measurement_history)]    

# #new plotting
# time_axis = np.arange(total_datapoints)*1/150

# individual_measurements_labels = ['Measured', 'Expected']

# fig,axs = plt.subplots(10,1,sharex=True)

# colors = ['red', 'green', 'orange', 'blue', ]

# axs[-1].set_xlabel("Time (s)")


# #Plot measurements
# for i in range(len(model_names)):
#     #Plot foot angle
#     fig.suptitle(f'Q: {q_diag}  RMSE State: {rmse_state} RMSE angles: {rmse_joint_angle}')
#     axs[i].plot(time_axis, real_measurements[:,i])
#     axs[i].plot(time_axis, measurement_history[:,i])
#     axs[i].set_ylabel(f"{' '.join(model_names[i].split('_')[1:-1])}")
#     axs[i].legend(individual_measurements_labels)

# #Plot the output
# output_index = 6
# axs[output_index].plot(time_axis, output_history)
# axs[output_index].set_ylabel("Torque Outputs")
# axs[output_index].legend(output_names)


# #Plot the gait fingerprints
# gait_fingerprint_index = 7
# custom_cycler = (cycler(color=['b', 'g', 'r', 'c', 'm']))
# axs[gait_fingerprint_index].set_prop_cycle(custom_cycler)    

# ls_gf = models[0].subjects[subject_name]['cross_model_gait_coefficients_unscaled']
# desired_gf = np.repeat(ls_gf,time_axis.shape[0], axis=1).T
# axs[gait_fingerprint_index].plot(time_axis, desired_gf, ":", lw=2)
# axs[gait_fingerprint_index].plot(time_axis, state_history[:,4:], lw=2)
# axs[gait_fingerprint_index].set_ylabel("Gait Fingerprints")
# axs[gait_fingerprint_index].legend([f'gf{i}' for i in range(1,6)])


# #Plot the estimated state and ground truth state
# ground_truth_index = 8
# custom_cycler = (cycler(color=['b', 'g', 'r']))
# axs[ground_truth_index].set_prop_cycle(custom_cycler)  

# axs[ground_truth_index].plot(time_axis, state_history[:,:3])
# axs[ground_truth_index].set_ylabel("Estimated State")
# axs[ground_truth_index].plot(time_axis, multiple_step_ground_truth[:,:3], 
#             linestyle = '--')
# axs[ground_truth_index].set_ylabel("Ground Truth")
# axs[ground_truth_index].legend(state_names[:3] + ground_truth_labels[:3])



# #Plot the ground truth states
# ramp_index = 9
# axs[ramp_index].plot(time_axis, state_history[:,3])
# axs[ramp_index].plot(time_axis, multiple_step_ground_truth[:,3])
# axs[ramp_index].set_ylabel("Ramp")
# axs[ramp_index].legend(["Estimated Ramp", "Ground Truth Ramp"])

# plt.show()



if __name__=='__main__':
    pass
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    simulate_ekf()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
    # stats.dump_stats('/content/export-data')



    #%matplotlib qt
    #%load_ext snakeviz
    #%snakeviz main()