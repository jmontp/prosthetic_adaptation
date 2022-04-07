#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
#Standard Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from time import sleep


#Relative Imports
from context import kmodel
from context import ekf
from context import utils
from kmodel.personalized_model_factory import PersonalizedKModelFactory
from rtplot import client
from ekf.measurement_model import MeasurementModel
from ekf.dynamic_model import GaitDynamicModel
from ekf.ekf import Extended_Kalman_Filter
from kmodel.kronecker_model import model_loader
from utils.math_utils import get_rmse
from rtplot import client

#Import low pass filtering for speed filtering
from scipy.signal import butter, lfilter, firwin


#numpy print configuration
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


#Import the personalized model 
factory = PersonalizedKModelFactory()

subject_model = "AB01"

model_dir = f'../../data/kronecker_models/left_one_out_model_{subject_model}.pickle'

model = factory.load_model(model_dir)

measurement_model = MeasurementModel(model,calculate_output_derivative=True)


#Define the number of gait fingerprints
num_gait_fingerprints = measurement_model.personal_model.num_gait_fingerprint


def simulate_ekf():
    pass
#%%
#Real time plotting configuration
plot_1_config = {#Trace Config
                 'names': ['phase', 'phase_dot', 'stride_length','phase_a','phase_dot_a','stride_length_a'],
                 'colors' : ['r','g','b']*2,
                 'line_style' : ['']*3 + ['-']*3,
                #Titles and labels 
                 'title': "States",
                 'ylabel': "reading (unitless)",
                 'xlabel': 'Varied',
                 'yrange': [2.5,-0.5]
                 }

plot_2_config = {#Trace Config
                'names': ['ramp', 'ramp_a'],
                'colors': ['r','r'],
                'line_style':['','-'],
                #Titles and labels 
                'title': "Ramp",
                'ylabel': "reading (unitless)",
                'xlabel': 'Degree Incline (Deg)',
                'yrange': [-10,10]
                }

plot_3_config = {#Trace Config
                'names': [f"gf{i+1}" for i in range(num_gait_fingerprints)] 
                       + [f"gf{i+1}_optimal" for i in range(num_gait_fingerprints)],
                'colors' : (['r','g','b','c','m'][:num_gait_fingerprints])*2,
                'line_style' : ['']*num_gait_fingerprints + ['-']*num_gait_fingerprints,
                #Titles and labels 
                'title': "Gait Fingerprint Vs Expected Gait Fingerprint",
                'ylabel': "reading (unitless)",
                'xlabel': 'STD Deviation',
                'yrange': [-10,7]
                }
# No joint velocity for now
plot_4_config = {
                'names' : ['meas knee vel', 'meas ankle vel', 'meas hip vel', 'pred knee vel', 'pred ankle vel', 'pred hip vel'],
                'colors' : ['r','b','g']*2,
                'line_style': ['']*3 + ['-']*3,
                'title' : "Measured vs Predicted Joint Angles",
                'ylabel': "Joint Angle (deg)",
                'yrange': [-180,180]
}

plot_5_config = {
                'names' : ['meas knee', 'meas ankle', 'meas hip', 'pred knee', 'pred ankle', 'pred hip'],
                'colors' : ['r','b','g']*2,
                'line_style': ['-']*3 + ['']*3,
                'title' : "Measured vs Predicted Joint Angle",
                'ylabel': "Joint Angle Velocity (deg)",
                'yrange': [-200,300]
}

client.local_plot()
client.initialize_plots([plot_1_config,
                         plot_2_config, 
                         plot_3_config, 
                         plot_4_config, 
                         plot_5_config
                         ])



#Define the joints that you want to import 
model_names = measurement_model.output_names
num_models = len(model_names)

#Initial State
initial_state_list = [0, #Phase
                      1, #Phase_dot
                      1.4, #ramp
                      0, #stride
                      ] + [0]*num_gait_fingerprints
#Convert to numpy array
initial_state = np.array(initial_state_list).reshape(-1,1)

#Generate the initial covariance as being very low
#TODO - double check with gray if this was the strategy that converged or not
cov_diag = 1
initial_state_diag = [cov_diag,cov_diag,cov_diag,cov_diag] + [cov_diag]*num_gait_fingerprints
initial_state_covariance = np.diag(initial_state_diag)


#Set state limits
# upper_limits = np.array([np.inf, 1.4, 2.0, 11] + [np.inf]*num_gait_fingerprints).reshape(-1,1)
# lower_limits = np.array([-np.inf, 0.6, 0.8, -11] + [-np.inf]*num_gait_fingerprints).reshape(-1,1)
upper_limits = np.array([np.inf, np.inf, 2, 20] + [10]*num_gait_fingerprints).reshape(-1,1)
lower_limits = np.array([-np.inf, 0, 0,-20] + [-10]*num_gait_fingerprints).reshape(-1,1)

#Process noise
#Phase, Phase, Dot, Stride_length, ramp, gait fingerprints
gf_var = 1e-15
phase_var = 1e-20
phase_dot_var = 5e-6
stride_length_var = 5e-6
ramp_var = 5e-4

#I like this cal
# gf_var = 0
# phase_var = 1e-20
# phase_dot_var = 5e-6
# stride_length_var = 5e-6
# ramp_var = 5e-4
#r_diag = [250]*int(num_models/2) + [400]*int(num_models/2)


q_diag = [phase_var,phase_dot_var,stride_length_var,ramp_var] + [gf_var*(100**i) for i in range (num_gait_fingerprints)]
Q = np.diag(q_diag)

 #Measurement covarience, Innovation
r_diag = [250]*int(num_models/2) + [400]*int(num_models/2)
R = np.diag(r_diag)

### Load the datasets

#File relative imports
# file_location = '../../data/r01_dataset/r01_Streaming_flattened_{}.parquet'
file_location = "../../data/flattened_dataport/dataport_flattened_partial_{}.parquet"

#Get the file for the corresponding subject
filename = file_location.format(subject_model)
print(f"Looking for {filename}")

#Read in the parquet dataframe
total_data = pd.read_parquet(filename)
print(total_data.columns)


#Phase, Phase Dot, Ramp, Step Length, 5 gait fingerprints
state_names = ['phase', 'phase_dot', 'stride_length', 'ramp',
                'gf1', 'gf2','gf3']


#Get the joint data to play back
joint_data = total_data[model_names]

#Get the ground truth from the datasets
ground_truth_labels = ['phase','phase_dot','stride_length','ramp']
ground_truth = total_data[ground_truth_labels]

#Initiailze gait dynamic model
d_model = GaitDynamicModel()

#Initialize the EKF instance
ekf_instance = Extended_Kalman_Filter(initial_state,initial_state_covariance, d_model, Q, measurement_model, R, 
                                      lower_state_limit=lower_limits, upper_state_limit=upper_limits
                                      )

############ Setup - Data Segments
### Setup data per section constants
trials = 5                      #How many trials
steps_per_trial = 25            #Steps per trial
points_per_step = 150               
points_per_trial = points_per_step * steps_per_trial
experiment_point_gap = 107 * points_per_step
#Skip x amount of steps from the start of the experiments
skip_steps = 200
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


#Clip the data
upper_measurement_bound = [np.inf,np.inf,np.inf,500,500,500]
lower_measurment_bound = [-np.inf,-np.inf,-np.inf,-500,-500,-500]

multiple_step_data = np.clip(multiple_step_data,lower_measurment_bound,upper_measurement_bound)

#Well really, just do all the data
multiple_step_data = joint_data.values
multiple_step_ground_truth = ground_truth.values
total_datapoints = multiple_step_data.shape[0]

#Calculate the time step based on the fact that phase_dot = dphase/dt
#And that dphase = 150 from the fact that we are using the normalized dataset
# dt = dt/dphase * dphase
time_step = (np.reciprocal(multiple_step_ground_truth[:,1])*1/150).reshape(-1)


### Low pass filter the velocity signals
#Set up filter frequency response
# order = 6
# fs = 150
# nyq =  0.5 * fs
# cutoff = 70
# normal_cutoff = cutoff / nyq

# b,a = butter(order, normal_cutoff, analog=False)

# multiple_step_data[:,3:] = lfilter(b, a, multiple_step_data[:,3:])


#Get the least squared estimated gait fingerprint
#TODO: I don't think I'm getting the correct gait fingerprint. Is it the same for all the models? 
ls_gf = model.kmodels[0].subject_gait_fingerprint

try:
    for i in range(total_datapoints):
        curr_data = multiple_step_data[i].reshape(-1,1)

        next_state = ekf_instance.calculate_next_estimates(time_step[i], curr_data)[0].T
        
        calculated_angles = ekf_instance.calculated_measurement_[:3]
        calculated_speeds = ekf_instance.calculated_measurement_[3:6]

        plot_array = np.concatenate([next_state[0,:3].reshape(-1,1),                    #phase, phase_dot, stride_length 
                                     multiple_step_ground_truth[i,:3].reshape(-1,1),    #phase, phase_dot, stride_length from dataset
                                     next_state[0,3].reshape(-1,1),                     #ramp
                                     multiple_step_ground_truth[i,3].reshape(-1,1) ,    #ramp from dataset
                                     next_state[0,4:].reshape(-1,1),                    #gait fingerprints
                                     ls_gf.reshape(-1,1),                                #gait fingerprints from least squares 
                                     curr_data[:3].reshape(-1,1),
                                     calculated_angles.reshape(-1,1),
                                     curr_data[3:].reshape(-1,1),
                                     calculated_speeds.reshape(-1,1)

                                    ])
    
        client.send_array(plot_array)
        print(f'{i} out of {total_datapoints} state {next_state} expected state {multiple_step_ground_truth[i,:4]} expected gf {ls_gf}')
        #sleep(0.01)

except KeyboardInterrupt:
    print(f"Interrupted at step {i}")

    pass




if __name__=='__main__':

    simulate_ekf()
    