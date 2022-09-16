#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
#Standard Imports
from calendar import c
import pandas as pd
import numpy as np
import sys

#Relative Imports
from context import kmodel
from context import ekf

from kmodel.model_definition.personal_measurement_function import PersonalMeasurementFunction
from ekf.measurement_model import MeasurementModel
from ekf.dynamic_model import GaitDynamicModel
from ekf.ekf import Extended_Kalman_Filter
from rtplot import client
from generate_simulation_validation_data import generate_data

#numpy print configuration
#Only display 3 decimal places
np.set_printoptions(precision=3)
#Use scientific notation when needed
np.set_printoptions(suppress=False)
#Make it so long arrays do not insert new lines
np.set_printoptions(linewidth=10000)
#Make it so that the positive numbers take the same amount as negative numbers
# np.set_printoptions(sign=' ')



def phase_dist(phase_a, phase_b):
    # computes a distance that accounts for the modular arithmetic of phase
    # guarantees that the output is between 0 and .5
    dist_prime = np.abs(phase_a-phase_b)
    return np.square(dist_prime) if dist_prime<.5 else np.square(1-dist_prime)


def simulate_ekf(ekf_instance, 
                 state_validation_data,
                 sensor_validation_data,
                 plot_local=False):

   
        
    
    #Define the number of gait fingerprints
    # num_gait_fingerprints = measurement_model.personal_model.num_gait_fingerprint

    #Get the number of states
    num_states = state_validation_data.shape[1]

    #Get the number of measurements
    num_measurements = sensor_validation_data.shape[1]

    #Real time plotting configuration
    X_AXIS_POINTS = 1000
    plot_1a_config = {
                    'names': ['phase', 'phase_a'],
                    'colors' : ['r']*2,
                    'line_style' : ['']*1 + ['-']*1,
                    #Titles and labels 
                    'title': "Phase",
                    'xlabel': "reading (unitless)",
                    'ylabel': 'Varied',
                    'yrange': [1.5,-0.5],                
                    'xrange':X_AXIS_POINTS
                    }
    plot_1b_config = {
                    'names':['phase_dot','stride_length','phase_dot_a','stride_length_a'],
                    'colors' : ['g','b']*2,
                    'line_style' : ['']*2 + ['-']*2,
                    #Titles and labels 
                    'title': "Phase Dot, Stride Length",
                    'xlabel': "reading (unitless)",
                    'ylabel': 'Varied',
                    'yrange': [0.2,1.5],
                    'xrange':X_AXIS_POINTS

                    }
    plot_2_config = {
                    'names': ['ramp', 'ramp_a'],
                    'colors': ['r','r'],
                    'line_style':['','-'],
                    #Titles and labels 
                    'title': "Ramp",
                    'ylabel': "reading (unitless)",
                    'xlabel': 'Degree Incline (Deg)',
                    'yrange': [-10,10],
                    'xrange':X_AXIS_POINTS
                    }

    # plot_3_config = {
    #                 'names': [f"gf{i+1}" for i in range(num_gait_fingerprints)] 
    #                     + [f"gf{i+1}_optimal" for i in range(num_gait_fingerprints)],
    #                 'colors' : (['r','g','b','c','m','brown','orange','yellow', 'purple'][:num_gait_fingerprints])*2,
    #                 'line_style' : ['']*num_gait_fingerprints + ['-']*num_gait_fingerprints,
    #                 #Titles and labels 
    #                 'title': "Gait Fingerprint Vs Expected Gait Fingerprint",
    #                 'ylabel': "reading (unitless)",
    #                 'xlabel': 'STD Deviation',
    #                 'yrange': [-20,20],
    #                 'xrange':X_AXIS_POINTS
    #                 }

    plot_4_config = {
                    'names' : ['True Thigh Angle', 
                               'True Shank Angle', 
                               'True Foot Angle', 
                               'Predicted Thigh Angle',
                               'Predicted Shank Angle',
                               'Predicted Foot Angle'],
                    'colors' : ['r']*6,
                    'line_style': ['-']*3 + ['']*3,
                    'title' : "Measured vs Predicted Joint Angles",
                    'ylabel': "Joint Angle (deg)",
                    'yrange': [-180,180],
                    'xrange':X_AXIS_POINTS
                    }
    plot_5_config = {
                    'names' : ['True Thigh Angular Velocity',
                               'True Shank Angular Velocity',
                               'True Foot Angular Velocity', 
                               'Predicted Thigh Angle Velocity',
                               'Predicted Shank Angle Velocity',
                               'Predicted Foot Angular Velocity'],
                    'colors' : ['b']*6,
                    'line_style': ['']*3 + ['-']*3,
                    'title' : "Measured vs Predicted Joint Angle",
                    'ylabel': "Joint Angle Velocity (deg)",
                    'yrange': [-700,700],
                    'xrange':X_AXIS_POINTS
                    }
    
    #Do a local plot
    if(plot_local == True):

        client.local_plot()
        
     
        client.initialize_plots([plot_1a_config,
                        plot_1b_config,
                        plot_2_config,
                        # plot_3_config,
                        # plot_4_config,
                        # plot_5_config,
                        ])

    
    #Get the total amount of data points
    points_per_step = 150
    total_datapoints = state_validation_data.shape[0]
    

    #Calculate the time step based on the fact that phase_dot = dphase/dt
    #And that dphase = 150 from the fact that we are using the normalized dataset
    # dt = dt/dphase * dphase
    time_step = (np.reciprocal(state_validation_data[:,1])*1/150).reshape(-1)


    #Get the least squared estimated gait fingerprint
    # ls_gf = model.kmodels[0].subject_gait_fingerprint

    #Initialize RMSE
    error_squared_acumulator = np.zeros((num_states))
    output_moment_accumulator = 0
    measurement_error_acumulator = np.zeros((num_measurements))

    #State buffer
    predicted_state_buffer = np.zeros((total_datapoints,num_states))

    #Iterate through all the datapoints
    for i in range(total_datapoints):
        
        #Get the current joint angles
        curr_data = sensor_validation_data[i].reshape(-1,1)

        #Calculate the next state with the ekf
        next_state,predicted_covar = \
            ekf_instance.calculate_next_estimates(time_step[i], curr_data)
            
        next_state = next_state.T
        # next_output = ekf_instance.get_output()

        #Store predicted state in buffer
        predicted_state_buffer[i,:] = next_state[0,:num_states].copy()

        #Get the predicted measurements
        predicted_measurements = ekf_instance.calculated_measurement_
        calculated_angles = ekf_instance.calculated_measurement_[:3]
        calculated_speeds = ekf_instance.calculated_measurement_[3:6]

        #Get ground truth state
        actual_state = state_validation_data[i,:num_states]

        #Track RMSE after certain amount of steady state steps
        track_rmse_after_steps = 20
        if i > track_rmse_after_steps*points_per_step:
            #Add to the rmse accumulator
            error_squared_acumulator[0] += phase_dist(next_state[0,0], 
                                                      actual_state[0])
            error_squared_acumulator[1:] += np.power(next_state[0,1:num_states]
                                                     -actual_state[1:], 2)

            #Add to the measurement output acumulator
            measurement_error_acumulator += np.power(predicted_measurements - curr_data, 2).reshape(-1)

            #Add the error for torque
            # output_moment_accumulator += np.power(next_output - multiple_step_ground_truth[i,-1],2)

        #Decide to plot or not
        if(plot_local == True):

            #Both send measurements in case they are being plotted
            #use subject average does not send gait fingerprints
            plot_array = np.concatenate([next_state[0,0].reshape(-1,1),                    #phase,
                                    state_validation_data[i,0].reshape(-1,1),                  #phase, 
                                    next_state[0,1:3].reshape(-1,1),                   # phase_dot, stride_length 
                                    state_validation_data[i,1:3].reshape(-1,1),                # phase_dot, stride_length from dataset
                                    next_state[0,3].reshape(-1,1),                     #ramp
                                    state_validation_data[i,3].reshape(-1,1) ,                 #ramp from dataset
                                    curr_data[:3].reshape(-1,1),                       #Sensor Angle Data
                                    calculated_angles.reshape(-1,1),                   #Predicted Angle 
                                    curr_data[3:].reshape(-1,1),                       #Sensor Angle Velocity Data
                                    calculated_speeds.reshape(-1,1)                    #Predicted Velocity 

                                    ])
    
        
            client.send_array(plot_array)
        
        # #Save steady state covariance
        # if(i == 30000):
        #     if(use_subject_average == True):
        #         np.save('pred_cov_avg',ekf_instance.predicted_covariance)
        #     else:
        #         np.save('pred_cov_gf',ekf_instance.predicted_covariance)
        
        
        sys.stdout.write("\033[K")
        # print((f'\r{i} out of {total_datapoints}'
        #        f' state rmse {np.sqrt(error_squared_acumulator/(i+1))}'
        #        f' measurement rmse {np.sqrt(measurement_error_acumulator/(i+1))}'
        #     #    f' state {next_state}'
        #     #    f' expected state {multiple_step_ground_truth[i,:4]}'
        #     #    f' expected gf {ls_gf}'
        #         # f' predicted covariance {np.diagonal(predicted_covar)}'
        #     )
        #        ,end="")
        

    #Print a new line since the current status text is supposed to be in line
    # print("\n\r")

    #Calculate the rmse
    rmse = np.sqrt(error_squared_acumulator/total_datapoints)

    #Calculate the measurement rmse
    measurement_rmse = np.sqrt(measurement_error_acumulator/(i+1))
    

    #Get the test id
    test_id = np.load("test_id.npy")
    # client.save_plot(f"{subject}_TestID_{test_id:04}")

    return rmse, measurement_rmse, track_rmse_after_steps

