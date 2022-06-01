#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
#Standard Imports
import pandas as pd
import numpy as np


#Relative Imports
from context import kmodel
from context import ekf
from context import utils
from kmodel.personalized_model_factory import PersonalizedKModelFactory
from rtplot import client
from ekf.measurement_model import MeasurementModel
from ekf.dynamic_model import GaitDynamicModel
from ekf.ekf import Extended_Kalman_Filter
from rtplot import client


#numpy print configuration
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def phase_dist(phase_a, phase_b):
    # computes a distance that accounts for the modular arithmetic of phase
    # guarantees that the output is between 0 and .5
    dist_prime = np.abs(phase_a-phase_b)
    return np.square(dist_prime) if dist_prime<.5 else np.square(1-dist_prime)


def simulate_ekf(subject,initial_state, initial_state_covariance, Q, R, 
                state_lower_limit, state_upper_limit, 
                use_subject_average=False,use_ls_gf = False,
                plot_local=False):

    #Import the personalized model 
    factory = PersonalizedKModelFactory()
    
    #Path to model
    model_dir = f'../../data/kronecker_models/left_one_out_model_{subject}.pickle'
    # model_output_dir = f'../../data/kronecker_models/left_one_out_model_{subject}_moment.pickle'

    #Load model from disk
    model = factory.load_model(model_dir)
    # model_output = factory.load_model(model_output_dir)

    #Initialize the measurement model
    measurement_model = MeasurementModel(model,calculate_output_derivative=True)
    # output_model = MeasurementModel(model_output, calculate_output_derivative=False)

    #Define the number of gait fingerprints
    num_gait_fingerprints = measurement_model.personal_model.num_gait_fingerprint

    #Get the number of states
    if use_subject_average == True or use_ls_gf == True:
        num_states = initial_state.size
    else:
        num_states = initial_state.size - num_gait_fingerprints

    #Real time plotting configuration
    plot_1a_config = {
                    'names': ['phase', 'phase_a'],
                    'colors' : ['r']*2,
                    'line_style' : ['']*1 + ['-']*1,
                    #Titles and labels 
                    'title': "Phase",
                    'xlabel': "reading (unitless)",
                    'ylabel': 'Varied',
                    'yrange': [2.5,-0.5]
                    }
    plot_1b_config = {
                    'names':['phase_dot','stride_length','phase_dot_a','stride_length_a'],
                    'colors' : ['g','b']*2,
                    'line_style' : ['']*2 + ['-']*2,
                    #Titles and labels 
                    'title': "Phase Dot, Stride Length",
                    'xlabel': "reading (unitless)",
                    'ylabel': 'Varied',
                    'yrange': [0.8,1.5]
                    }
    plot_2_config = {
                    'names': ['ramp', 'ramp_a'],
                    'colors': ['r','r'],
                    'line_style':['','-'],
                    #Titles and labels 
                    'title': "Ramp",
                    'ylabel': "reading (unitless)",
                    'xlabel': 'Degree Incline (Deg)',
                    'yrange': [-10,10]
                    }

    plot_3_config = {
                    'names': [f"gf{i+1}" for i in range(num_gait_fingerprints)] 
                        + [f"gf{i+1}_optimal" for i in range(num_gait_fingerprints)],
                    'colors' : (['r','g','b','c','m','brown','orange','yellow', 'purple'][:num_gait_fingerprints])*2,
                    'line_style' : ['']*num_gait_fingerprints + ['-']*num_gait_fingerprints,
                    #Titles and labels 
                    'title': "Gait Fingerprint Vs Expected Gait Fingerprint",
                    'ylabel': "reading (unitless)",
                    'xlabel': 'STD Deviation',
                    'yrange': [-20,20]
                    }

    plot_4_config = {
                    'names' : ['True Thigh Angle', 'Predicted Thigh Angle'],
                    'colors' : ['r']*2,
                    'line_style': ['-'] + [''],
                    'title' : "Measured vs Predicted Joint Angles",
                    'ylabel': "Joint Angle (deg)",
                    'yrange': [-180,180]}

    plot_5_config = {
                    'names' : ['True Thigh Angular Velocity', 'Predicted Thigh Angle Velocity'],
                    'colors' : ['r']*2,
                    'line_style': [''] + ['-'],
                    'title' : "Measured vs Predicted Joint Angle",
                    'ylabel': "Joint Angle Velocity (deg)",
                    'yrange': [-700,700]}
    
    #Do a local plot
    if(plot_local == True):

        client.local_plot()
        
        #Don't plot gait fingerprints
        if(use_subject_average == True or use_ls_gf == True):
            client.initialize_plots([plot_1a_config,
                                plot_1b_config,
                                #plot_2_config, 
                                #plot_3_config, 
                                plot_4_config, 
                                plot_5_config,
                                ])
        else:
            client.initialize_plots([plot_1a_config,
                            plot_1b_config,
                            #plot_2_config, 
                            plot_3_config, 
                            plot_4_config, 
                            plot_5_config,
                            ])

    #Define the joints that you want to import 
    model_names = measurement_model.output_names
    num_models = len(model_names)

    ### Load the datasets
    #File relative imports
    # file_location = '../../data/r01_dataset/r01_Streaming_flattened_{}.parquet'
    file_location = "../../data/flattened_dataport/dataport_flattened_partial_{}.parquet"

    #Get the file for the corresponding subject
    filename = file_location.format(subject)
    # print(f"Looking for {filename}")

    #Read in the parquet dataframe
    total_data = pd.read_parquet(filename)
    # print(total_data.columns)

    #Phase, Phase Dot, Ramp, Step Length, 5 gait fingerprints
    state_names = ['phase', 'phase_dot', 'stride_length']

    #Get the ground truth from the datasets
    ground_truth_labels = ['phase','phase_dot','stride_length','ramp','jointmoment_ankle_x']
    
    #Initiailze gait dynamic model
    d_model = GaitDynamicModel()

    #Initialize the EKF instance
    ekf_instance = Extended_Kalman_Filter(initial_state,initial_state_covariance, d_model, Q, measurement_model, R, 
                                        lower_state_limit=state_lower_limit, upper_state_limit=state_upper_limit,
                                        use_subject_average_fit=use_subject_average,
                                        use_least_squares_gf = use_ls_gf,
                                        # output_model=output_model
                                        )


    ############ Setup - Data Segments
    #Get list of conditions (incline, speed)
    condition_list = [
                      *([(0.0, 0.8),(0.0, 1.0),(0.0, 1.2),(0.0, 1.0),(0.0, 0.8)]*1),
                      (0.0, 1.0),(0.0, 0.8),(0.0, 1.0),(0.0, 1.2)
                    #   (-2.5,1.2),
                    #   (-5,1.2),
                    #   (-7.5,1.2),
                    #   (-10,0.8),
                    #   (-7.5,0.8),
                    #   (-5,1.2),
                    #   (-2.5,0.8),
                    #   (0.0, 0.8),
                    #   (2.5,1.2),
                    #   (5,1.2),
                    #   (7.5,1.2),
                    #   (10,0.8),
                    #   (7.5,0.8),
                    #   (5,1.2),
                    #   (2.5,0.8),
                    #   (0.0, 1.2),
                    #   (-7.5,0.8),
                    #   (10,0.8),
                      ]

    condition_list = condition_list * 1

    num_steps_per_condition = 26
    points_per_step = 150
    points_per_condition = num_steps_per_condition * points_per_step
    num_trials = len(condition_list)


    #Calculate the total number of datapoints
    datapoints = points_per_condition * num_trials 

    #Pre-allocate memory
    multiple_step_data = np.zeros((datapoints,len(model_names)))
    multiple_step_ground_truth = np.zeros((datapoints,len(ground_truth_labels)))

    #Skip steps, don't do this by default
    skip = 0
    #this subject's data is very noisy in these conditions, skip
    if(subject == "AB02"):
        skip = 1

    #Create the data array based on the setup above
    for i,condition in enumerate(condition_list):
        
        #Get the desired incline and speed from the condition
        ramp, speed = condition

        #Get the filtered data based on the condition
        mask = (total_data['ramp'] == ramp) & (total_data['speed'] == speed)
        filtered_data = total_data[mask]

        #Get the sensor data
        multiple_step_data[i*points_per_condition: (i+1)*points_per_condition,:] = \
         filtered_data[model_names].values[skip*points_per_condition:(skip+1)*points_per_condition,:]

        #Get the ground truth data
        multiple_step_ground_truth[i*points_per_condition: (i+1)*points_per_condition,:] = \
         filtered_data[ground_truth_labels].values[skip*points_per_condition:(skip+1)*points_per_condition,:]
        
    total_datapoints = datapoints
    


    #Well really, just do all the data
    # multiple_step_data = joint_data.values
    # multiple_step_ground_truth = ground_truth.values
    # total_datapoints = multiple_step_data.shape[0]

    #Calculate the time step based on the fact that phase_dot = dphase/dt
    #And that dphase = 150 from the fact that we are using the normalized dataset
    # dt = dt/dphase * dphase
    time_step = (np.reciprocal(multiple_step_ground_truth[:,1])*1/150).reshape(-1)


    #Get the least squared estimated gait fingerprint
    ls_gf = model.kmodels[0].subject_gait_fingerprint

    #Initialize RMSE
    error_squared_acumulator = np.zeros((num_states))
    output_moment_accumulator = 0

    #Iterate through all the datapoints
    for i in range(total_datapoints):
        
        #Get the current joint angles
        curr_data = multiple_step_data[i].reshape(-1,1)

        #Calculate the next state with the ekf
        next_state = ekf_instance.calculate_next_estimates(time_step[i], curr_data)[0].T
        # next_output = ekf_instance.get_output()

       

        #Get the predicted measurements
        calculated_angles = ekf_instance.calculated_measurement_[:3]
        calculated_speeds = ekf_instance.calculated_measurement_[3:6]

        #Get ground truth state
        actual_state = multiple_step_ground_truth[i,:num_states]

        #Track RMSE after certain amount of steady state steps
        track_rmse_after_steps = 20
        if i > track_rmse_after_steps*points_per_step:
            #Add to the rmse accumulator
            error_squared_acumulator[0] += phase_dist(next_state[0,0], actual_state[0])
            error_squared_acumulator[1:] += np.power(next_state[0,1:num_states]-actual_state[1:], 2)

            #Add the error for torque
            # output_moment_accumulator += np.power(next_output - multiple_step_ground_truth[i,-1],2)


        #Decide to plot or not
        if(plot_local == True):
            num_measurements = 1


            #Both send measurements in case they are being plotted
            #use subject average does not send gait fingerprints
            if (use_subject_average == True or use_ls_gf == True):
                plot_array = np.concatenate([next_state[0,0].reshape(-1,1),                    #phase,
                                            multiple_step_ground_truth[i,0].reshape(-1,1),    #phase, 
                                            next_state[0,1:3].reshape(-1,1),                    # phase_dot, stride_length 
                                            multiple_step_ground_truth[i,1:3].reshape(-1,1),    # phase_dot, stride_length from dataset
                                        #next_state[0,3].reshape(-1,1),                     #ramp
                                        #multiple_step_ground_truth[i,3].reshape(-1,1) ,    #ramp from dataset
                                        curr_data[:num_measurements].reshape(-1,1),
                                        calculated_angles.reshape(-1,1),
                                        curr_data[num_measurements:].reshape(-1,1),
                                        calculated_speeds.reshape(-1,1)

                                        ])
            else:
                 plot_array = np.concatenate([next_state[0,0].reshape(-1,1),                    #phase,
                                            multiple_step_ground_truth[i,0].reshape(-1,1),    #phase, 
                                            next_state[0,1:3].reshape(-1,1),                    # phase_dot, stride_length 
                                            multiple_step_ground_truth[i,1:3].reshape(-1,1),    # phase_dot, stride_length from dataset
                                            #next_state[0,3].reshape(-1,1),                     #ramp
                                            #multiple_step_ground_truth[i,3].reshape(-1,1) ,    #ramp from dataset
                                            next_state[0,num_states:].reshape(-1,1),                    #gait fingerprints
                                            ls_gf.reshape(-1,1),                                #gait fingerprints from least squares 
                                            curr_data[:num_measurements].reshape(-1,1),
                                            calculated_angles.reshape(-1,1),
                                            curr_data[num_measurements:].reshape(-1,1),
                                            calculated_speeds.reshape(-1,1)

                                        ])
        
            client.send_array(plot_array)
        
        #Save steady state covariance
        # if(i == 50000):
        #     np.save('pred_cov',ekf_instance.predicted_covariance)

        print(f'\r{i} out of {total_datapoints} moment rmse {np.sqrt(output_moment_accumulator/(i+1))} rmse {np.sqrt(error_squared_acumulator/(i+1))}',end=" ")# state {next_state} expected state {multiple_step_ground_truth[i,:4]} expected gf {ls_gf}')

    #Calculate the rmse
    rmse = np.sqrt(error_squared_acumulator/total_datapoints)
    
    #
    ls_gf = model.kmodels[0].subject_gait_fingerprint

    return rmse, ls_gf




if __name__=='__main__':

    simulate_ekf()
    