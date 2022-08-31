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
from context import utils
from kmodel.personalized_model_factory import PersonalizedKModelFactory
from rtplot import client
from ekf.measurement_model import MeasurementModel
from ekf.dynamic_model import GaitDynamicModel
from ekf.ekf import Extended_Kalman_Filter
from rtplot import client


#numpy print configuration
#Only display 3 decimal places
np.set_printoptions(precision=3)
#Use scientific notation when needed
np.set_printoptions(suppress=False)
#Make it so long arrays do not insert new lines
np.set_printoptions(linewidth=10000)
#Make it so that the positive numbers take the same amount as negative numbers
# np.set_printoptions(sign=' ')

def low_pass_filter(adata: np.ndarray,
                     bandlimit: int = 20,
                     sampling_rate: int = 100) -> np.ndarray:
        """
        Low pass filter implementation by Iwohlhart
        https://stackoverflow.com/
        questions/70825086/python-lowpass-filter-with-only-numpy

        Keyword Arguments
        adata -- the data that will be filtered
            Data type: np.ndarray with no specific range
        bandlimit -- the bandwidth limit in Hz
            Data type: int
        sampling_rate -- the sampling rate of the data
            Data type: int

        Returns
        filtered_data
            Data type: np.ndarray
        """
        
        # translate bandlimit from Hz to dataindex according to
        # sampling rate and data size
        bandlimit_index = int(bandlimit * adata.size / sampling_rate)
    
        fsig = np.fft.fft(adata)
        
        for fourier_i in range(bandlimit_index + 1,
                               len(fsig) - bandlimit_index ):
            fsig[fourier_i] = 0
            
        adata_filtered = np.fft.ifft(fsig)
    
        return np.real(adata_filtered)

def phase_dist(phase_a, phase_b):
    # computes a distance that accounts for the modular arithmetic of phase
    # guarantees that the output is between 0 and .5
    dist_prime = np.abs(phase_a-phase_b)
    return np.square(dist_prime) if dist_prime<.5 else np.square(1-dist_prime)


def simulate_ekf(subject,initial_state, initial_state_covariance, Q, R,
                state_lower_limit, state_upper_limit,
                use_subject_average=False,use_ls_gf = False, use_optimal_fit=False,calculate_gf_after_step=False,
                plot_local=False,null_space_projection=False,
                heteroschedastic_model=False):

    #Import the personalized model 
    factory = PersonalizedKModelFactory()
    
    #Path to model
    if null_space_projection is True:
        model_dir = f'../../data/kronecker_models/left_one_out_model_{subject}_null.pickle'
    else:
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
    num_states = model.num_states

    #Get the number of measurements
    num_measurements = model.num_kmodels

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
                    'yrange': [0.2,1.2],
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

    plot_3_config = {
                    'names': [f"gf{i+1}" for i in range(num_gait_fingerprints)] 
                        + [f"gf{i+1}_optimal" for i in range(num_gait_fingerprints)],
                    'colors' : (['r','g','b','c','m','brown','orange','yellow', 'purple'][:num_gait_fingerprints])*2,
                    'line_style' : ['']*num_gait_fingerprints + ['-']*num_gait_fingerprints,
                    #Titles and labels 
                    'title': "Gait Fingerprint Vs Expected Gait Fingerprint",
                    'ylabel': "reading (unitless)",
                    'xlabel': 'STD Deviation',
                    'yrange': [-20,20],
                    'xrange':X_AXIS_POINTS
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
                                plot_2_config,
                                #plot_3_config,
                                # plot_4_config,
                                # plot_5_config,
                                ])
        else:
            client.initialize_plots([plot_1a_config,
                            plot_1b_config,
                            plot_2_config,
                            plot_3_config,
                            # plot_4_config,
                            # plot_5_config,
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
    ground_truth_labels = ['phase','phase_dot','stride_length','ramp']
    
    #Initiailze gait dynamic model
    d_model = GaitDynamicModel()

    #Initialize the EKF instance
    ekf_instance = Extended_Kalman_Filter(initial_state,initial_state_covariance, d_model, Q, measurement_model, R, 
                                        lower_state_limit=state_lower_limit, upper_state_limit=state_upper_limit,
                                        use_subject_average_fit=use_subject_average,
                                        use_least_squares_gf = use_ls_gf,
                                        use_optimal_fit=use_optimal_fit,
                                        # output_model=output_model,
                                        heteroschedastic_model = heteroschedastic_model
                                        )


    ############ Setup - Data Segments
    #Get list of conditions (incline, speed)
    condition_list = [
                      (0.0, 0.8),
                      (0.0, 1.0),
                      (0.0, 1.2),
                      (-2.5,1.2),
                      (-5,1.2),
                      (-7.5,1.2),
                      (-10,0.8),
                      (-7.5,0.8),
                      (-5,1.2),
                      (-2.5,0.8),
                      (0.0, 0.8),
                      (2.5,1.2),
                      (5,1.2),
                      (7.5,1.2),
                      (10,0.8),
                      (7.5,0.8),
                      (5,1.2),
                      (2.5,0.8),
                      (0.0, 1.2),
                      (-7.5,0.8),
                      (10,0.8),
                      ]

    condition_list = condition_list * 1

    num_steps_per_condition = 10
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
    measurement_error_acumulator = np.zeros((2*num_measurements))

    #State buffer
    predicted_state_buffer = np.zeros((total_datapoints,num_states))

    #Iterate through all the datapoints
    for i in range(total_datapoints):
        
        #Get the current joint angles
        curr_data = multiple_step_data[i].reshape(-1,1)

        #Calculate the next state with the ekf
        next_state,predicted_covar = ekf_instance.calculate_next_estimates(time_step[i], curr_data)
        next_state = next_state.T
        # next_output = ekf_instance.get_output()

        #Store predicted state in buffer
        predicted_state_buffer[i,:] = next_state[0,:num_states].copy()

        #Get the predicted measurements
        predicted_measurements = ekf_instance.calculated_measurement_
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

            #Add to the measurement output acumulator
            measurement_error_acumulator += np.power(predicted_measurements - curr_data, 2).reshape(-1)

            #Add the error for torque
            # output_moment_accumulator += np.power(next_output - multiple_step_ground_truth[i,-1],2)


        ########################################
        # Calculate offline gait fingerprint 
        ########################################
        if(use_subject_average == True and calculate_gf_after_step == True):
            #Wait until we have at least a few steps worth of data
            calculate_gf_num_steps = 5
            calculate_gf_datapoints = calculate_gf_num_steps*150
            wait_to_stabilize_datapoints = 15*150
            
            #Store the original models
            if (i == 0):
                original_average_vectors = [kmodel.average_fit for kmodel in model.kmodels]
                prev_state = next_state.copy()
                step_counter = 0
           
            #Don't do anythign until you stabalize
            if (i > wait_to_stabilize_datapoints):
                
                #To understand if a step has passed, verify if the predicted state has changed a lot
                delta_phase = prev_state[0,0] - next_state[0,0]
                prev_state = next_state.copy()

                #If there is a big jump in phase, its because a step just happened
                if(abs(delta_phase) > 0.8):
                    step_counter += 1

                if(step_counter > calculate_gf_num_steps):

                    #Reset step counter
                    step_counter = 0
                    # print("")
                    for joint_index,(og_vector,joint_kmodel) in enumerate(zip(original_average_vectors, model.kmodels)):
                        #Calculate g for the last steps
                        #Get the state and data for the last steps
                        # regressor_state = predicted_state_buffer[i-calculate_gf_datapoints:i]
                        regressor_state = low_pass_filter(predicted_state_buffer[i-calculate_gf_datapoints:i])
                        # regressor_state = multiple_step_ground_truth[i-calculate_gf_datapoints:i]
                        regressor_joint_angles = multiple_step_data[i-calculate_gf_datapoints:i,[joint_index]]
                        #Calculate the regressor matrix that corresponds to that solution
                        regressor_matrix = joint_kmodel.model.evaluate(regressor_state)
                        
                        
                        # ## Calculate the regression without personalization map 
                        # A = regressor_matrix
                        # fit = np.linalg.solve(A.T @ A, A.T @ regressor_joint_angles)
                        # joint_kmodel.average_fit = fit.T
                        
                        
                        ##Calculate gait fingerprint of personalization map
                        #Calculate the average fit for that subject
                        regressor_average_estimation = (regressor_matrix @ og_vector.T).reshape(-1,1)
                        diff_from_average = regressor_joint_angles - regressor_average_estimation
                        #Calculate G using least squares
                        A = regressor_matrix @ joint_kmodel.pmap.T
                        g = np.linalg.solve(A.T @ A + 0 * np.eye(A.shape[1]) , A.T @ diff_from_average)
                        # print(f"{joint_kmodel.output_name} g magnitude {g.T}")
                        #Update the average fit to the new subject fit
                        joint_kmodel.average_fit = g.T @ joint_kmodel.pmap + og_vector


                        #Make sure that they are the same
                        assert np.linalg.norm(model.kmodels[0].average_fit - ekf_instance.measurement_model.personal_model.kmodels[0].average_fit) < 1e-7

                        #print(f" Gait fingerprint {g.T}")

        #Decide to plot or not
        if(plot_local == True):
            plot_num_measurements = 1 #Select how many measurements, set to 1 to not transmit data


            #Both send measurements in case they are being plotted
            #use subject average does not send gait fingerprints
            if (use_subject_average == True or use_ls_gf == True):
                plot_array = np.concatenate([next_state[0,0].reshape(-1,1),                    #phase,
                                            multiple_step_ground_truth[i,0].reshape(-1,1),    #phase, 
                                            next_state[0,1:3].reshape(-1,1),                    # phase_dot, stride_length 
                                            multiple_step_ground_truth[i,1:3].reshape(-1,1),    # phase_dot, stride_length from dataset
                                        next_state[0,3].reshape(-1,1),                     #ramp
                                        multiple_step_ground_truth[i,3].reshape(-1,1) ,    #ramp from dataset
                                        curr_data[:plot_num_measurements].reshape(-1,1),
                                        calculated_angles.reshape(-1,1),
                                        curr_data[plot_num_measurements:].reshape(-1,1),
                                        calculated_speeds.reshape(-1,1)

                                        ])
            else:
                 plot_array = np.concatenate([next_state[0,0].reshape(-1,1),                    #phase,
                                            multiple_step_ground_truth[i,0].reshape(-1,1),    #phase, 
                                            next_state[0,1:3].reshape(-1,1),                    # phase_dot, stride_length 
                                            multiple_step_ground_truth[i,1:3].reshape(-1,1),    # phase_dot, stride_length from dataset
                                            next_state[0,3].reshape(-1,1),                     #ramp
                                            multiple_step_ground_truth[i,3].reshape(-1,1) ,    #ramp from dataset
                                            next_state[0,num_states:].reshape(-1,1),                    #gait fingerprints
                                            ls_gf.reshape(-1,1),                                #gait fingerprints from least squares 
                                            curr_data[:plot_num_measurements].reshape(-1,1),
                                            calculated_angles.reshape(-1,1),
                                            curr_data[plot_num_measurements:].reshape(-1,1),
                                            calculated_speeds.reshape(-1,1)

                                        ])
        
            client.send_array(plot_array)
        
        #Save steady state covariance
        if(i == 30000):
            if(use_subject_average == True or use_ls_gf == True):
                np.save('pred_cov_avg',ekf_instance.predicted_covariance)
            else:
                np.save('pred_cov_gf',ekf_instance.predicted_covariance)
        
        
        sys.stdout.write("\033[K")
        print((f'\r{i} out of {total_datapoints}'
               f' state rmse {np.sqrt(error_squared_acumulator/(i+1))}'
               f' measurement rmse {np.sqrt(measurement_error_acumulator/(i+1))}'
            #    f' state {next_state}'
            #    f' expected state {multiple_step_ground_truth[i,:4]}'
            #    f' expected gf {ls_gf}'
                # f' predicted covariance {np.diagonal(predicted_covar)}'
            )
               ,end="")
        

    #Print a new line since the current status text is supposed to be in line
    print("\n\r")

    #Calculate the rmse
    rmse = np.sqrt(error_squared_acumulator/total_datapoints)

    #Calculate the measurement rmse
    measurement_rmse = np.sqrt(measurement_error_acumulator/(i+1))
    
    #Get the least squares gait fingerprint
    ls_gf = model.kmodels[0].subject_gait_fingerprint

    #Get the test id
    test_id = np.load("test_id.npy")
    client.save_plot(f"{subject}_TestID_{test_id:04}")

    return rmse, measurement_rmse, condition_list, num_steps_per_condition,track_rmse_after_steps

