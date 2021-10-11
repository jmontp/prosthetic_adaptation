#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Standard Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

#Relative Imports
from context import kmodel
from context import ekf
from context import utils
from ekf.measurement_model import MeasurementModel
from ekf.dynamic_model import GaitDynamicModel
from ekf.ekf import Extended_Kalman_Filter
from kmodel.kronecker_model import model_loader
from kmodel import kronecker_model
from utils.math_utils import get_rmse




def simulate_ekf():
    pass
    #%%
    file_location = '../../data/flattened_dataport/dataport_flattened_partial_{}.parquet'
    subject_name = 'AB07'
    filename = file_location.format(subject_name)
    print(f"Looking for {filename}")



    ## Load the models
    #Define the joints that you want to import 
    joint_names = ['jointangles_hip_dot_x','jointangles_hip_x',
                    'jointangles_knee_dot_x','jointangles_knee_x',
                    'jointangles_thigh_dot_x','jointangles_thigh_x']

    model_dir = '../../data/kronecker_models/model_{}.pickle'

    models = [model_loader(model_dir.format(joint)) for joint in joint_names]

    #Get the torque models from 
    torque_names = ['jointmoment_hip_x', 'jointmoment_knee_x']
    torque_model = [model_loader(model_dir.format(torque)) for torque in torque_names]


    ### Load the datasets
    #Read in the parquet dataframe
    total_data = pd.read_parquet(filename)

    #Get the joint data to play back
    joint_data = total_data[joint_names]


    #Phase, Phase Dot, Ramp, Step Length, 5 gait fingerprints
    state_names = ['phase', 'phase_dot', 'stride_length', 'ramp',
                    'gf1', 'gf2','gf3', 'gf4', 'gf5']


    #Delete next line if nothing breaks
    #measurement_names = ["Foot", "Shank", "Foot Dot", "Shank Dot"]


    #Get the ground truth from the datasets
    ground_truth_labels = ['phase','phase_dot','stride_length','ramp']
    ground_truth = total_data[ground_truth_labels]


    #Initialize the measurement model
    measurement_model = MeasurementModel(state_names,models)


    ############ Setup - Initial Gait Fingerprint
    ## Can select either 0 initial gait fingerprint or 
    ## the least square solution gait fingerprint
    least_squares_gait_fingerprint = models[0].subjects[subject_name]['cross_model_gait_coefficients_unscaled']
    zero_gait_fingerprint = np.array([[0.0,0.0,0.0,0.0,0.0]]).T

    initial_gait_fingerprint = zero_gait_fingerprint


    ############ Setup - Initial States
    #### Pretty much random but reasonable initial states 
    #Phase, Phase, Dot, Stride_length, ramp
    initial_state_partial= np.array([[0.0,1.0,1.2,-10.0]]).T
    initial_state = np.concatenate((initial_state_partial,initial_gait_fingerprint))

    #Generate the initial covariance as being very low
    #TODO - double check with gray if this was the strategy that converged or not
    initial_state_diag = [1e-14,1e-14,1e-14,1e-14,
                            1e-14,1e-14,1e-14,1e-14,1e-14]
    initial_state_covariance = np.diag(initial_state_diag)



    ############ Setup - Noise
    #Measurement covarience, Innovation
    #Safe
    r_diag = [25,25,25,3000,3000,3000]
    #Do not trust the sensor at all, turn off innovation and just use dynamic model
    #r_diag = [1e8,1e8,1e8,1e8]
    #Trust the sensors a lot
    #r_diag = [65,65,65,65]
    R = np.diag(r_diag)

    #Process noise
    #['phase','phase_dot','stride_length','ramp']
    #Turn off gait fingerprint
    # q_diag = [0,1e-5,1e-6,3e-4,
    #          1e-28,1e-28,1e-28,1e-28,1e-28]

    #Best for long test
    # q_diag = [0,1e-7,5e-8,1e-8,
    #           1e-13,1e-13,1e-13,1e-13,1e-13]
    #Test cal
    q_diag = [0,3e-7,5e-8,1e-8,
            1e-8,1e-8,1e-8,1e-8,1e-8]
    Q = np.diag(q_diag)

    ###################

    d_model = GaitDynamicModel()

    ekf = Extended_Kalman_Filter(initial_state,initial_state_covariance, d_model, Q, measurement_model, R)

    #We do not include a control input for the EKF, so just set it to zero
    control_input_u = 0 

    #TODO - talk to Ting Wei about EKF kidnapping robustness


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
    multiple_step_data = np.zeros((datapoints,len(models)))
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
    state_history = np.zeros((total_datapoints,len(initial_state)))
    real_measurements = np.zeros((total_datapoints, len(models)))
    measurement_history = np.zeros((total_datapoints, len(models)))
    predicted_states = np.zeros((total_datapoints, len(initial_state)))
    #delta_state = np.zeros((total_datapoints,len(initial_state)))
    #delta_measurements = np.zeros((total_datapoints, len(models)))

    try:
        for i in range(total_datapoints):
            curr_data = multiple_step_data[i].reshape(-1,1)
            real_measurements[i,:] = curr_data.T
            next_state = ekf.calculate_next_estimates(time_step[i], control_input_u, curr_data)[0].T
            state_history[i,:] = next_state
            measurement_history[i,:] = ekf.calculated_measurement_.T
            predicted_states[i,:] = ekf.predicted_state.T
            #delta_state[i,:] = ekf.delta_state.T
            #delta_measurements[i,:] = ekf.y_tilde.T

            print(f'{i} out of {total_datapoints}')
    except KeyboardInterrupt:
        print(f"Interrupted at step {i}")
        total_datapoints = i+1
        curr_data = curr_data[:total_datapoints,:]
        real_measurements = real_measurements[:total_datapoints,:]
        state_history = state_history[:total_datapoints,:]
        measurement_history = measurement_history[:total_datapoints,:]
        predicted_states = predicted_states[:total_datapoints,:]
        multiple_step_ground_truth = multiple_step_ground_truth[:total_datapoints,:]
        pass

    #old plotting    
    #print(state_history[:,0])
    #plt.plot(state_history[:,:])


    rmse_state = [get_rmse(state_history[:,i],multiple_step_ground_truth[:,i]) for i in range(4)]

    rmse_joint_angle = [get_rmse(real_measurements, measurement_history)]    

    #new plotting
    time_axis = np.arange(total_datapoints)*1/150

    individual_measurements_labels = ['Measured', 'Expected']

    fig,axs = plt.subplots(9,1,sharex=True)

    colors = ['red', 'green', 'orange', 'blue', ]

    axs[-1].set_xlabel("Time (s)")


    #Plot measurements
    for i in range(len(joint_names)):
        #Plot foot angle
        fig.suptitle(f'Q: {q_diag}  RMSE State: {rmse_state} RMSE angles: {rmse_joint_angle}')
        axs[i].plot(time_axis, real_measurements[:,i])
        axs[i].plot(time_axis, measurement_history[:,i])
        axs[i].set_ylabel(f"{' '.join(joint_names[i].split('_')[1:-1])}")
        axs[i].legend(individual_measurements_labels)

    #Plot the gait fingerprints
    custom_cycler = (cycler(color=['b', 'g', 'r', 'c', 'm']))
    axs[6].set_prop_cycle(custom_cycler)    

    ls_gf = models[0].subjects[subject_name]['cross_model_gait_coefficients_unscaled']
    desired_gf = np.repeat(ls_gf,time_axis.shape[0], axis=1).T
    axs[6].plot(time_axis, desired_gf, ":", lw=2)
    axs[6].plot(time_axis, state_history[:,4:], lw=2)
    axs[6].set_ylabel("Gait Fingerprints")
    axs[6].legend([f'gf{i}' for i in range(1,6)])
    

    #Plot the estimated state and ground truth state
    custom_cycler = (cycler(color=['b', 'g', 'r']))
    axs[7].set_prop_cycle(custom_cycler)  

    axs[7].plot(time_axis, state_history[:,:3])
    axs[7].set_ylabel("Estimated State")
    axs[7].plot(time_axis, multiple_step_ground_truth[:,:3], 
                linestyle = '--')
    axs[7].set_ylabel("Ground Truth")
    axs[7].legend(state_names[:3] + ground_truth_labels[:3])



    #Plot the ground truth states
    axs[8].plot(time_axis, state_history[:,3])
    axs[8].plot(time_axis, multiple_step_ground_truth[:,3])
    axs[8].set_ylabel("Ramp")
    axs[8].legend(["Estimated Ramp", "Ground Truth Ramp"])

    plt.show()



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