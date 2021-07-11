#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys

#Importing the janky way since its too hard to do it the right way
PACKAGE_PARENT = '../model_fitting/'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
new_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
print(new_path)
sys.path.insert(1,new_path)


import pandas
import numpy as np
import matplotlib.pyplot as plt

from measurement_model import Measurement_Model
from dynamic_model import Gait_Dynamic_Model
from ekf import Extended_Kalman_Filter
from kronecker_model import Kronecker_Model, model_loader


#%matplotlib qt


def simulate_ekf(filename):
    #%%
    file_location = '../local-storage/test/dataport_flattened_partial_{}.parquet'
    subject_name = 'AB10'
    filename = file_location.format(subject_name)
    print(f"Looking for {filename}")
    
    total_data = pandas.read_parquet(filename)
    
    data = total_data[['jointangles_foot_x', 'jointangles_shank_x', 
                 'jointangles_foot_dot_x','jointangles_shank_dot_x']]
    
    ground_truth_labels = ['phase','phase_dot','step_length','ramp']
    
    ground_truth = total_data[ground_truth_labels]
    
    model_foot = model_loader('../model_fitting/foot_model.pickle')
    model_shank = model_loader('../model_fitting/shank_model.pickle')
    model_foot_dot = model_loader('../model_fitting/foot_dot_model.pickle')
    model_shank_dot = model_loader('../model_fitting/shank_dot_model.pickle')
    
    #Phase, Phase Dot, Ramp, Step Length, 5 gait fingerprints
    state_names = ['phase', 'phase_dot', 'step_length', 'ramp',
                    'gf1', 'gf2','gf3', 'gf4', 'gf5']
    
    measurement_names = ["Foot", "Shank", "Foot Dot", "Shank Dot"]
    
      

    models = [model_foot,model_shank,model_foot_dot,model_shank_dot]

    measurement_model = Measurement_Model(state_names,models)

    num_states = len(state_names)
    num_outputs = len(models)
    
    ############ Initial States
    #Phase, Phase, Dot, Stride_length, ramp
    initial_state = np.array([[0.0,1.3,1.2,-9.0,
                                0.0,0.0,0.0,0.0,0.0]]).T
    
    initial_state_covariance = np.eye(num_states)*1e-14
    ############ Noise
    #Measurement covarience, Innovation
    r_diag = [25,25,300,300]
    #Do not trust the sensor at all, turn off innovation and just use dynamic model
    #r_diag = [1e8,1e8,1e8,1e8]
    #Trust the sensors a lot
    #r_diag = [65,65,65,65]
    R = np.diag(r_diag)
    
    #Process noise
    #['phase','phase_dot','step_length','ramp']

    q_diag = [0,1e-5,1e-5,1e-5,
              1e-7,1e-7,1e-7,1e-7,1e-7]
    
    Q = np.diag(q_diag)
    ###################

    d_model = Gait_Dynamic_Model()

    ekf = Extended_Kalman_Filter(initial_state,initial_state_covariance, d_model, Q, measurement_model, R)

    #TODO: the timestep can be extracted from the data
    #steps are roughly 0.8 - 1.2 seconds long, we have 150 datapoints
    #therefore a rough estimate is 1 second with 150 datapoints e.g. 1/150
    time_step = 1/150

    #Really just want to prove that we can do one interation of this
    #Dont really want to pove much more than this since we would need actual data for that
    
    control_input_u = 0 


    #Setup data per section constants
    sections = 10
    steps_per_sections = 10
    points_per_step = 150
    points_per_section = points_per_step * steps_per_sections
    datapoints = points_per_section * sections
    skip_points = 16000

    
    #Generate data per step and ground truth
    multiple_step_data = np.zeros((datapoints,len(models)))
    multiple_step_ground_truth = np.zeros((datapoints,len(models)))
    for i in range(sections):
        
        multiple_step_data[i*points_per_section:(i+1)*points_per_section, :] = \
            data.iloc[i*skip_points: i*skip_points + points_per_section, :]
            
        multiple_step_ground_truth[i*points_per_section:(i+1)*points_per_section, :] = \
            ground_truth.iloc[i*skip_points: i*skip_points + points_per_section, :]
            
    #Create storage for state history
    state_history = np.zeros((datapoints,len(initial_state)))
    real_measurements = np.zeros((datapoints, len(models)))
    expected_measurement = np.zeros((datapoints, len(models)))
    predicted_states = np.zeros((datapoints, len(initial_state)))
    delta_state = np.zeros((datapoints,len(initial_state)))
    delta_measurements = np.zeros((datapoints, len(models)))

    try:
        for i in range(datapoints):
            curr_data = multiple_step_data[i].reshape(-1,1)
            real_measurements[i,:] = curr_data.T
            next_state = ekf.calculate_next_estimates(time_step, control_input_u, curr_data)[0].T
            state_history[i,:] = next_state
            expected_measurement[i,:] = ekf.calculated_measurement_.T
            predicted_states[i,:] = ekf.predicted_state.T
            delta_state[i,:] = ekf.delta_state.T
            delta_measurements[i,:] = ekf.y_tilde.T

            print(f'{i} out of {datapoints}')
    except KeyboardInterrupt:
        pass
    
    #old plotting    
    #print(state_history[:,0])
    #plt.plot(state_history[:,:])

    
    #new plotting
    time_axis = np.arange(datapoints)*time_step
    
    individual_measurements_labels = ['Measured', 'Expected']
    
    figs,axs = plt.subplots(8,1,sharex=True)
    
    axs[-1].set_xlabel("Time (s)")

    #Plot foot angle
    axs[0].plot(time_axis, real_measurements[:,0])
    axs[0].plot(time_axis, expected_measurement[:,0])
    axs[0].set_ylabel("Foot Angle (deg)")
    axs[0].legend(individual_measurements_labels)
    
    #Plot shank angle
    axs[1].plot(time_axis, real_measurements[:,1])
    axs[1].plot(time_axis, expected_measurement[:,1]) 
    axs[1].set_ylabel("Shank Angle (deg)")
    axs[1].legend(individual_measurements_labels)
    
    #Plot foot dot angle
    axs[2].plot(time_axis, real_measurements[:,2])
    axs[2].plot(time_axis, expected_measurement[:,2])
    axs[2].set_ylabel("Foot Dot Angle (deg/s)")
    axs[2].legend(individual_measurements_labels)

    #Plot shank dot angle
    axs[3].plot(time_axis, real_measurements[:,3])
    axs[3].plot(time_axis, expected_measurement[:,3])
    axs[3].set_ylabel("Shank Dot Angle (deg/s)")
    axs[3].legend(individual_measurements_labels)

    #Plot the residual (y_tilde)
    axs[4].plot(time_axis, delta_measurements)
    axs[4].set_ylabel("State residual (y_tilde)")
    axs[4].legend(measurement_names)

    #Plot the Difference between real state and actual state 
    axs[5].plot(time_axis, delta_state)
    axs[5].set_ylabel("State Estimation Error")
    axs[5].legend(state_names)
    
    #Plot the estimated state
    axs[6].plot(time_axis, state_history)
    axs[6].set_ylabel("Estimated State")
    axs[6].legend(state_names)

    #Plot the ground truth states
    axs[7].plot(time_axis, multiple_step_ground_truth)
    axs[7].set_ylabel("Ground Truth")
    axs[7].legend(ground_truth_labels)
    
    
    
    
    plt.show()
    #%%
def main(): 
    #%%
    file_location = '../local-storage/test/dataport_flattened_partial_{}.parquet'
    subject_name = 'AB10'
    filename = file_location.format(subject_name)
    print(f"Looking for {filename}")
    simulate_ekf(filename)
    
    #%%
if __name__=='__main__':
    main()
    