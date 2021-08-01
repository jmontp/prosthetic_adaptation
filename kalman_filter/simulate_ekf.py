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
from cycler import cycler

from measurement_model import Measurement_Model
from dynamic_model import Gait_Dynamic_Model
from ekf import Extended_Kalman_Filter
from kronecker_model import Kronecker_Model, model_loader


#%matplotlib qt


def simulate_ekf(filename):
    #%%
    file_location = '../local-storage/test/dataport_flattened_partial_{}.parquet'
    subject_name = 'AB09'
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
    
    initial_gait_fingerprint = model_foot.subjects[subject_name]['cross_model_gait_coefficients_unscaled']
    zero_initial_gf = np.array([[0.0,0.0,0.0,0.0,0.0]]).T
    measurement_model = Measurement_Model(state_names,models)

    num_states = len(state_names)
    num_outputs = len(models)
    
    ############ Initial States
    #Phase, Phase, Dot, Stride_length, ramp
    initial_state_partial= np.array([[0.0,1.3,1.2,-9.0]]).T
    #initial_state = np.concatenate((initial_state_partial,initial_gait_fingerprint))
    initial_state = np.concatenate((initial_state_partial,zero_initial_gf))

    initial_state_diag = [1e-14,1e-14,1e-14,1e-14,
                          1e0,1e0,1e0,1e0,1e0]
    
    initial_state_covariance = np.diag(initial_state_diag)
    ############ Noise
    #Measurement covarience, Innovation
    #Safe
    #r_diag = [25,25,3000,3000]
    #Test
    r_diag = [25,25,3000,3000]
    #Do not trust the sensor at all, turn off innovation and just use dynamic model
    #r_diag = [1e8,1e8,1e8,1e8]
    #Trust the sensors a lot
    #r_diag = [65,65,65,65]
    R = np.diag(r_diag)
    
    #Process noise
    #['phase','phase_dot','step_length','ramp']
    #Best so far
    # q_diag = [0,5e-9,8e-10,9e-3,
    #           1e-7,1e-7,1e-7,1e-7,1e-7]
    
    #Best for long test
    # q_diag = [0,1e-5,1e-6,3e-3,
    #          1e-7,1e-7,1e-7,1e-7,1e-7]
    #Test cal
    q_diag = [0,1e-5,1e-6,7e-3,
              0,0,0,0,0]
    
    Q = np.diag(q_diag)
    ###################

    d_model = Gait_Dynamic_Model()

    ekf = Extended_Kalman_Filter(initial_state,initial_state_covariance, d_model, Q, measurement_model, R)

    #TODO: the timestep can be extracted from the data
    #steps are roughly 0.8 - 1.2 seconds long, we have 150 datapoints
    #therefore a rough estimate is 1 second with 150 datapoints e.g. 1/150
    #Gray = use ground truth phase dot to generate correct time

    #Really just want to prove that we can do one interation of this
    #Dont really want to pove much more than this since we would need actual data for that
    
    control_input_u = 0 

    #TODO - talk to Ting Wei about EKF kidnapping robustness
    #Setup data per section constants
    sections = 16
    steps_per_sections = 20
    points_per_step = 150
    points_per_section = points_per_step * steps_per_sections
    experiment_point_gap = 107 * points_per_step
    #Skip x amount of steps from the start of the experiments
    skip_steps = 10
    skip_points = skip_steps * points_per_step
    #Make the first section very long to learn gait fingerprint
    first_section_steps = 40
    first_section_points = first_section_steps * points_per_step + 75
    #Generate data per step and ground truth
    
    datapoints = points_per_section * sections + first_section_points

    multiple_step_data = np.zeros((datapoints,len(models)))
    multiple_step_ground_truth = np.zeros((datapoints,len(models)))
    

    
    for i in range(-1,sections):
        
        f = first_section_points
        
        if i == -1:
            multiple_step_data[:f, :] = \
                data.iloc[:f, :]
            
            multiple_step_ground_truth[:f, :] = \
                ground_truth.iloc[:f, :]
        else:
            multiple_step_data[(i*points_per_section) + f : \
                               (i*points_per_section) + f + points_per_section , :] = \
                data.iloc[i*experiment_point_gap + skip_points + f:\
                          i*experiment_point_gap + skip_points + f + points_per_section, :]
                
            multiple_step_ground_truth[i*points_per_section + f:(i+1)*points_per_section + f, :] = \
                ground_truth.iloc[i*experiment_point_gap + skip_points + f: i*experiment_point_gap + points_per_section + skip_points + f, :]
    
    
    #Calculate the time step based on the fact that phase_dot = dphase/dt
    #And that dphase = 150 from the fact that we are using the normalized dataset
    # dt = dt/dphase * dphase
    
    time_step = (np.reciprocal(multiple_step_ground_truth[:,1])*1/150).reshape(-1)
    #time_step = np.repeat(1/150,datapoints)

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
            next_state = ekf.calculate_next_estimates(time_step[i], control_input_u, curr_data)[0].T
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
    time_axis = np.arange(datapoints)*1/150
    
    individual_measurements_labels = ['Measured', 'Expected']
    
    fig,axs = plt.subplots(8,1,sharex=True)
    
    colors = ['red', 'green', 'orange', 'blue', ]
    
    axs[-1].set_xlabel("Time (s)")

    #Plot foot angle
    fig.suptitle(f'Q: {q_diag}')
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
    axs[4].plot(time_axis, state_history[:,4:])
    axs[4].set_ylabel("Gait Fingerprints")
    axs[4].legend([f'gf{i}' for i in range(1,6)])

    #Plot the Difference between real state and actual state 
    axs[5].plot(time_axis, delta_state)
    axs[5].set_ylabel("State Estimation Error")
    axs[5].legend(state_names)
    
    
     
    custom_cycler = (cycler(color=['c', 'g', 'orange']))
    axs[6].set_prop_cycle(custom_cycler)    
    
    #Plot the estimated state and ground truth state
    axs[6].plot(time_axis, state_history[:,:3])
    axs[6].set_ylabel("Estimated State")
    axs[6].plot(time_axis, multiple_step_ground_truth[:,:3], 
                linestyle = '--')
    axs[6].set_ylabel("Ground Truth")
    axs[6].legend(state_names[:3] + ground_truth_labels[:3])
    
  
    
    #Plot the ground truth states
    axs[7].plot(time_axis, state_history[:,3])
    axs[7].plot(time_axis, multiple_step_ground_truth[:,3])
    axs[7].set_ylabel("Ramp")
    axs[7].legend(["Estimated Ramp", "Ground Truth Ramp"])
    
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
    