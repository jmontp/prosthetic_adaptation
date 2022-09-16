"""
This file is meant to tune the EKF 

It will do so by doing a poincare section (Gray Cortright suggested this)

In essence, it will identify the error dynamics as a linear system 
computationally by disturbing one of the states and then measuring how long it 
takes the error dynamics to reach zero. 

Ideally we want the error to go down before the end of the step. Therefore, 
we can tune the filter until it will reach the performance metrics that we 
want. 

The limitations is that the linear system assumption will work for small 
disturbances of the gait state, however, for large disturbances it might not 
be effective. 

"""

import time
import numpy as np 

from context import kmodel
from context import ekf

from kmodel.model_fitting.load_models import load_simple_models
from kmodel.model_definition.personal_measurement_function import PersonalMeasurementFunction
from ekf.measurement_model import MeasurementModel
from ekf.dynamic_model import GaitDynamicModel
from ekf.ekf import Extended_Kalman_Filter

import pickle
from rtplot import client

#Define the states of the system
state_list = ['phase','phase_dot','stride_length','ramp']

#Define Time Properties
num_timesteps = 100
end_time = 1.0
dt = end_time/num_timesteps


#Define the plot configuration
plot_x_axis_points = num_timesteps*len(state_list)

phase_subplot = {
                    'names': ['phase', 'phase_a'],
                    'colors' : ['r','r'],
                    'line_style' : [''] + ['-'],
                    #Titles and labels 
                    'title': "Phase",
                    'yrange': [1.1,-0.5],                
                    'xrange':plot_x_axis_points
                    }
stride_length_phase_dot_subplot =  {
                    'names':['phase_dot','stride_length',
                             'phase_dot_a','stride_length_a'],
                    'colors' : ['g','b']*2,
                    'line_style' : ['']*2 + ['-']*2,
                    #Titles and labels 
                    'title': "Phase Dot, Stride Length",
                    'yrange': [0.4,1.5],
                    'xrange':plot_x_axis_points
                    }
ramp_subplot =  {
                'names': ['ramp', 'ramp_a'],
                'colors': ['r','r'],
                'line_style':['','-'],
                #Titles and labels 
                'title': "Ramp",
                'yrange': [-10,10],
                'xrange': plot_x_axis_points
                }
plot_config = [phase_subplot, stride_length_phase_dot_subplot, ramp_subplot]

client.local_plot()
client.initialize_plots(plot_config)
time.sleep(1)

#Create the normal states
t = np.linspace(0,end_time,num_timesteps)
target_ramp = 0
target_phase_dot = 1.0
target_stride_length = 0.9
target_phase = np.linspace(0,target_phase_dot*end_time,num_timesteps)


#Load in the model for the average person
joint_list = ['jointangles_thigh_x',
               'jointangles_shank_x',
               'jointangles_foot_x']

#Load in the average models for the joitn angles
fitted_model_list = [load_simple_models(joint,"AVG")
                     for joint
                     in joint_list]

#Create the model 
model = PersonalMeasurementFunction(fitted_model_list, joint_list, "AVG")


#Initialize the measurement model
measurement_model = MeasurementModel(model,calculate_output_derivative=True)

#Initiailze gait dynamic model
d_model = GaitDynamicModel()


##Create the initial state and covariance
initial_state = np.array([target_phase[0], target_phase_dot, 
                 target_stride_length, target_ramp]).reshape(-1,1)

#State initial covariance
COV_DIAG = 1e-5

#Gait fingerprint initial covariance
initial_state_covariance = np.diag([COV_DIAG]*len(state_list))



##Process Model Noise Tunning
scale = 1e+0
PHASE_VAR_AVG =         0     
PHASE_DOT_VAR_AVG =     2E+1 * scale
STRIDE_LENGTH_VAR_AVG = 1E-5 * scale
RAMP_VAR_AVG =          1E-3 * scale

Q_dig = [PHASE_VAR_AVG,
         PHASE_DOT_VAR_AVG,
         STRIDE_LENGTH_VAR_AVG,
         RAMP_VAR_AVG
         ]

Q = np.diag(Q_dig)

upper_limits = np.array([ np.inf, np.inf, np.inf, 15]).reshape(-1,1)
lower_limits = np.array([-np.inf, -np.inf,0.5,-15]).reshape(-1,1)

##Measurement model noise tunning
#Load in the saved model parameters
joint_fit_info_list = []

for joint in joint_list:
    
    #Get the saved fit info files
    save_location = '../../data/optimal_model_fits/' 
    save_file_name = save_location + joint + "_optimal.p"

    #Open and load the file
    with open(save_file_name,'rb') as data_file:
        fit_results = pickle.load(data_file)

    #append the fit information
    joint_fit_info_list.append(fit_results)
    
    
residual_list = [np.mean(joint_fit['residual variance list'])
                for joint_fit 
                in joint_fit_info_list]

r_diag = [(residual) 
         for residual
         in residual_list*2]

R = np.diag(r_diag)

#List of disturbances
epsilon_list = [0.40, 0.3, 0.5, 5]

#Run a simulation per state
for state_i, state in enumerate(state_list):

    print(f"Doing the simulation on {state}")
    
    #Initial state
    initial_state_copy = initial_state.copy()
    initial_state_copy[state_i] += epsilon_list[state_i]
    
    #Initialize the EKF instance
    ekf_instance = Extended_Kalman_Filter(initial_state_copy, initial_state_covariance, 
                                      d_model, Q, 
                                      measurement_model, R,
                                      lower_state_limit=lower_limits,
                                      upper_state_limit=upper_limits,# output_model=output_model,
                                      heteroschedastic_model = True
                                    )
    
    #Run the simulation
    # input("Press enter to start the simulation")
    for j in range(num_timesteps):
        
        #Get the current state
        current_state = np.array([target_phase[j], target_phase_dot, 
                         target_stride_length, target_ramp]).reshape(-1,1)
        
        #Get the current expected measurement
        current_measurement = measurement_model.evaluate_h_func(current_state) \
            # + np.sqrt([20.6, 13.0, 14.0]*2).reshape(-1,1)
        
        #Get the next state for the ekf
        next_state,predicted_covar = ekf_instance.calculate_next_estimates(dt, current_measurement)
        next_state = next_state.T

        #Plot the expected state against the real state
        next_state = next_state.ravel()
        current_state = current_state.ravel()
        
        data = [next_state[0],      #Estimated Phase
                current_state[0],   #Actual Phase
                next_state[1],      #Estimated Phase Dot
                next_state[2],      #Estimated Stride Length
                current_state[1],   #Actual Phase Dot
                current_state[2],   #Actual Stride Length
                next_state[3],      #Estimated Ramp
                current_state[3],   #Actual Ramp
            ]
        
        #Send Data
        client.send_array(data)
        