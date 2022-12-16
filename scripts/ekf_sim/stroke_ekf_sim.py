"""

This file is meant to run the stroke kinematics through the ekf

"""

#Common imports
import numpy as np
import pandas as pd
from itertools import product
import pickle
import matplotlib.pyplot as plt


#Import custom library
from ekf_loader import ekf_loader
from context import kmodel
from context import ekf
from context import utils
from kmodel import model_fitting
from model_fitting import k_model_fitting
from model_fitting import load_models
from kmodel.model_definition import function_bases
from kmodel.model_definition import k_model
from kmodel.model_definition import personal_measurement_function
from ekf.measurement_model import MeasurementModel


###############################################################################
###############################################################################
# Define constants to use later on

#Define the joint names and joint speed
JOINT_NAMES = [
            # 'jointangles_hip_x',
               'jointangles_knee_x',
                # 'jointangles_ankle_x'
               ]

JOINT_SPEEDS = [
    # 'jointangles_hip_dot_x',
               'jointangles_knee_dot_x',
                # 'jointangles_ankle_dot_x'
               ]

NUM_MEASUREMENTS = len(JOINT_NAMES) + len(JOINT_SPEEDS)

STATE_NAMES = ['phase', 'phase_dot', 'stride_length', 'ramp']

NUM_STATES = len(STATE_NAMES)


###############################################################################
###############################################################################
# Load the fit data so that we can use the average subject residual

#Path to model
#Load in the saved model parameters
joint_fit_info_list = []

for joint in JOINT_NAMES:
    
    #Get the saved fit info files
    save_location = '../../data/optimal_model_fits/' 
    save_file_name = save_location + joint + "_optimal.p"

    #Open and load the file
    with open(save_file_name,'rb') as data_file:
        fit_results = pickle.load(data_file)

    #append the fit information
    joint_fit_info_list.append(fit_results)
    

#Get the state names
STATE_NAMES = [basis.var_name for basis in fit_results['basis list']]


residual_list = [np.mean(joint_fit['residual variance list'])
                for joint_fit 
                in joint_fit_info_list]
avg_residual_list = [np.mean(joint_fit['avg residual variance list'])
                for joint_fit 
                in joint_fit_info_list]


###############################################################################
###############################################################################
# Setup Noise Parameters of the EKF while updating the gait fingerprints online

#Optimal test results from empirical testing
q_diag = [0, 1e-9, 3.75e-7, 8.125e-4]
Q = np.diag(q_diag)


#Set the noise model to the average model noise parameters
#Measurement covarience, Innovation
RESIDUAL_POWER = 1

#Get the joint degree tracking error and then joint velocity measurement error
r_diag = [(residual**RESIDUAL_POWER) 
        for residual
        in residual_list] + \
        [(residual**RESIDUAL_POWER) 
        for residual
        in residual_list]
                

R = np.diag(r_diag)

#State initial covariance
# COV_DIAG = 1e-8

#Gait fingerprint initial covariance
# INITIAL_STATE_DIAG = [COV_DIAG]*NUM_STATES
INITIAL_STATE_DIAG = [1e-3,1e-7,1e-3,1e-3]

#Define the initial covariance
initial_state_covar = np.diag(INITIAL_STATE_DIAG)

#Static initial condition near the expected values
initial_state = np.array([0.0,0.4,0.8,0]).reshape(-1,1)

#Define the limits for the state 
upper_limits = np.array([ np.inf, np.inf, np.inf, 15]).reshape(-1,1)
lower_limits = np.array([-np.inf, -np.inf, -np.inf,-15]).reshape(-1,1)

#Define the columns names in the dataframe to filter the data to only the one 
# that we are going to use
noise_columns_header = [state + 'process model noise' 
                        for state
                        in STATE_NAMES]
column_headers = STATE_NAMES + JOINT_NAMES \
    + JOINT_SPEEDS + noise_columns_header + ['Subject','Test',
                                             'conditions','num conditions',
                                             'num_train_steps']


###################################################################
###################################################################
# Create models 

#Load the intersubject average
fitted_model_list = [load_models.load_simple_models(joint,"AVG") 
                        for joint 
                        in JOINT_NAMES]

#Generate a Personal measurement function 
model = personal_measurement_function.PersonalMeasurementFunction(fitted_model_list, 
                                    JOINT_NAMES, "Stroke")
#Initialize the measurement model
ISA_measurement_model = MeasurementModel(model,
                                    calculate_output_derivative=True)


###################################################################
###################################################################
# Load the data for stroke

# Load in the data
df = pd.read_csv('../../data/stroke_data.csv')

# Do only the first step and then repeat it multiple times
REPEAT_STEPS = 100
df = df[(df['time'] <= 2.65) 
        #& (df['time'] >= 0.5)
        ]
df = pd.concat([df]*REPEAT_STEPS)
df['time'] = np.linspace(0,REPEAT_STEPS,len(df))

# Filter the data
time = df['time'].values
dt = df['dt']
sensor_data = df[JOINT_NAMES + JOINT_SPEEDS]

ekf_instance = ekf_loader("Stroke", JOINT_NAMES, 
    initial_state, initial_state_covar, Q, R, 
    lower_limits, upper_limits,
    measurement_model=ISA_measurement_model)


#Store the state and predicted measurements
state_estimates = []
predicted_measurements = []


#Go through the ekf
for data_index in range(sensor_data.shape[0]):
    
    print(f"{data_index}/{sensor_data.shape[0]}")
    
    #Get the data 
    curr_dt = dt.iloc[data_index]
    curr_sensor_data = sensor_data.iloc[data_index].values\
        .reshape(NUM_MEASUREMENTS,1)
    
    #Get next State
    state, covar = ekf_instance.calculate_next_estimates(
        curr_dt,curr_sensor_data
    )
    
    #Add to state list
    state_estimates.append(state)
    predicted_measurements.append(ekf_instance.calculated_measurement_)

#Initialize states
state_estimates_np = np.array(state_estimates).reshape(-1,4)
predicted_measurements_np = np.array(predicted_measurements)\
                        .reshape(-1,len(JOINT_NAMES*2))

#Create a subplot per state
fig, plts = plt.subplots(len(STATE_NAMES),2)


#Plot all the states
for i in range(len(STATE_NAMES)):
    plts[i,0].plot(time, state_estimates_np[:,i])
    plts[i,0].legend([STATE_NAMES[i]])
    
    
#Add grf to phase plot
plts[0,0].plot(time,df['vgrf_paretic'], 'red')
plts[0,0].legend(['phase','grf'])
plts[0,0].set_title('Measurement Model - ' + ' / '.join(JOINT_NAMES))

#Add Fig title
fig.tight_layout()

#Plot all the joint data that is used
for plot_index,joint_name in enumerate(JOINT_NAMES):
    joint_kinematics = df[joint_name].values
    
    plts[plot_index,1].plot(time,joint_kinematics)
    plts[plot_index,1].plot(time, predicted_measurements_np[:,plot_index])

    
    
    plts[plot_index,1].legend([joint_name, joint_name+'_predicted'])
    
    
    
    

#Display legend
plt.show()