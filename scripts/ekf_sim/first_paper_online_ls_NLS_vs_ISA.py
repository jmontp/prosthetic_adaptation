"""
This file is meant to run the first test in the Offline EKF Comparison Paper

of the Naive Least Squares (NLS) vs the Inter-Subject Average (ISA)

"""


#Import Common libraries
import pickle
import json

#Common imports
import numpy as np
import pandas as pd
from itertools import product

#Import custom library
from simulate_ekf import simulate_ekf
from generate_simulation_validation_data import generate_data
from generate_simulation_validation_data import generate_random_condition
from create_partial_data_conditions import task_data_constraint_list
from create_partial_data_conditions import filter_data_for_condition
from ekf_loader import ekf_loader
from context import kmodel
from context import ekf
from context import utils
from kmodel.model_fitting import k_model_fitting
from kmodel.model_definition import function_bases
from kmodel.model_definition import k_model

###############################################################################
###############################################################################
# Get the data to fit the models

#Set the current index to start within the least_squares_condition_list
START_INDEX=0

#Filter the data constraints based on the Start index
task_data_constraint_list = task_data_constraint_list[START_INDEX:]


###############################################################################
###############################################################################
# Define constants to use later on

subject_list = [f"AB{i:02}" for i in range(1,11)]


#Define the joint names and joint speed
JOINT_NAMES = ['jointangles_thigh_x',
               'jointangles_shank_x',
               'jointangles_foot_x']

JOINT_SPEEDS = ['jointangles_thigh_dot_x',
               'jointangles_shank_dot_x',
               'jointangles_foot_dot_x']

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
NUM_STATES = len(STATE_NAMES)

residual_list = [np.mean(joint_fit['residual variance list'])
                for joint_fit 
                in joint_fit_info_list]
avg_residual_list = [np.mean(joint_fit['avg residual variance list'])
                for joint_fit 
                in joint_fit_info_list]



###############################################################################
###############################################################################
# Least Squares Model Definition

# Define the output of the joint angles
output_list = [#'jointmoment_hip_x','jointmoment_knee_x','jointmoment_ankle_x',
               'jointangles_thigh_x', 'jointangles_shank_x','jointangles_foot_x']


## Defines the model and which states will be used
states = ['phase', 'phase_dot', 'stride_length', 'ramp']

phase_basis = function_bases.FourierBasis(20, 'phase')
phase_dot_basis = function_bases.PolynomialBasis(2,'phase_dot')
stride_length_basis = function_bases.PolynomialBasis(2,'stride_length')
ramp_basis = function_bases.PolynomialBasis(2,'ramp')


#Create a list with all the basis functions
basis_list = [phase_basis, phase_dot_basis, stride_length_basis, ramp_basis]

#Create the kronecker model
k_model_instance = k_model.KroneckerModel(basis_list)

###############################################################################
###############################################################################
# Misc Simulation Settings

#Run in the real time plotter
RT_PLOT = False

#Run the heteroschedastic model to add noise near the intersection
HETEROSCHEDASTHIC_MODEL = True

###############################################################################
###############################################################################
# Setup Noise Parameters of the EKF while updating the gait fingerprints online

#Optimal test results from empirical testing
q_diag = [0, 1e-6, 3.75e-7, 8.125e-4]
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
COV_DIAG = 1e-5

#Gait fingerprint initial covariance
INITIAL_STATE_DIAG = [COV_DIAG]*NUM_STATES

#Define the initial covariance
initial_state_covar = np.diag(INITIAL_STATE_DIAG)

#Static initial condition near the expected values
initial_state = np.array([0,0.9,0.5,0]).reshape(-1,1)

#Define the limits for the state 
upper_limits = np.array([ np.inf, np.inf, np.inf, 15]).reshape(-1,1)
lower_limits = np.array([-np.inf, 0.6,0.5,-15]).reshape(-1,1)

#Define the columns names in the dataframe to filter the data to only the one 
# that we are going to use
noise_columns_header = [state + 'process model noise' 
                        for state
                        in STATE_NAMES]
column_headers = STATE_NAMES + JOINT_NAMES \
    + JOINT_SPEEDS + noise_columns_header + ['Subject','Test','conditions']


#Iterate through the task constraints to do least squares with 
for task_data_constraint in task_data_constraint_list:

    print(f"Testing Least Squares Condition {task_data_constraint} ")
    

    
    #Generate a random list of ramp, speed task conditions to test out
    random_test_condition = generate_random_condition()
   
    #Run the simulation for all the subjects
    for subject_index, subject in enumerate(subject_list):
        
        
        ###################################################################
        ###################################################################
        # Do least squares fit for each the naive least squares and the 
        # optimal least squares
        
        #Load the training data
        file_location = ("../../data/flattened_dataport/validation/"
                         "dataport_flattened_training_{}.parquet")
        
        #Create the fitting object
        model_fitter = k_model_fitting
        
        #Generate the validation data for this subject
        state_data, sensor_data, steps_per_condition, condition_list\
            = generate_data(subject, 
                            STATE_NAMES,
                            JOINT_NAMES,
                            random_test_condition)
        
        #Convert the condition list into tuple of tuples from list of lists
        # This will make it hashable to enable elimination of rows through 
        # df.drop_duplicates
        condition_list = json.dumps(condition_list)
        
        #Do not real time plot after subject AB01
        if subject_index > 0:
            RT_PLOT = False
        
        #Wait for the plotter to catch up, if not it will crash
        if RT_PLOT:
            input("Press enter between tests")
        print(f"{subject} Naive Least Squares")
        
        
        ## Do the personalized model fit 
        ekf_instance = ekf_loader(subject, JOINT_NAMES, 
                                initial_state, initial_state_covar, Q, R, 
                                lower_limits, upper_limits,
                                use_subject_average=False)
        
        #Run the simulation
        try:
            rmse_testr, measurement_rmse, rmse_delay\
                    = simulate_ekf(ekf_instance,state_data,sensor_data,
                                plot_local=RT_PLOT)
            print((f"State rmse {rmse_testr}, "
                    f"measurement rmse {measurement_rmse}"))
            
            #Get the data to save
            data = [[*rmse_testr, *measurement_rmse, *q_diag, subject,
                 'NSL',condition_list]]
        except AssertionError:
            print(f"Failed Assertion on {q_diag}")
            #Set data to invalid
            data = [[*([None]*10), *q_diag, subject, 'NSL',condition_list]]
      
        
        #Add to stored data
        dataframe = pd.concat([dataframe,pd.DataFrame(data,columns=column_headers)],
                            ignore_index=True)

        
        #Wait for the plotter to catch up, if not it will crash
        if RT_PLOT:
            input("Press enter between tests")
        
        print(f"{subject} Inter-Subject Average")
        #Run the test for the subject with avearge model fit
        ## Do the personalized model fit 
        ekf_instance = ekf_loader(subject, JOINT_NAMES, 
                                initial_state, initial_state_covar, Q, R, 
                                lower_limits, upper_limits,
                                use_subject_average=True)
        try:
            #Run the simulation
            rmse_testr, measurement_rmse, rmse_delay\
                        = simulate_ekf(ekf_instance,state_data,sensor_data,
                                    plot_local=RT_PLOT)
                    
            print(f"State rmse {rmse_testr}, "
                  f"measurement rmse {measurement_rmse}")
            
            #Create CSV data vector
            data = [[*rmse_testr, *measurement_rmse, *q_diag, subject,
                    'ISA',condition_list]]
            
        except AssertionError:
            print(f"Failed Assertion on {q_diag}")
            #Set data to invalid
            data = [[*([None]*10), *q_diag, subject, 'ISA',condition_list]]
            
        #Add to stored data
        dataframe = pd.concat([dataframe,pd.DataFrame(data,columns=column_headers)],
                            ignore_index=True)

        ##Since I had to run duplicate tests due to a bug, make sure that 
        # we don't have duplicate rows which can alter the data analysis
        dataframe.drop_duplicates(ignore_index=True)
        
        #Add offset of table for visualization
        state_offset_list = ['phase','phase_dot','stride_length', 'ramp']
        offset_labels = [state+'_row_offset' for state in state_offset_list]
        
        state_offset = np.concatenate([dataframe[state_offset_list].values[1:], 
                                       [[None]*len(offset_labels)]])
        dataframe[offset_labels] = state_offset
        
        ##Save to CSV
        dataframe.to_csv("online_ls_test.csv", sep=',')