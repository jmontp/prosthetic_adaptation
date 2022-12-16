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
from fit_least_squares_model import fit_measurement_model

#Import custom library
from simulate_ekf import simulate_ekf
from generate_simulation_validation_data import generate_data
from generate_simulation_validation_data import generate_random_condition
from generate_partial_ramp_speed_conditions import task_data_constraint_list
from generate_partial_ramp_speed_conditions import filter_data_for_condition
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
    + JOINT_SPEEDS + noise_columns_header + ['Subject','Test',
                                             'conditions','num conditions',
                                             'num_train_steps']




###############################################################################
###############################################################################
# Load the stored data

SAVE_FILE_NAME = "online_ls_test.csv"

#Verify current progress
try:
    #Load in existing process if 
    dataframe = pd.read_csv(SAVE_FILE_NAME)

    first_test_number_of_trials = 125*10
   
    
#If the file is not found, create a new pandas object to keep track of progress
except FileNotFoundError:
    dataframe = pd.DataFrame(columns=column_headers).astype('object')
    start_index = 0

#Load the speed ramp condition, it will be fixed for every subject and  
# process model tuning
with open('random_ramp_speed_condition.pickle','rb') as file:
    random_test_condition = pickle.load(file)

#Create a string that contains the file information
file_location = ("../../data/flattened_dataport/training/"
                    "dataport_flattened_training_{}.parquet")
    
#Iterate through the task constraints to do least squares with 
for task_index, task_data_constraint in enumerate(task_data_constraint_list):

    print((f"Testing Least Squares Condition {task_data_constraint}"
          f" -- {task_index}/{len(task_data_constraint_list)}"))
    
   
    #Run the simulation for all the subjects
    for subject_index, subject in enumerate(subject_list):
        
        
        ###################################################################
        ###################################################################
        # Do least squares fit for each the naive least squares and the 
        # gait fingerprint least squares
        
        #Get the subject-specific fit 
        subject_data = pd.read_parquet(file_location.format(subject))
        
        #Filter for a particular amount of steps
        steps_to_train_on = 150
        
        #Filter the data to only contain the current conditions and
        # number of steps to train on
        subject_data = filter_data_for_condition(subject_data,
                                                 task_data_constraint,
                                                 steps_to_train_on)
        
        
        # optimal least squares
        least_squares_measurement_model = fit_measurement_model(subject, 
            subject_data,output_list)
        
        # Craete the personalized least squares model
        gait_fingerprint_model_list = load_models.load_personalized_models(
            output_list,subject,subject_data)
        
        #Create a measurement model using the gait fingerprint personal model
        gait_fingerprint_measurement_model = \
            MeasurementModel(gait_fingerprint_model_list, 
                             calculate_output_derivative=True)
        
        
        
        #Load the intersubject average
        fitted_model_list = [load_models.load_simple_models(joint,"AVG",
                                                leave_subject_out=subject) 
                                for joint 
                                in output_list]
       
        
        #Generate a Personal measurement function 
        model = personal_measurement_function.PersonalMeasurementFunction(fitted_model_list, 
                                            output_list, subject)
        #Initialize the measurement model
        ISA_measurement_model = MeasurementModel(model,
                                            calculate_output_derivative=True)
        
            
        ###################################################################
        ###################################################################
        #Generate the validation data for this subject
        state_data, sensor_data, steps_per_condition, condition_list\
            = generate_data(subject, 
                            STATE_NAMES,
                            JOINT_NAMES,
                            random_test_condition)
        
        #Convert conditoin list to a tuple instead so that it is hashable
        condition_list = tuple(condition_list)
        
        #Do not real time plot after subject AB01
        if subject_index > 0:
            RT_PLOT = False
        
        #Wait for the plotter to catch up, if not it will crash
        if RT_PLOT:
            input("Press enter between tests")
        print(f"{subject} Naive Least Squares")
        
        ###################################################################
        ###################################################################
        ## Do the personalized model fit 
        ekf_instance = ekf_loader(subject, JOINT_NAMES, 
                                initial_state, initial_state_covar, Q, R, 
                                lower_limits, upper_limits,
                                measurement_model=least_squares_measurement_model)
        
        #Run the simulation
        try:
            rmse_testr, measurement_rmse, rmse_delay\
                    = simulate_ekf(ekf_instance,state_data,sensor_data,
                                plot_local=RT_PLOT)
            print((f"State rmse {rmse_testr}, "
                    f"measurement rmse {measurement_rmse}"))
            
            results = [*rmse_testr, *measurement_rmse]
            
            
        except AssertionError:
            print(f"Failed Assertion on {q_diag}")
            #Set data to invalid
            results = ([None]*10)
               
        #Get the data to save
        data = [[*results, *q_diag, subject,'NLS',task_data_constraint, 
                 len(task_data_constraint),steps_to_train_on]]
            
        #Add to stored data
        dataframe = pd.concat([dataframe,pd.DataFrame(data,columns=column_headers)],
                            ignore_index=True)

        ###################################################################
        ###################################################################
        # Do the gait fingerprint analysis
        
        #Wait for the plotter to catch up, if not it will crash
        if RT_PLOT:
            input("Press enter between tests")
        
        print(f"{subject} Gait Fingerprint")
        #Run the test for the subject with avearge model fit
        ## Do the personalized model fit 
        ekf_instance = ekf_loader(subject, JOINT_NAMES, 
                                initial_state, initial_state_covar, Q, R, 
                                lower_limits, upper_limits,
                                measurement_model=gait_fingerprint_measurement_model)
        try:
            #Run the simulation
            rmse_testr, measurement_rmse, rmse_delay\
                        = simulate_ekf(ekf_instance,state_data,sensor_data,
                                    plot_local=RT_PLOT)
                    
            print(f"State rmse {rmse_testr}, "
                  f"measurement rmse {measurement_rmse}")
            
            #Create CSV data vector
            results = [*rmse_testr, *measurement_rmse]

            
        except AssertionError:
            print(f"Failed Assertion on {q_diag}")
            #Set data to invalid
            results = ([None]*10)

        data = [[*results, *q_diag, subject, 'PCA_GF',task_data_constraint, 
                 len(task_data_constraint),steps_to_train_on]]
            
        #Add to stored data
        dataframe = pd.concat([dataframe,pd.DataFrame(data,columns=column_headers)],
                            ignore_index=True)

        
        
        ###################################################################
        ###################################################################
        # Do the intersubject average model
        
        #Wait for the plotter to catch up, if not it will crash
        if RT_PLOT:
            input("Press enter between tests")
        
        print(f"{subject} Inter Subject Average")
        #Run the test for the subject with avearge model fit
        ## Do the personalized model fit 
        ekf_instance = ekf_loader(subject, JOINT_NAMES, 
                                initial_state, initial_state_covar, Q, R, 
                                lower_limits, upper_limits,
                                measurement_model=ISA_measurement_model)
        try:
            #Run the simulation
            rmse_testr, measurement_rmse, rmse_delay\
                        = simulate_ekf(ekf_instance,state_data,sensor_data,
                                    plot_local=RT_PLOT)
                    
            print(f"State rmse {rmse_testr}, "
                  f"measurement rmse {measurement_rmse}")
            
            #Create CSV data vector
            results = [*rmse_testr, *measurement_rmse]
            
        except AssertionError:
            print(f"Failed Assertion on {q_diag}")
            #Set data to invalid
            results = ([None]*10)

        data = [[*results, *q_diag, subject, 'ISA',task_data_constraint, 
                 len(task_data_constraint),steps_to_train_on]]
        
        #Add to stored data
        dataframe = pd.concat([dataframe,pd.DataFrame(data,
                                                      columns=column_headers)],
                            ignore_index=True)
        
        
        ###################################################################
        ###################################################################
        # Save the data
        
        ##Since I had to run duplicate tests due to a bug, make sure that 
        # we don't have duplicate rows which can alter the data analysis
        dataframe.drop_duplicates(ignore_index=True)
        
        ##Save to CSV
        dataframe.to_csv(SAVE_FILE_NAME, sep=',')