"""
This file is meant to run the first test in the Offline EKF Comparison Paper

of the Naive Least Squares (NLS) vs the Inter-Subject Average (ISA)

"""


#Import Common libraries
import pickle

#Common imports
import numpy as np
import pandas as pd

#Import custom library
from simulate_ekf import simulate_ekf
from generate_simulation_validation_data import generate_data
from generate_simulation_validation_data import generate_random_condition
from ekf_loader import ekf_loader
from context import kmodel
from context import ekf
from context import utils


###############################################################################
###############################################################################
# Get model information
#Define the subject
subject_list = [f"AB{i:02}" for i in range(1,11)]


#Get the joint names
JOINT_NAMES = ['jointangles_thigh_x',
               'jointangles_shank_x',
               'jointangles_foot_x']

JOINT_SPEED = ['jointangles_thigh_dot_x',
               'jointangles_shank_dot_x',
               'jointangles_foot_dot_x']

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
# Misc Settings

#Run in the real time plotter
RT_PLOT = False

#Run the heteroschedastic model to add noise near the intersection
HETEROSCHEDASTHIC_MODEL = True

###############################################################################
###############################################################################
# Setup Noise Parameters of the EKF while updating the gait fingerprints online
print(f"Residual list {residual_list}")

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


#Define the limits
upper_limits = np.array([ np.inf, np.inf, np.inf, 15]).reshape(-1,1)
lower_limits = np.array([-np.inf, 0.6,0.5,-15]).reshape(-1,1)


#Run the EKF across all the subjects
noise_columns_header = [state + 'process model noise' 
                        for state
                        in STATE_NAMES]
column_headers = STATE_NAMES + JOINT_NAMES \
    + JOINT_SPEED + noise_columns_header + ['Subject','Test','conditions']

dataframe = pd.DataFrame(columns=column_headers).astype('object')


#Load samples 
process_model_noise_samples = np.load('process_model_noise_samples.npy')

for process_model_noise_index in range(process_model_noise_samples.shape[0]):

    #Process noise
    #Phase, Phase, Dot, Stride_length, ramp, gait fingerprints
    # PHASE_VAR_AVG =         0     
    # PHASE_DOT_VAR_AVG =     8E-7 #2E-8
    # STRIDE_LENGTH_VAR_AVG = 8E-7  #7E-8
    # RAMP_VAR_AVG =          5E-5

    #Test calibration
    # PHASE_VAR_AVG =         0     
    # PHASE_DOT_VAR_AVG =     8E-7
    # STRIDE_LENGTH_VAR_AVG = 1E-6
    # RAMP_VAR_AVG =          6E-4

    #From tuner
    # scale = 7e-2
    # PHASE_VAR_AVG =         0     
    # PHASE_DOT_VAR_AVG =     2E+1 * scale
    # STRIDE_LENGTH_VAR_AVG = 1E-5 * scale
    # RAMP_VAR_AVG =          1E-3 * scale

    # Q = [PHASE_VAR_AVG,
    #         PHASE_DOT_VAR_AVG,
    #         STRIDE_LENGTH_VAR_AVG,
    #         RAMP_VAR_AVG
    #         ]
    
    print((f"Testing Process Model Noise Sample {process_model_noise_index} "
           f"{process_model_noise_samples[process_model_noise_index]}"))
    q_diag = process_model_noise_samples[process_model_noise_index]
    Q = np.diag(q_diag)


    #Generate a random list of conditions to test out
    random_test_condition = generate_random_condition()


    #DEBUG -- Take a subset to quickly run through the test and debug the csv saving
    subject_list = subject_list[:1]

    for subject_index, subject in enumerate(subject_list):
        
        #Generate the validation data for this subject
        state_data, sensor_data, steps_per_condition, condition_list\
            = generate_data(subject, 
                            STATE_NAMES,
                            JOINT_NAMES,
                            random_test_condition)
        
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
                    "measurement rmse {measurement_rmse}"))
            
            #Get the data to save
            data = [[*rmse_testr, *measurement_rmse, *q_diag, subject,
                 'NSL',condition_list]]
        except AssertionError:
            print(f"Failed Assertion on {q_diag}")
            #Set data to invalid
            data = [*([None]*10), *q_diag, subject, 'NSL',condition_list]
      
        
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
                    
            print(f"State rmse {rmse_testr}, measurement rmse {measurement_rmse}")
            
            #Create CSV data vector
            data = [[*rmse_testr, *measurement_rmse, *q_diag, subject,
                    'ISA',condition_list]]
        except AssertionError:
            print(f"Failed Assertion on {q_diag}")
            #Set data to invalid
            data = [*([None]*10), *q_diag, subject, 'ISA',condition_list]
            
        #Add to stored data
        dataframe = pd.concat([dataframe,pd.DataFrame(data,columns=column_headers)],
                            ignore_index=True)


        ##Save to CSV
        dataframe.to_csv("first_paper_NLS_vs_ISA.csv", sep=',')