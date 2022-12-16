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

#Import custom library
from simulate_ekf import simulate_ekf
from generate_simulation_validation_data import generate_data
from generate_simulation_validation_data import generate_random_condition
from fit_least_squares_model import fit_measurement_model
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



#Load samples 
process_model_noise_samples = np.load('process_model_noise_samples.npy')

#Verify current progress
try:
    #Load in existing process if 
    dataframe = pd.read_csv("first_paper_NLS_vs_ISA.csv")
    first_test_number_of_trials = 125*10
    # start_index = int(len(dataframe)/2)
    #Finished all of subject 1 as a trial, set to zero so that 
    # we can do all the other subjects without having to repeat subject 1
    start_index = 0
    
#If the file is not found, create a new pandas object to keep track of progress
except FileNotFoundError:
    dataframe = pd.DataFrame(columns=column_headers).astype('object')
    start_index = 0
    

#DEBUG - I set this manually since I had a weird error where it stopped 
# running the simulations at index 9
start_index = 89

#Create a generator for the indexes that we want to visit
process_model_noise_index_list =  range(start_index, 
                                        process_model_noise_samples.shape[0])
print(f"Total number of samples {process_model_noise_samples.shape[0]}")


#DEBUG -- Take a subset to quickly run through the test and debug the csv saving
# subject_list = subject_list[1:]

#Load the speed ramp condition, it will be fixed for every subject and  
# process model tuning
with open('random_ramp_speed_condition.pickle','rb') as file:
    random_test_condition = pickle.load(file)

#Iterate through all the process model tunings
for process_model_noise_index in process_model_noise_index_list:

    #Print the tuning that you are doing right now
    print((f"Testing Process Model Noise Sample {process_model_noise_index} "
           f"{process_model_noise_samples[process_model_noise_index]}"))
    
    #Convert the process model tuning array into a matrix
    q_diag = process_model_noise_samples[process_model_noise_index]
    Q = np.diag(q_diag)

    
    #Apply the process model tuning to all the subjects
    for subject_index, subject in enumerate(subject_list):
        
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
        dataframe.to_csv("first_paper_NLS_vs_ISA.csv", sep=',')