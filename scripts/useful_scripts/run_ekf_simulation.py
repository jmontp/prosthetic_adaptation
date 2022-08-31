"""
This file is meant to launch many different simulations of extended kalman 
filters and log their results

"""

#Import Common libraries
from datetime import date
import json
from collections import OrderedDict
import time

#Common imports
import numpy as np

#Import custom library
from simulate_ekf import simulate_ekf
from context import kmodel
from context import ekf
from context import utils
from kmodel.personalized_model_factory import PersonalizedKModelFactory

#Imoprt google sheet library to store results
import gspread


###############################################################################
###############################################################################
# Get model information
#Define the subject
subject_list = [f"AB{i:02}" for i in range(1,11)]

#Import the personalized model 
factory = PersonalizedKModelFactory()

#Path to model
model_dir = (f'../../data/kronecker_models/'
             f'left_one_out_model_{subject_list[0]}.pickle')

#Load model from disk
model = factory.load_model(model_dir)

#Set the number of gait fingerprints
NUM_GAIT_FINGERPRINTS = model.num_gait_fingerprint
NUM_MODELS = len(model.output_names)

#Define states
NUM_STATES = model.num_states
STATE_NAMES = model.kmodels[0].model.basis_names

#Get the joint names
JOINT_NAMES = model.output_names

###############################################################################
###############################################################################
# Setup up google sheets writter

#Log in based on the service_account key found in ~/.config/gspread
sa = gspread.service_account()

#Get the file that we want
google_file = sa.open("EKF Simulation Results")

#Get the worksheet
google_worksheet = google_file.worksheet("Incoming Data")
calibration_worksheet = google_file.worksheet("Calibration Testing")


###############################################################################
###############################################################################
# Setup Noise Parameters of the EKF while updating the gait fingerprints online

#Measurement covarience, Innovation
POSITION_NOISE_SCALE = 1
VELOCITY_NOISE_SCALE = 1 
RESIDUAL_POWER = 2

#Get the joint degree tracking error and then joint velocity measurement error
r_diag = [POSITION_NOISE_SCALE*(kmodel.avg_model_residual**RESIDUAL_POWER) 
         for kmodel
         in model.kmodels] + \
         [VELOCITY_NOISE_SCALE*(kmodel.avg_model_residual**RESIDUAL_POWER) 
         for kmodel
         in model.kmodels]

R = np.diag(r_diag)

#Try to load saved initial covariance
# If no covariance is found then use default value
#Create helper function for when we want to update covar online
def load_initial_covar():
    try:
        initial_state_covar = np.load('pred_cov_gf.npy')

        #If the covariance has the wrong shape, use default values
        if initial_state_covar.shape[0] != NUM_STATES + NUM_GAIT_FINGERPRINTS:
            raise FileNotFoundError

    except FileNotFoundError:
        #State initial covariance
        COV_DIAG = 1e-3
        #Gait fingerprint initial covariance
        GF_COV_INIT = 1e-5
        INITIAL_STATE_DIAG = [COV_DIAG]*NUM_STATES + \
                            [GF_COV_INIT]*(NUM_GAIT_FINGERPRINTS)
        
        #Define the initial covariance
        initial_state_covar = np.diag(INITIAL_STATE_DIAG)

    return initial_state_covar

#Get initial covariance
initial_state_covariance = load_initial_covar()

#Create state limits so that the models do not have to extrapolate over
# conditions that they have not seen before
#Set state limits
upper_limits = np.array([ np.inf, np.inf, 2,20] +\
                        [np.inf]*NUM_GAIT_FINGERPRINTS
                       ).reshape(-1,1)

lower_limits = np.array([-np.inf,-np.inf, 0,-20] + \
                        [-np.inf]*NUM_GAIT_FINGERPRINTS
                       ).reshape(-1,1)


#Process Model Noise For each state
PHASE_VAR =             0
PHASE_DOT_VAR =         4e-9
STRIDE_LENGTH_VAR =     5e-9
RAMP_VAR =              6e-6
GF_VAR =                5e-10


#Try to load from file
try:
    gf_var = np.load("gf_sample_covar").reshape(-1)
    # gf_var = np.power(gf_var,2)
    gf_var_list = list(gf_var*GF_VAR)
    print("... loading gf variance from file")
except FileNotFoundError as e:

    # gf_var_list = [GF_VAR]*(NUM_GAIT_FINGERPRINTS)
    SCALE_FACTOR = 5
    gf_var_list = [(1/SCALE_FACTOR)**i * GF_VAR
                    for i
                    in range(NUM_GAIT_FINGERPRINTS)]

#Create a list with all the elements
Q_DIAG = [PHASE_VAR,
          PHASE_DOT_VAR,
          STRIDE_LENGTH_VAR,
          RAMP_VAR] + gf_var_list

#Convert into a diagonal matrix
Q_GF = np.diag(Q_DIAG)

###############################################################################
###############################################################################
# Setup Noise Parameters of the EKF while the gait fingerprints are fixed

#Try to load saved initial covariance
# If no covariance is found then use default value
#Create helper function for when we want to update covar online
def load_initial_covariance_avg():
    """
    This function attempts to load stored covariance and defaults when it is
    not found
    """
    try: 
        initial_state_covar_avg = np.load('pred_covar_avg.npy')

        #If the covariance has the wrong shape, use default values
        if initial_state_covariance.shape[0] != NUM_STATES:
            raise FileNotFoundError

    except FileNotFoundError:
        #State Initial Covariance
        COV_DIAG = 5e-6
        INITIAL_STATE_DIAG_AVG = [COV_DIAG]*NUM_STATES
        initial_state_covar_avg = np.diag(INITIAL_STATE_DIAG_AVG)

    return initial_state_covar_avg

#Get Initial covariance
initial_state_covariance_avg = load_initial_covariance_avg()

#Process noise
#Phase, Phase, Dot, Stride_length, ramp, gait fingerprints
PHASE_VAR_AVG =         0     
PHASE_DOT_VAR_AVG =     8E-10 #2E-8
STRIDE_LENGTH_VAR_AVG = 8E-10  #7E-8
RAMP_VAR_AVG =          5E-7


Q_AVG = [PHASE_VAR_AVG,
         PHASE_DOT_VAR_AVG,
         STRIDE_LENGTH_VAR_AVG,
         RAMP_VAR_AVG
         ]

Q_AVG_DIAG = np.diag(Q_AVG)


#Define state limits so that we do not extrapolate too far from the parameters
# that the model has not seen before
average_subject_upper_limit = upper_limits[:NUM_STATES,:]
average_subject_lower_limit = lower_limits[:NUM_STATES,:]


#Static initial condition near the expected values
initial_conditions = np.array([0,0.8,0.5,0] +
                              [0]*NUM_GAIT_FINGERPRINTS).reshape(-1,1)
N_INITIAL_CONDITIONS = 1

##############################################################################################################################################################s
##############################################################################################################################################################
#Setup Flags   

#Different test scenarios, only have one set to true
DO_GF = False
DO_AVG = True 
DO_GF_NULL = False
DO_LS_GF = False
DO_AFTER_GF = False
DO_OPTIMAL_FIT = False

#Do the heteroschedastic mreasurement model
HETEROSCHEDASTHIC_MODEL = True

#Plot in the real time plotter interface
RT_PLOT = True 

#Update the initial covariance between subjects
UPDATE_COVAR_BETWEEN_SUBJECTS = False

#Save to google sheets
SAVE_TO_GOOGLE_SHEETS = True

#Repeast Tests to converte the initial covariance
NUM_REPEAT_TESTS = 1

##############################################################################################################################################################
##############################################################################################################################################################
# Give and receive user feedback about current test

#Make sure that you only run one test case at a time
assert sum([DO_GF, DO_AVG, DO_AFTER_GF, 
            DO_LS_GF,DO_GF_NULL,DO_OPTIMAL_FIT]) == 1,\
     "Only one test case per simulation!"

#Mention which test id is currently running
test_id = np.load("test_id.npy")
print(f"Running test: {test_id}")

#Get experiment information
experiment_comment = \
    input("Describe the experiment that you are about to run: ")

###############################################################################
###############################################################################
# Run the simulation
#Repeat to converge the variance
# for test_num in range (NUM_REPEAT_TESTS):

#     #Print where we are in the tests if we are doing multiple
#     if NUM_REPEAT_TESTS > 1:
#         print(f"Repeating test {test_num+1}/{NUM_REPEAT_TESTS}")

#Initialize a list to store the results
subject_results = []

#Calculate the RMSE for all the subjects
for subject_index,subject in enumerate(subject_list):

    print(f"Doing Subject {subject}")

    if (UPDATE_COVAR_BETWEEN_SUBJECTS is True):
        initial_state_covariance = load_initial_covar()
        initial_state_covariance_avg = load_initial_covariance_avg()
        print(f"new covar = {initial_state_covariance}")


    #Iterate through all the initial conditions
    for i in range(N_INITIAL_CONDITIONS):

        #Get the random initial state
        initial_state = initial_conditions[:,[i]]
        
        #Define for cases when gf is not in the state
        initial_state_no_gf = initial_state[:NUM_STATES,:]

        #Wait until the other plot is done to start
        if(RT_PLOT is True):
            # input("Press enter when plot is done updating")
            # for now, we are assuming it takes around 10 seconds to update
            time.sleep(2)

        if(DO_GF is True):
            #Create string to identify test case
            TEST_NAME = "Online Gait Fingerprint"

            #Get the rmse for that initial state
            rmse_testr, measurement_rmse, test_condition, steps_per_cond, rmse_delay\
                            = simulate_ekf(subject,
                                            initial_state,
                                            initial_state_covariance,
                                            Q_GF, R,
                                            lower_limits, upper_limits,
                                            plot_local=RT_PLOT,
                                            heteroschedastic_model=\
                                                HETEROSCHEDASTHIC_MODEL)
        if(DO_GF_NULL is True):
            #Create string to identify test case
            TEST_NAME = "Null Space Gait Fingerprint"

            #Get the rmse for that initial state
            rmse_testr, measurement_rmse, test_condition, steps_per_cond, rmse_delay\
                            = simulate_ekf(subject, 
                                            initial_state,
                                            initial_state_covariance,
                                            Q_GF, R,
                                            lower_limits, upper_limits,
                                            plot_local=RT_PLOT,
                                            null_space_projection=True,
                                            heteroschedastic_model=\
                                                HETEROSCHEDASTHIC_MODEL)
        if(DO_AFTER_GF):
            
            #Create string to identify test case
            TEST_NAME = "After-Step Least Squares Gait Fingerprint"

            #Run the simulation again, with the average fit
            #Get the rmse for that initial state
            rmse_testr, measurement_rmse, test_condition, steps_per_cond, rmse_delay\
                            = simulate_ekf(subject,
                                            initial_state_no_gf,
                                            initial_state_covariance_avg,
                                            Q_AVG_DIAG, R,
                                            average_subject_lower_limit,
                                            average_subject_upper_limit,
                                            plot_local=RT_PLOT,
                                            use_subject_average=True,
                                            calculate_gf_after_step=True,
                                            heteroschedastic_model=\
                                                HETEROSCHEDASTHIC_MODEL)
        if(DO_AVG):
            #Create string to identify test case
            TEST_NAME = "Average Model"

            #Run the simulation again, with the average fit
            #Get the rmse for that initial state
            rmse_testr, measurement_rmse, test_condition, steps_per_cond, rmse_delay\
                            = simulate_ekf(subject,                                             
                                            initial_state_no_gf,
                                            initial_state_covariance_avg,
                                            Q_AVG_DIAG, R,
                                            average_subject_lower_limit,
                                            average_subject_upper_limit,
                                            plot_local=RT_PLOT,
                                            use_subject_average=True,
                                            heteroschedastic_model=\
                                                HETEROSCHEDASTHIC_MODEL)

        
        if(DO_LS_GF):
            #Create string to identify test case
            TEST_NAME = "Offline Least Squares Gait Fingerprint"
            
            #Run the simulation again, with the average fit
            #Get the rmse for that initial state
            rmse_testr, measurement_rmse, test_condition, steps_per_cond, rmse_delay\
                            = simulate_ekf(subject,
                                            initial_state_no_gf,
                                            initial_state_covariance_avg,
                                            Q_AVG_DIAG, R,
                                            average_subject_lower_limit,
                                            average_subject_upper_limit,
                                            plot_local=RT_PLOT,
                                            use_ls_gf=True, 
                                            heteroschedastic_model=\
                                                HETEROSCHEDASTHIC_MODEL)

        if (DO_OPTIMAL_FIT):
            #Create string to identify test case
            TEST_NAME = "Optimal Subject Fit"
            
            #Run the simulation again, with the average fit
            #Get the rmse for that initial state
            rmse_testr, measurement_rmse, test_condition, steps_per_cond, rmse_delay\
                            = simulate_ekf(subject,
                                            initial_state_no_gf,
                                            initial_state_covariance_avg,
                                            Q_AVG_DIAG, R,
                                            average_subject_lower_limit,
                                            average_subject_upper_limit,
                                            plot_local=RT_PLOT,
                                            use_ls_gf=False,
                                            use_optimal_fit=True, 
                                            heteroschedastic_model=\
                                                HETEROSCHEDASTHIC_MODEL)
                                
        #Store results for the subject with the first element being the
        # subject name and the rest being the states
        results = [subject] + [rmse_testr[i] for i in range(NUM_STATES)]

        #Store results for each subject
        subject_results.append(results)
    
        #Print which test we are doing once
        if(subject_index == 0):
            print(f"Doing test {TEST_NAME}")

#Show that we have finished
print("Done")

###########################################################################
###########################################################################
# Send Data to google sheets
#If you do not want it to be saved in the main sheet
if SAVE_TO_GOOGLE_SHEETS is True:

    #If we are doing calibrations, save to separate sheet
    if UPDATE_COVAR_BETWEEN_SUBJECTS is True:
        google_worksheet = calibration_worksheet

    #Create helper function to convert numpy arrays into json format
    np_to_json = lambda arr: json.dumps(arr.tolist())

    #Create function to normalize colors to how google sheets expects [0-1]
    nc = lambda color: color/255.0

    #Create function that formats updated range field that is sent back
    # when append_row is called
    format_range = lambda range: range[range.index('!')+1:]

    #Get the local counter for the test id
    test_id = int(np.load("test_id.npy"))

    #Create the header to store results
    google_sheet_test_metadata_dict = OrderedDict([
        ("Experiment", 
            TEST_NAME),
        ("Process Model Phase Noise", 
            PHASE_VAR),
        ("Process Model Phase_Rate Noise", 
            PHASE_DOT_VAR),
        ("Process Model Stride Length Noise", 
            STRIDE_LENGTH_VAR),
        ("Process Model Ramp Noise", 
            RAMP_VAR),
        ("Process Model Gait Fingerprint Noise",
            GF_VAR), 
        ("Number of Gait Fingerprints",
            NUM_GAIT_FINGERPRINTS),
        ("State Names",
            json.dumps(STATE_NAMES)),
        ("Measurement Model",
            json.dumps(JOINT_NAMES)),
        ("Initial Condition",
            np_to_json(initial_conditions)),
        ("Initial Covariance",
            np_to_json(initial_state_covariance)),
        ("Heteroschedastic Model",
            HETEROSCHEDASTHIC_MODEL),
        ("Test Conditions (Incline, Speed)",
            json.dumps(test_condition)),
        ("Steps per Condition",
            steps_per_cond),
        ("Date",
            str(date.today())),
        ("Test ID",
            test_id)
    ])

    #if you run a test without the gait fingerprint, adjust the parameters
    # accordingly
    if not (DO_GF is True or DO_GF_NULL is True):

        google_sheet_test_metadata_dict["Process Model Phase Noise"] \
            = PHASE_VAR_AVG
        google_sheet_test_metadata_dict["Process Model Phase_Rate Noise"] \
            = PHASE_DOT_VAR_AVG
        google_sheet_test_metadata_dict["Process Model Stride Length Noise"] \
            = STRIDE_LENGTH_VAR_AVG
        google_sheet_test_metadata_dict["Process Model Ramp Noise"] \
            = RAMP_VAR_AVG
        google_sheet_test_metadata_dict[("Process Model Gait "
            "Fingerprint Noise")] = "N/A"
        google_sheet_test_metadata_dict["Number of Gait Fingerprints"] \
            = "N/A"


    #Convert ordered dict into two lists
    google_sheet_test_header = \
        list(google_sheet_test_metadata_dict.keys())
    google_sheet_test_description_data = \
        list(google_sheet_test_metadata_dict.values())

    #Add header for the test metadata
    metadata_row1_info = google_worksheet.append_row(google_sheet_test_header)
    #Add test metadata
    metadata_row2_info = \
        google_worksheet.append_row(google_sheet_test_description_data)

    #Add a line for comments for the experiment 
    # I'm leaving three columns blank so that the comment can be seen clearly
    # without needing the merge columns
    google_worksheet.append_row(['Experiment Comment',
                                experiment_comment,'','','',
                                'Results Comments'])


    #Create the subject trial metadata
    google_sheet_subject_trial_header = [
        "Subject",
        "Phase RMSE",
        "Phase Dot RMSE",
        "Stride Length RMSE",
        "Ramp RMSE"
    ] 

    #Add trial header to display test results
    google_worksheet.append_row(google_sheet_subject_trial_header)
    #Add subject trial for all the subjects
    for trial in subject_results:
        google_worksheet.append_row(trial)

    #Add mean and variance information
    #First, remove the name from each of the subjects
    subject_results_without_names = [trial[1:] for trial in subject_results]
    #Convert into a numpy array
    results_results_np = np.array(subject_results_without_names)\
                                .reshape(len(subject_list),NUM_STATES)

    #Get mean and variance from numpy built in functions
    mean_per_state = np.mean(results_results_np,axis=0)
    var_per_state = np.var(results_results_np,axis=0)

    #Add text header
    mean_with_header = ["Mean"] + \
        [mean_per_state[i] for i in range(NUM_STATES)]
    var_with_header = ["Variance"] + \
        [var_per_state[i] for i in range(NUM_STATES)]

    #Save to google sheets and store the reply information
    mean_update_info = google_worksheet.append_row(mean_with_header)
    var_update_info = google_worksheet.append_row(var_with_header)



    #Add formatting to the metadata so that it is easy to distinguish tests
    # from one another. Format after the data has been stored to be robust
    # against formatting errors crashing the program


    #Define the light grey color for the two metadata rows
    light_grey = {"red":nc(239.0), "green":nc(239.0), "blue":nc(239.0)}
    #Create the metadata format
    meatadata_format = {"backgroundColorStyle":{"rgbColor":light_grey}}
    #Get the rows that will be updated for the first row
    metadata_row1_range = metadata_row1_info['updates']['updatedRange']
    #Remove the worksheet from the name
    metadata_row1_range = format_range(metadata_row1_range)
    #Update metadata row 1 to have light grey background
    google_worksheet.format(metadata_row1_range,meatadata_format)
    #Get the rows that will be updated for the second row
    metadata_row2_range = metadata_row2_info['updates']['updatedRange']
    #Remove the worksheet from the name
    metadata_row2_range = format_range(metadata_row2_range)
    #Update metadata row 2 to have light grey background
    google_worksheet.format(metadata_row2_range,meatadata_format)


    # Format the mean so it is easier to read
    #Get the range to update
    mean_update_range = mean_update_info['updates']['updatedRange']
    #Remove the worksheet name from the range
    mean_update_range = format_range(mean_update_range)
    #Define the rgb values for light green
    light_green = {"red":nc(217.0),"green":nc(234.0),"blue":nc(211.0)}
    #Define the google sheets format dictionary
    mean_format = {"backgroundColor":light_green}
    #Send the request
    google_worksheet.format(mean_update_range,mean_format)


    # Format the variance so it is easier to read
    #Get the range to update
    var_update_range = var_update_info['updates']['updatedRange']
    #Remove the worksheet name from the range
    var_update_range = format_range(var_update_range)
    #Define the rgb values for light red
    light_red = {"red":nc(244.0),"green":nc(204.0),"blue":nc(204.0)}
    #Define the google sheets format dictionary
    var_format = {"backgroundColor":light_red}
    #Send the request
    google_worksheet.format(var_update_range,var_format)


    #Print confirmation
    print("Saved results to google sheets")

#If we are not saving the test that we are doing online, make sure to 
# save the text description offline
else:
    
    #Write to the local file for storing test cases
    with open("offline_test_results.txt", "w") as offline_test_logger:
        offline_test_logger.write(f"{str(date.today())}: {experiment_comment}")

#If we made it to the end, update the local test id
test_id = np.load("test_id.npy")
test_id += 1
np.save("test_id",test_id)