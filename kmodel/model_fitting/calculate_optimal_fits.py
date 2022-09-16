"""
This code is meant to perform least squares based on the defined states 
and defined output. It will save the file as numpy datastructure
"""


#Common imports
import pandas as pd
import numpy as np 

#Personal imports
from context import model_definition
from model_definition.function_bases import FourierBasis, PolynomialBasis
from k_model_fitting import KModelFitter
from model_definition.k_model import KroneckerModel
from model_definition.fitted_model import SimpleFitModel
import pickle

###############################################################################
###############################################################################
# Setup parameters
l2_regularization = 0.0


# Define the output of the joint angles
output_list = [#'jointmoment_hip_x','jointmoment_knee_x','jointmoment_ankle_x',
               'jointangles_thigh_x', 'jointangles_shank_x','jointangles_foot_x']


## Defines the model and which states will be used
states = ['phase', 'phase_dot', 'stride_length', 'ramp']

phase_basis = FourierBasis(20, 'phase')
phase_dot_basis = PolynomialBasis(2,'phase_dot')
stride_length_basis = PolynomialBasis(2,'stride_length')
ramp_basis = PolynomialBasis(2,'ramp')


#Create a list with all the basis functions
basis_list = [phase_basis, phase_dot_basis, stride_length_basis, ramp_basis]

#Create the kronecker model
k_model_instance = KroneckerModel(basis_list)

###############################################################################
###############################################################################
# Get user data

# Load in the data files
# Define a list of all the subjects
subjects = [f'AB{i:02}' for i in range(1,11)]

# Create a function to return the filename for a given subject
file_location = lambda subject : ("../../data/flattened_dataport/training/"
                                  f"dataport_flattened_training_{subject}"
                                  ".parquet")

#Save location
save_location = "../../data/optimal_model_fits/"

# Get the data for all the subjects
# contains tuples of (subject name, subject data)
subject_data_list = [(sub,pd.read_parquet(file_location(sub))) for sub in subjects]


#Initialize the model fitter object
model_fitter = KModelFitter()

#Calculate the optimal models for each joint
for output_name in output_list:
    
    #Create lists for this output 
    model_fits = []
    RTR_list = []
    residual_list = []
    residual_variance_list = []
    num_datapoints_list = []

    
    for subject_name, subject_data in subject_data_list:
        
        #Get the model fit for the subject
        model_fit, (residual,RTR,num_datapoints) =\
            model_fitter.fit_data(k_model_instance,
                                  subject_data,
                                  output_name,
                                  l2_lambda=l2_regularization,
                                  weight_col='Steps in Condition')
        
        if np.isnan(model_fit).sum() > 0:
            raise ValueError
        
        #Calculate residual variance
        subject_model = SimpleFitModel(basis_list, model_fit, output_name)
        #Get the expected output
        expected_output = subject_model.evaluate(subject_data)
        #Get the true output to compare
        true_output = subject_data[output_name].values.reshape(-1,1)
        #Calculate the residual variance using numpy
        residual_variance = np.var(true_output - expected_output)
        
        #Store the results in the list
        model_fits.append(model_fit)
        residual_list.append(residual)
        residual_variance_list.append(residual_variance)
        RTR_list.append(RTR)
        num_datapoints_list.append(num_datapoints)
        
    
    ##Calculate the average model residual variance
    avg_model_fit = np.mean(model_fits, axis=0)
    avg_residual_variance_list = []

    #Calculate per subject
    for subject_name, subject_data in subject_data_list:
        
        #Calculate residual variance
        subject_model = SimpleFitModel(basis_list, avg_model_fit, output_name)
        #Get the expected output
        expected_output = subject_model.evaluate(subject_data)
        #Get the true output to compare
        true_output = subject_data[output_name].values.reshape(-1,1)
        #Calculate the residual variance using numpy
        residual_variance = np.var(true_output - expected_output)
        #Store the residual variance
        avg_residual_variance_list.append(residual_variance)
    
    #Save the fits for the model into the fit info dataclass object
    save_data = {"model fits": model_fits, 
                 "RTR list": RTR_list,
                 "residual list": residual_list,
                 "residual variance list": residual_variance_list,
                 "avg residual variance list": avg_residual_variance_list,
                 "num datapoints list":num_datapoints_list,
                 "basis list":basis_list,
                 "l2 regularization":l2_regularization}

    #Create save file name
    save_file_name = save_location + output_name + "_optimal.p"
    
    #Print status message
    print(f"Saving {output_name} data: {save_data}")
    
    #Save the data to a file
    with open(save_file_name, 'wb') as save_file:
        pickle.dump(save_data, save_file)
    
    print(f"Done with {output_name}")
    


    

