"""
This code is meant to perform least squares based on the defined states 
and defined output. It will save the file as numpy datastructure
"""


#Common imports
import pandas as pd
import numpy as np 

#Personal imports
from context import kmodel
from context import ekf
from context import utils
from kmodel.model_fitting import k_model_fitting
from kmodel.model_definition import function_bases
from kmodel.model_definition import k_model
from kmodel.model_definition import fitted_model

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
model_fitter = k_model.KModelFitter()

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
        
        #If there is nan values is because there was a problem with least 
        # squares. Therefore throw an error
        if np.isnan(model_fit).sum() > 0:
            raise ValueError
        
        #Calculate residual variance
        subject_model = fitted_model.SimpleFitModel(basis_list, model_fit, output_name)
        #Get the expected output
      
        
        #Store the results in the list
        model_fits.append(model_fit)

    
    

