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
from kmodel.model_definition import personal_measurement_function
from ekf.measurement_model import MeasurementModel

#Other common imports
import pickle
from typing import List


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
model_fitter = k_model_fitting.KModelFitter()




def fit_measurement_model(subject:str,
                          subject_data:pd.DataFrame,
                          output_list:List[str]):
    """
    This function will generate a measurement model based on the outputs that
    are fed in
    
    Args: 
    
    subject - string that represents the subject name. In the format AB0x
        where x is the subject number
        
    subject_data - this is the data from the dataframe
    
    output_list - This is a list of the strings for each output name
    
    
    """
    
    #Create a list to store the simple model fit
    simple_fitted_model_list = []
    
    #Calculate the optimal models for each joint
    for output_name in output_list:
        
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
        subject_model = fitted_model.SimpleFitModel(basis_list, 
                                                    model_fit, output_name)
        
        #Store the results in the list
        simple_fitted_model_list.append(subject_model)
        
    #Once all the models are calculated, we can return the model fit object
    model = personal_measurement_function.PersonalMeasurementFunction(
        simple_fitted_model_list,output_list, subject)
    
    #Initialize the measurement model
    measurement_model = MeasurementModel(model, 
                                         calculate_output_derivative=True)
    
    return measurement_model

    
    

