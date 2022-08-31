"""
This code is meant to perform least squares based on the defined states 
and defined output. It will save the file as numpy datastructure
"""


#Common imports
import numpy as np 
import pandas as pd
from sympy import Poly

#Personal imports
from .context import kmodel
from kmodel.function_bases import FourierBasis, PolynomialBasis


###############################################################################
###############################################################################
# Setup parameters

#Define the output of the joint angles
output = ['jointmoment_knee_x']


## Defines the model and which states will be used
states = ['phase', 'phase_dot', 'stride_length', 'ramp']

phase_basis = FourierBasis(10, 'phase')
phase_dot_basis = PolynomialBasis(2,'phase_dot')
stride_length_basis = PolynomialBasis(2,'stride_length')
ramp_basis = PolynomialBasis(2,'ramp')

basis_list = [phase_basis, phase_dot_basis, stride_length_basis, ramp_basis]

###############################################################################
###############################################################################
# Get user data

#Load in the data files
#Define a list of all the subjects
subjects = [f'AB{i:02}' for i in range(1,11)]

#Create a function to return the filename for a given subject
file_location = lambda subject : ("../../data/flattened_dataport/"
                                  f"dataport_flattened_partial_{subject}"
                                  ".parquet")

#Get the data for all the subjects
# contains tuples of (subject name, subject data)
subject_data_list = [(sub,pd.read(file_location(sub))) for sub in subjects]