#Imoprt the kronecker model
from context import k_model
from k_model import KroneckerModel
from prosthetic_adaptation.kmodel.model_fitting.k_model_fitting import KModelFitter

#Import the function bases that will be used to create the functions
from context import function_bases
from function_bases import FourierBasis, HermiteBasis

#Common imports
import numpy as np
import pandas as pd
#
#Initialize the random basis functions
phase_basis = FourierBasis(1,'phase')
phase_dot_basis = HermiteBasis(2,'phase_dot')
ramp_basis = HermiteBasis(6,'ramp')
stride_basis = HermiteBasis(2,'stride_length')

basis_list = [phase_basis, phase_dot_basis, ramp_basis, stride_basis]

#Create the module 
test_kronecker_model = KroneckerModel(basis_list)

#Initialize the model fitter
k_mode_fitter = KModelFitter()

#Dataset
r01_dataset_location = "../../data/flattened_dataport/dataport_flattened_partial_{}.parquet"
r01_dataset = pd.read_parquet(r01_dataset_location.format('AB01'))

#Get a random output
output_name = r01_dataset.columns[0]

#Test things out 
fit, residual = k_mode_fitter.fit_data(test_kronecker_model,r01_dataset, output_name)


print(f"Fit {fit}, residual {residual}")