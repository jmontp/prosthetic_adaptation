#Imoprt the kronecker model
from context import kmodel
from kmodel.personalized_model_factory import PersonalizedKModelFactory
from kmodel.function_bases import FourierBasis, HermiteBasis
from kmodel.k_model import KroneckerModel

#Import the function bases that will be used to create the functions
#Common imports
import numpy as np
import pandas as pd


#Initialize the random basis functions
phase_basis = FourierBasis(6,'phase')
phase_dot_basis = HermiteBasis(1,'phase_dot')
ramp_basis = HermiteBasis(2,'ramp')
stride_basis = HermiteBasis(2,'stride_length')

basis_list = [phase_basis, 
              phase_dot_basis,  
              stride_basis,
              ramp_basis]

#Create the module 
test_kronecker_model = KroneckerModel(basis_list)

#Dataset
# dataset_location = "../../data/r01_dataset/r01_Streaming_flattened_{}.parquet"
dataset_location = "../../data/flattened_dataport/dataport_flattened_partial_{}.parquet"

#Define the subject list
subject_list = [f'AB{i:02}' for i in range(1,11)]

#Load the datasets
r01_dataset = [(subject_name,pd.read_parquet(dataset_location.format(subject_name))) for subject_name in subject_list]

#Print the column names that have angles in the name
#Set the number of gait fingerptings
num_gf = 5

#Get the output names
output_name = ['jointangles_foot_x','jointangles_shank_x','jointangles_thigh_x',]# 'jointangles_knee_x', 'jointangles_hip_x',]
               #'jointangles_shank_dot_x', 'jointangles_foot_dot_x', 'jointangles_knee_dot_x', 'jointangles_hip_dot_x','jointangles_thigh_dot_x',] #Dataport Dataset
#output_name = ['jointAngles_LAnkleAngles_x', 'jointAngles_LHipAngles_x', 'jointAngles_LKneeAngles_x'] #R01 Dataset

print("Starting to fit the models")

#Create the personalized kronecker model 
factory = PersonalizedKModelFactory()

#Make a new list to define the runtime 
subject_left_out_run_list = ['AB01']

#Create a model per subject for the leave-one-out cross validation
for left_out_subject_name in subject_left_out_run_list:
    
    print(f"Doing left out experiment for {left_out_subject_name}")

    #Fit the model
    personalized_k_model = \
        factory.generate_personalized_model(test_kronecker_model, r01_dataset, output_name, 
                                            num_pca_vectors=num_gf, 
                                            keep_subject_fit=left_out_subject_name, 
                                            left_out_subject=left_out_subject_name)

    #Save the model
    # factory.save_model(personalized_k_model, f'../../data/kronecker_models/left_one_out_model_{left_out_subject_name}.pickle')
    print(f"Done with {left_out_subject_name}")

    #Print out the personalization factor for each model 
    for model in personalized_k_model.kmodels:
        print(f"Gait fingerprint for {model.output_name} is {model.subject_gait_fingerprint}")
