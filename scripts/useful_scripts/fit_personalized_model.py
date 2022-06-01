#Imoprt the kronecker model
from context import kmodel
from kmodel.personalized_model_factory import PersonalizedKModelFactory
from kmodel.function_bases import FourierBasis, HermiteBasis, PolynomialBasis
from kmodel.k_model import KroneckerModel

#Import the function bases that will be used to create the functions
#Common imports
import numpy as np
import pandas as pd


#Initialize the random basis functions
# phase_basis = FourierBasis(6,'phase')
# phase_dot_basis = HermiteBasis(2,'phase_dot')
# ramp_basis = HermiteBasis(2,'ramp')
# stride_basis = HermiteBasis(2,'stride_length')
# l2_lambda = [0.2,0.01,0,0]

phase_basis = FourierBasis(3,'phase')
phase_dot_basis = HermiteBasis(1,'phase_dot')
# ramp_basis = HermiteBasis(2,'ramp')
stride_basis = HermiteBasis(2,'stride_length')

#l2 regularization on the output functions, not the basis functions
l2_lambda = [0.0000,
             0.0000,
             0.00,
             0.0]

basis_list = [phase_basis, 
              phase_dot_basis,  
              stride_basis,
              #ramp_basis
              ]

#Create the module 
test_kronecker_model = KroneckerModel(basis_list)

#Dataset
# dataset_location = "../../data/r01_dataset/r01_Streaming_flattened_{}.parquet"
dataset_location = "../../data/flattened_dataport/dataport_flattened_partial_{}.parquet"

#Define the subject list
subject_list = [f'AB{i:02}' for i in range(1,11)]


### Dataset modification
#Load the datasets
r01_dataset_per_person = [(subject_name,pd.read_parquet(dataset_location.format(subject_name))) for subject_name in subject_list]

#Filter for ramp = 0
mask_list = [dataset['ramp'] == 0.0 for name,dataset in r01_dataset_per_person]
r01_dataset_per_person = [(name,dataset[mask]) for (name,dataset),mask in zip(r01_dataset_per_person, mask_list)]

#Get the least amount of steps per condition
speed_list = [0.8,1.0,1.2]
find_min_stride = lambda dataset: min([(dataset.speed == speed).sum() for speed in speed_list])
min_steps_in_conidtion_per_person = [find_min_stride(dataset) for name,dataset in r01_dataset_per_person]

#Filter to get the
remove_steps_per_condition = lambda dataset,min_stride: pd.concat([dataset[dataset.speed == speed].iloc[:min_stride] for speed in speed_list])
r01_dataset_per_person = [(name,remove_steps_per_condition(dataset,min_stride)) for (name,dataset),min_stride in zip(r01_dataset_per_person, min_steps_in_conidtion_per_person)]

#Print the column names that have angles in the name
#Set the number of gait fingerptings
num_gf = 3

#Get the output names
output_name = ['jointangles_thigh_x']#,'jointmoment_ankle_x]# 'jointangles_knee_x', 'jointangles_hip_x',]
               #'jointangles_shank_dot_x', 'jointangles_foot_dot_x', 'jointangles_knee_dot_x', 'jointangles_hip_dot_x','jointangles_thigh_dot_x',] #Dataport Dataset
#output_name = ['jointAngles_LAnkleAngles_x', 'jointAngles_LHipAngles_x', 'jointAngles_LKneeAngles_x'] #R01 Dataset

print("Starting to fit the models")

#Create the personalized kronecker model 
factory = PersonalizedKModelFactory()

#Make a new list to define the runtime 
subject_left_out_run_list = ['AB01']
subject_left_out_run_list = subject_list


#Create a model per subject for the leave-one-out cross validation
for left_out_subject_name in subject_left_out_run_list:
    
    print(f"Doing left out experiment for {left_out_subject_name}")

    #Fit the model
    personalized_k_model, ekf_output_k_model = \
        factory.generate_personalized_model(test_kronecker_model, r01_dataset_per_person, output_name, 
                                            num_pca_vectors=num_gf, 
                                            keep_subject_fit=left_out_subject_name, 
                                            left_out_subject=left_out_subject_name,
                                            l2_lambda=l2_lambda,
                                            vanilla_pca=False,
                                            #leave_out_model_name='jointmoment_ankle_x'
                                            )

    #Save the model
    factory.save_model(personalized_k_model, f'../../data/kronecker_models/left_one_out_model_{left_out_subject_name}.pickle')
    factory.save_model(ekf_output_k_model, f'../../data/kronecker_models/left_one_out_model_{left_out_subject_name}_moment.pickle')
    print(f"Done with {left_out_subject_name}")

    #Print out the personalization factor for each model 
    for model in personalized_k_model.kmodels:
        print(f"Gait fingerprint for {model.output_name} is {model.subject_gait_fingerprint}")
