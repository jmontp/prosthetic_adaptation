"""
This file is deprecated
"""


#Imoprt the kronecker model
from .context import kmodel
from kmodel.personalized_model_factory import PersonalizedKModelFactory
from kmodel.function_bases import FourierBasis, PolynomialBasis
from kmodel.k_model import KroneckerModel

#Import the function bases that will be used to create the functions
#Common imports
import numpy as np
import pandas as pd
import itertools

#Initialize the random basis functions
# phase_basis = FourierBasis(6,'phase')
# phase_dot_basis = HermiteBasis(2,'phase_dot')
# ramp_basis = HermiteBasis(2,'ramp')
# stride_basis = HermiteBasis(2,'stride_length')
# l2_lambda = [0.2,0.01,0,0]

phase_basis = FourierBasis(10,'phase')
phase_dot_basis = PolynomialBasis(1,'phase_dot')
stride_basis = PolynomialBasis(2,'stride_length')
ramp_basis = PolynomialBasis(2,'ramp')

#l2 regularization on the output functions, not the basis functions
l2_lambda = [50,
             1000,
             50]
#Get the output names
output_name = ['jointangles_knee_x',
               'jointangles_thigh_x',
               'jointangles_foot_x']
#Repeat data to act as weighed least squares
REPEAT_FAKE_DATA = 10


basis_list = [phase_basis, 
              phase_dot_basis,  
              stride_basis,
              ramp_basis
              ]

#Create the module 
test_kronecker_model = KroneckerModel(basis_list)

#Dataset
# dataset_location = "../../data/r01_dataset/r01_Streaming_flattened_{}.parquet"
dataset_location = \
    "../../data/flattened_dataport/dataport_flattened_partial_{}.parquet"

#Define the subject list
subject_list = [f'AB{i:02}' for i in range(1,11)]


### Dataset modification
##########
## Import and format dataset

DATASET_LOCATION = ("../../data/flattened_dataport/"
                    "dataport_flattened_partial_{}.parquet")

#Define the subject list
subject_list = [f'AB{i:02}' for i in range(1,11)]

#Load the datasets
r01_dataset_per_person = [(subject_name,
                        pd.read_parquet(DATASET_LOCATION
                        .format(subject_name)))
                        for subject_name
                        in subject_list]

### Balance the dataset so that all conditions have the same amount influence

#Get the least amount of steps per condition
speed_list = [0.8,1.0,1.2]
ramp_list = [-10,-7.5,-5,0,5,7.5,10]

#Get all the combinations of ramp and speed
speed_ramp_comb = list(itertools.product(speed_list,ramp_list))

#Create function that iterates through conditions in a dataset to find the
# condition with the samllest datapoints
find_min_stride = lambda dataset: min([((dataset.speed == speed) &
                                        (dataset.ramp == ramp)).sum()
                                      for speed,ramp
                                      in speed_ramp_comb])

#Find the minimum condition per step
min_steps_in_conidtion_per_person = [find_min_stride(person_dataset)
                                    for _,person_dataset
                                    in r01_dataset_per_person]

assert min(min_steps_in_conidtion_per_person) > 0, \
    "subject with no data for a particular speed, incline condition"

#Function that only keeps the specified amount of datapoints
remove_steps_per_condition = lambda dataset,min_stride: \
    pd.concat([dataset[(dataset.speed == speed) & (dataset.ramp==ramp)]
            .iloc[:min_stride]
            for speed,ramp
            in speed_ramp_comb])

#Get the reduced dataset per person
r01_dataset_per_person = \
    [(name,remove_steps_per_condition(dataset,min_stride))
    for (name,dataset),min_stride
    in zip(r01_dataset_per_person, min_steps_in_conidtion_per_person)]


#Get the state names
state_names = ['phase', 'phase_dot', 'stride_length','ramp']

#column names
c_names = output_name + state_names

#Keep only the columns that are required
r01_dataset_per_person = [(subject,data[c_names])
                          for subject, data
                          in r01_dataset_per_person]

#Add fake data so that the joint angles are fixed to a value when at a certain point
#First define what phase is
phase_list = np.linspace(0,1,150)

#Create combinations of phase and ramp
phase_ramp_combinations = np.array(list(itertools.product(phase_list,ramp_list)))

#Get the number of fake datapoints to use to generate other states later
num_fake_datapoints = phase_ramp_combinations.shape[0]

fake_data_df = pd.DataFrame(phase_ramp_combinations,columns=['phase','ramp'])

#Phase dot is not part of the model so set any value just so that it works
fake_data_df['phase_dot'] = [0]*num_fake_datapoints
#The data is meant so simulate when stride length is equal to 0, so set it to 0
fake_data_df['stride_length'] = [0]*num_fake_datapoints 
#Set the values of the joint angles
fake_data_df['jointangles_thigh_x'] = [115]*num_fake_datapoints
fake_data_df['jointangles_shank_x'] = [11]*num_fake_datapoints
fake_data_df['jointangles_foot_x'] = fake_data_df['ramp']
fake_data_df['jointangles_knee_x'] = [20]*num_fake_datapoints
#Repeat data to act as weighed least squares
for i in range(REPEAT_FAKE_DATA):
    fake_data_df = fake_data_df.append(fake_data_df)


#Append the fake data to the real data
if REPEAT_FAKE_DATA >  0:
    r01_dataset_per_person = [(subject, data.append(fake_data_df))
                            for subject, data 
                            in r01_dataset_per_person]

#Print the column names that have angles in the name
#Set the number of gait fingerptings
num_gf = 5


print("Starting to fit the models")

#Create the personalized kronecker model 
factory = PersonalizedKModelFactory()

#Make a new list to define the runtime 
# subject_left_out_run_list = ['AB01']
subject_left_out_run_list = subject_list

#Keep track of the gait fingerprint list
gf_list = []

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

    #Add the gait fingerprint
    gf_list.append(personalized_k_model.kmodels[0].subject_gait_fingerprint)


#Make it a numpy array
gait_fingerprints_np = np.concatenate(gf_list,axis=0)

#Get the sample covariance
gait_fingerprints_cov = np.var(gait_fingerprints_np,axis=0)

#Print the same covariance
print(f"Sample covariance is {gait_fingerprints_cov}")
np.save("gf_sample_covar", gait_fingerprints_cov)