#Imoprt the kronecker model
from context import k_model
from k_model import KroneckerModel
from personal_k_model import PersonalKModel
from personalized_model_factory import PersonalizedKModelFactory

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

#Dataset
r01_dataset_location = "../../data/flattened_dataport/dataport_flattened_partial_{}.parquet"

subject_list = [f'AB{i:02}' for i in range(1,5)]

r01_dataset = [(subject_name,pd.read_parquet(r01_dataset_location.format(subject_name))) for subject_name in subject_list]

#Get a random output
output_name = r01_dataset[0][1].columns[0]

#Set the number of gait fingerptings
num_gf = 3

#Create the personalized kronecker model 
factory = PersonalizedKModelFactory()
personalized_k_model = \
    factory.generate_personalized_model(test_kronecker_model, r01_dataset, output_name, 
                                        num_pca_vectors=num_gf, keep_subject_fit='AB01', left_out_subject='AB01')

print("Personalized the kronecker model")
print(f"The personalized k model shape {personalized_k_model.kmodels[0].model.get_output_size()} and the size of the average model fit {personalized_k_model.kmodels[0].average_fit.shape}")
print(f"The personalized pmap shape {personalized_k_model.kmodels[0].pmap.shape}")
print(f"The personalized gait fingerprint shape {personalized_k_model.kmodels[0].subject_gait_fingerprint.shape}")

#One entry for every basis function and then one gait fingerprint 
test_numpy_array = np.array([1,2,3,4] + [1]*num_gf).reshape(1,-1)
test_numpy_array_zero_gf = np.array([1,2,3,4] + [0]*num_gf).reshape(1,-1)
test_multi_data_array = np.repeat(test_numpy_array,2,axis=0)


#Normal usage when the input array has the gait fingerprint
test1 = personalized_k_model.evaluate(test_numpy_array)
test_zero_gf1 = personalized_k_model.evaluate(test_numpy_array_zero_gf)


#Use the subject average model
test2 = personalized_k_model.evaluate(test_numpy_array,use_average_fit=True)

#These two should be the same since its the same model values and zero gait fingerprint
assert test2 == test_zero_gf1

#Calculate the subject optimal fit
test3 = personalized_k_model.evaluate(test_numpy_array,use_personalized_fit=True)

#These should all be different
assert test1 != test2
assert test1 != test3
assert test2 != test3


#Repeat the test with multi-data array
test_multi_1 = personalized_k_model.evaluate(test_multi_data_array)
test_multi_2 = personalized_k_model.evaluate(test_multi_data_array, use_average_fit=True)
test_multi_3 = personalized_k_model.evaluate(test_multi_data_array, use_personalized_fit=True)

assert test_multi_1[0,:] == test1
assert test_multi_1[1,:] == test1
assert test_multi_2[0,:] == test2
assert test_multi_3[0,:] == test3


print("All tests passed succesfully")