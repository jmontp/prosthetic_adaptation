#Imoprt the kronecker model
from context import k_model
from k_model import KroneckerModel


#Import the function bases that will be used to create the functions
from context import function_bases
from function_bases import FourierBasis, HermiteBasis

#Common imports
import numpy as np
import pandas as pd

#Initialize the random basis functions
phase_basis = FourierBasis(1,'phase')
phase_dot_basis = HermiteBasis(2,'phase_dot')
ramp_basis = HermiteBasis(6,'ramp')
stride_basis = HermiteBasis(2,'stride_length')

basis_list = [phase_basis, phase_dot_basis, ramp_basis, stride_basis]

#Create the module 
test_kronecker_model = KroneckerModel(basis_list)

#Numpy array test input
test_array = np.array([1,2,3,4]).reshape(1,-1)

#Dataframe test input
test_df = pd.DataFrame(test_array, columns = ["phase",'phase_dot','ramp','stride_length'])

#Make sure that the evaluate function works with both np array and dataframe
arr1 = test_kronecker_model.evaluate(test_array)
arr2 = test_kronecker_model.evaluate(test_df)

#Both arrays should be the same
print(f"Test Array 1 == Test Array 2: {np.all(arr1 == arr2)}")

#The stored size must be the same as the output
print(f"Size must match up with stored size {test_kronecker_model.get_output_size()} vs {arr1.shape[1]}: {test_kronecker_model.get_output_size() == arr1.shape[1]}")


#Test that the output of the evaluate function is correct with one basis
k_model_just_fourier = KroneckerModel([phase_basis])

output1 = k_model_just_fourier.evaluate(np.zeros((1,1)))
expected_output1 = np.array([1,0,1]).reshape(1,-1)

print(f"The expected output 1 is {expected_output1} and we got {output1} are they the same? {np.all(output1 == expected_output1)}")

#Test with multiple basis
k_model_two_basis = KroneckerModel([phase_basis,phase_dot_basis])

output2 = k_model_two_basis.evaluate(np.zeros((1,2)))
expected_output2 = np.array([1,0,1] + [0]*int(k_model_two_basis.get_output_size()/2)).reshape(1,-1)

print(f"The expected output 2 is {expected_output2} and we got {output2} are they the same? {np.all(output2 == expected_output2)}")
