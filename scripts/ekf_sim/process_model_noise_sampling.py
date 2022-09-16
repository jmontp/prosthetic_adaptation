"""
This file is meant to generate a file that contains the process modle 
noise for phase rate, stride length, and ground inclination and store it 
in disk
"""


import numpy as np 
import itertools


num_samples = 5

#Define the ranges of each model noise
phase_noies_values = [0]
phase_rate_noise_values = np.linspace(1e-1,  1e-8, num_samples)
stride_lenght_noise_values = np.linspace(1e-5, 1e-9, num_samples)
ramp_length_noise_values = np.linspace(1e-2,1e-5,num_samples)

#Get the cartesian product of each
list_of_values =list(itertools.product(phase_noies_values, 
                                   phase_rate_noise_values,
                                   stride_lenght_noise_values,
                                   ramp_length_noise_values))

list_of_values_np = np.array(list_of_values)

print(f"{list_of_values_np.shape=}")

np.save('process_model_noise_samples',list_of_values_np)