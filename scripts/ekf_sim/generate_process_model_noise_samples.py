"""
This file is meant to generate a file that contains the process modle 
noise for phase rate, stride length, and ground inclination and store it 
in disk
"""


import numpy as np 
import itertools


num_samples = 9

#Define the ranges of each model noise
phase_noies_values = [0]
phase_rate_noise_values = [1*np.power(10.0,-i) for i in range(0,9)]
stride_length_noise_values = [1*np.power(10.0,-i) for i in range(3,11)]
ramp_length_noise_values = [1*np.power(10.0,-i) for i in range(1,8)]

#Get the cartesian product of each
list_of_values =list(itertools.product(phase_noies_values, 
                                   phase_rate_noise_values,
                                   stride_length_noise_values,
                                   ramp_length_noise_values))

list_of_values_np = np.array(list_of_values)

print(f"{list_of_values_np.shape=}")

np.save('process_model_noise_samples',list_of_values_np)