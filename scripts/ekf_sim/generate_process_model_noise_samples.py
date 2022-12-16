"""
This file is meant to generate a file that contains the process modle 
noise for phase rate, stride length, and ground inclination and store it 
in disk
"""


import numpy as np 
import itertools


num_samples = 9

#Define the ranges of each model noise


#Part 1 - do different order of magnitudes
# phase_noies_values = [0]
# phase_rate_noise_values = [1*np.power(10.0,-i) for i in range(0,9)]
# stride_length_noise_values = [1*np.power(10.0,-i) for i in range(3,11)]
# ramp_length_noise_values = [1*np.power(10.0,-i) for i in range(1,8)]


#Part 2 - linearly interpolate in optimal values

#Define the optimal magnitudes for each power
phase_rate_optimal = 1e-6
stride_length_optimal = 1e-7
ramp_optimal = 1e-4

#Create a range of values that will be multiplied by the optimal magnitudes
spread = np.linspace(0.1,10,6)

#Create the samples that we will test in
phase_noies_values = [0]
phase_rate_noise_values = phase_rate_optimal * spread
stride_length_noise_values = stride_length_optimal * spread
ramp_length_noise_values = ramp_optimal * spread


#Get the cartesian product of each
list_of_values =list(itertools.product(phase_noies_values, 
                                   phase_rate_noise_values,
                                   stride_length_noise_values,
                                   ramp_length_noise_values))

list_of_values_np = np.array(list_of_values)

print(f"{list_of_values_np.shape=}")
print(f"{list_of_values_np=}")

np.save('process_model_noise_samples',list_of_values_np)