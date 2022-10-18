"""
This is meant to test if the results of the tests are statistically significant

"""

import numpy as np 
import pandas as pd
import scipy.stats

#Load data
df = pd.read_csv('first_paper_NLS_vs_ISA.csv')


#Define the states list
state_list = ['phase', 'phase_dot', 'stride_length', 'ramp']


#Set the names to extract from python 
noise_parameter_names = ['phaseprocess model noise', 
                         'phase_dotprocess model noise',
                         'stride_lengthprocess model noise', 
                         'rampprocess model noise']

#Iterate through the state list
for state in state_list:

    #Get the data for the ISA and NSL test
    ISA_pd = df[(df['Test'] == "ISA") & (df['Subject'] != "AB01")][state].reset_index(drop=True)
    NSL_pd = df[(df['Test'] == "NSL") & (df['Subject'] != "AB01")][state].reset_index(drop=True)
    
    #Get the rows that are null in each of them
    ISA_nan = ~ ISA_pd.isna()
    
    #Remove ISA nans from both test sets
    ISA_pd = ISA_pd[ISA_nan].reset_index(drop=True)
    NSL_pd = NSL_pd[ISA_nan].reset_index(drop=True)
    
    #Get the rows that are null in NSL
    NSL_nan = ~ NSL_pd.isna()
    
    #Remove ISA nans from both test sets
    ISA_pd = ISA_pd[NSL_nan].reset_index(drop=True)
    NSL_pd = NSL_pd[NSL_nan].reset_index(drop=True)
    
    #Get the numpy data from the filtered datasets
    ISA = ISA_pd.values
    NSL = NSL_pd.values
    
    #Calculate the mean and var of the ISA
    ISA_mean = np.mean(ISA)
    
    #Calculate the mean and var of the NSL
    NSL_mean = np.mean(NSL)


    mean_diff = ISA_mean - NSL_mean
    
    stardard_error = np.std(ISA - NSL) / np.sqrt(len(ISA) - 1)

    z_score = mean_diff/stardard_error

    p_value = scipy.stats.t.sf(z_score, len(ISA) - 1)
    
    
    
    #While whe're at it, get the minimum values for each state
    min_phase_index = df[state].idxmin()
    min_calibration = df[noise_parameter_names].iloc[min_phase_index]
    
    #Create a string to report the noise values of the optimal
    noise_names = ' '.join([f"{noise}, {min_calibration[noise]:.5f}" for noise in noise_parameter_names])
    print(f"{state.ljust(20)} {z_score=:.3f}, {p_value=:.3f} {noise_names}")


    
  