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


#Iterate through the state list
for state in state_list:

    
    ISA = df[(df['Test'] == "ISA") & (df['Subject'] != "AB01")][state].values
    NSL = df[(df['Test'] == "NSL") & (df['Subject'] != "AB01")][state].values
    
    
    #Calculate the mean and var of the ISA
    ISA_mean = np.mean(ISA)
    
    #Calculate the mean and var of the NSL
    NSL_mean = np.mean(NSL)


    mean_diff = ISA_mean - NSL_mean
    
    stardard_error = np.std(ISA - NSL) / np.sqrt(len(ISA) - 1)

    z_score = mean_diff/stardard_error

    p_value = scipy.stats.t.sf(z_score, len(ISA) - 1)
    

    print(f"{state.ljust(20)} {z_score=:.3f}, {p_value=:.3f}")