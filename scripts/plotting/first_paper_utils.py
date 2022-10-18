"""
This file is meant to store helper functions that all plots do 

"""

import pandas as pd 
import numpy as np 
from decimal import Decimal
from scipy.stats import ttest_rel


#Gait state RMSE that we want
gait_states= ['phase', 'stride_length', 'ramp']



#Name the values of the noise parameters in the dataset
phase_rate_noise = 'phase_dotprocess model noise'
stride_length_noise = 'stride_lengthprocess model noise'
ramp_noise = 'rampprocess model noise'

#Create a list of gait state noise parameters
gait_state_noise_list = [phase_rate_noise, stride_length_noise, ramp_noise]





#Integer part of numpy log 10
def fexp(number):
    """Get the exponenet in base 10"""
    #Implement custom log binning
    # this is because of the ordinal data type formatting limitations in altair
    # https://github.com/vega/vega-lite/issues/1763
    # Source:
    ##https://stackoverflow.com/questions/45332056/decompose-a-float-into-
    # mantissa-and-exponent-in-base-10-without-strings
    (sign, digits, exponent) = Decimal(number).as_tuple()
    
    #Calculate the exponent
    exponent=len(digits) + exponent - 1
    
    #Edge case where sometimes 1 will be a recurrent 9,9,9,... in digits
    # causing an error of one
    edge_case_1_digits = (9,9,9,9,9,9)
    if digits[:len(edge_case_1_digits)] == edge_case_1_digits:
        #Add or substract based on the sign
        exponent = exponent - np.sign(exponent)
    
    return np.power(10.0,exponent)


def filter_to_good_range_case(df, filter=True, inplace=False):
    """
    This function will filter the dataframe to only have the good values
    of the phase rmse
    
    This function is also overloaded to do the log binning in the heatmap 
    by setting the inplace to be true and filter to be false
    """

    #Create a new temporary dataframe
    temp_df = pd.DataFrame()

    #Apply function to all noise parameters
    for noise_param in gait_state_noise_list:
        temp_df[noise_param] = df[noise_param].apply(fexp)
    
    
    #Create a base for the detailed analysis
    #Part 2 - select only the ones that we want, chosen by looking at the 
    # heatmap
    phase_noise_good = 1e-6
    stride_length_noise_good = 1e-7
    ramp_noise_good = 1e-4
    good_noise_list = [phase_noise_good, 
                       stride_length_noise_good, 
                       ramp_noise_good]
    
    if filter:
        #Apply function to all noise parameters
        for noise_param, value in zip(gait_state_noise_list, good_noise_list):
            df = df[temp_df[noise_param] == value].reset_index(drop=True)
            temp_df = temp_df[temp_df[noise_param] == value].reset_index(drop=True)
        
    #IF we want in in place, modify the original dataframe
    if inplace:
        #Apply function to all noise parameters
        for noise_param in gait_state_noise_list:
            df[noise_param] = temp_df[noise_param]
    
    
    return df


def filter_to_optimal_case(df, test_type):
    """
    This function will calculate the best condition when only looking at the 
    inter-subject and then filter the dataset based on the best condition
    """
    
    #Filter to get the optimal cases to remove the effect of outliers since
    # we are normalizing the cost function using the maximum error
    df = filter_to_good_range_case(df)
    
    #Create the cost function
    max_phase_rmse = np.max(df['phase'])
    max_stride_length = np.max(df['stride_length'])
    max_ramp = np.max(df['ramp'])
    
    #Create a new column that is the cost function
    df['cost_func'] = np.power(df['phase'] / max_phase_rmse,2) \
                    + np.power(df['stride_length'] / max_stride_length,2) \
                    + np.power(df['ramp'] / max_ramp, 2)
                    
    #Create a pivot table to find the test that has the best average
    # cost function when only looking at the previous method
    # (to remove any possible bias for the new method)
    #Verify if we want to filter
    if test_type == 'all':
        df_filt = df
        # print(f'{len(df_filt)=}, {len(df)=}')
    else: #We want to filter for the supplied test case
        df_filt = df[df['Test']==test_type]

        
    df_agg = (df_filt).groupby(gait_state_noise_list).mean()
    
    #Get the minimum cost condition. 
    min_cost_condition = df_agg['cost_func'].idxmin()
    print(f"Best tune condition for {test_type} is {min_cost_condition}")
    
    #Filter for this condition
    cond_filter = (df[gait_state_noise_list] == min_cost_condition).all(axis=1)
    df = df[cond_filter]
    
    assert len(df) == 20, ("Only one condition must have passed, "
                           "therefore, it should have 20 datapoints")
    
    return df

 
 
 
# def gait_pairwise_difference(df):
    
    # We want to create two plots - one with the units in stride length
    # and one with the units in percent difference. Therefore, calculate the 
    # median value that we will use to scale the axis by grouping across all subjects
    # gait_state_ISA_median = {state:df[df['Test'] == 'ISA'].groupby(gait_state_noise_list).mean()[state].median()
    #                         for state
    #                         in gait_states}
        
#     #Gait states
#     gait_states= ['phase', 'stride_length', 'ramp']


#     #Create names for the dataframe pairwise difference
#     gait_state_diff = [f'{state}_diff' for state in gait_states]
#     gait_state_diff_norm = [f'{state}_diff_norm' for state in gait_states]

#     #Create temp dataframe
#     df2=pd.DataFrame()

#     #Get filters for the tests that are ran
#     ISA = df[df['Test'] == 'ISA'].reset_index(drop=True)
#     NSL = df[df['Test'] == 'NSL'].reset_index(drop=True)

#     #Aggregate t-test results per state
#     df_t_test_result_list = []

#     #Create the pairwise differences for each state
#     for state,df_name,df_name_norm in zip(gait_states,gait_state_diff,
#                                         gait_state_diff_norm):
        
#         #Get the values for this state in each test case
#         ISA_state = ISA[state]
#         NSL_state = NSL[state]
        
#         #Calculate the pairwise difference 
#         df2[df_name] = ISA_state - NSL_state
#         #Calculate the pairwise difference based on scaled by the difference
#         df2[df_name_norm] = (ISA_state - NSL_state) / gait_state_ISA_median[state]
        
#         #Calculate and store p-value
#         t_test_result = ttest_rel(ISA_state, NSL_state)
#         df_t_test_result_list.append(t_test_result)
        
        

#     #Add the subject indicator
#     df2['Subject'] = NSL['Subject']
#     df2[gait_state_noise_list] = NSL[gait_state_noise_list]

#     #Average out the subjects
#     df2 = df2.groupby(gait_state_noise_list, as_index=False).mean()

#     #Rename diff labels
#     df2.rename(columns=change_dict,inplace=True)

#     #Rename to new dataframe    
#     df_output_list.append(df2)

#     #Store the t-test result list
#     t_test_results_list.append(df_t_test_result_list)
        
        
        
if __name__ == '__main__':
    
    for x in range(10):
        print(["{:.2e}".format(fexp(i*float(f"1e-{x}"))) for i in range(10)])