"""
Thise file is meant to generate all the partial learning conditions in the
online least squares training. 

"""

import itertools
import pandas as pd
from typing import List, Tuple

###############################################################################
###############################################################################
# Create the conditions that we want to run the test

#Create a condition list to train all the datasets on 
# this will be a list of lists where the internal list will have the tuples of
# ramp, speed that you want to filter by
task_data_constraint_list = []

#Define lists for the conditions in the dataset 
# (yes its easier in text than range())
ramp_list_no_levelground = [-10.0, -7.5, -5.0, -2.5, 2.5, 5, 7.5, 10.0]
ramp_list_levelground = [0.0]
ramp_list = ramp_list_levelground + ramp_list_no_levelground
speed_list = [0.8, 1.0, 1.2]
normal_speed_list = [1.0]

#First condition - just levelground at one speed with one extra ramp at 
# normal speed
# This should have 3*8 conditions = 24
#Add the leveground conditions
condition1_levelground = list(itertools.product(ramp_list_levelground, speed_list))
#Add the other ramp conditions
condition1_ramps = list(itertools.product(ramp_list_no_levelground,normal_speed_list))

#Aggregate every levelground condition to another ramp condition.
condition1_list = list(itertools.product(condition1_levelground,condition1_ramps))


#Second Condition - just levelground at one speed with one extra ramp at 
# normal speed. The second ramp is just the negative of the first ramp

#Create an empty list so that we can fill it up
condition2_list = []

#Sadly, we need to do some indexing to remove duplicates
# therefore we can't do it in one line using list comprehension.
# Condition 2 is just a subset of condition 1 since ramps with same magnitude
# are now the same condition
for index, (cond1, cond2)  in enumerate(condition1_list):
    #If the index is greater than halfway, the ramps start repeating
    # therefore skip those values
    if index % 8 < 4:
        condition2_list.append((cond1, cond2, (-cond2[0],cond2[1])))
                  

#Aggregate all the condition lists
task_data_constraint_list.extend(condition1_list)
task_data_constraint_list.extend(condition2_list)


#Display condition for debugging
log_condition = task_data_constraint_list
    
    
def filter_data_for_condition(df:pd.DataFrame, 
                              condition_list:List[Tuple[float,float]]):
    """
    This function will filter a dataset for a list of conditions
    
    Args:
    
    df (pd.DataFrame) - dataset to be filtered
    condition_list (List) - list of tuples in the form of (ramp, speed )
    
    """
    
    #Create a dataframe for every condition in the condition list
    df_list = []
    #Iterate through all the pairwise ramp and speed condition
    for condition in condition_list:
        #Get each condition
        ramp, speed = condition
        #Filter the dataframe 
        df_filtered = df[(df['ramp']==ramp) & (df['speed']==speed)]
        # and add it to the list of filtered dataframe
        df_list.append(df_filtered)
    
    #Expand the list to a pandas concatenate to aggregate all the filtered data
    return pd.concat(*df_list)





if __name__ == '__main__':
    
    #If you run the function as the main code, print out the conditions that 
    # we are generating
    
    print(f"Number of conditoins is {len(log_condition)}")
    print("Conditions:")
    for cond in log_condition:
        print(cond)
        