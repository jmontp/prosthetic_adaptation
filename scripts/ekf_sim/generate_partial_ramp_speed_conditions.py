"""
Thise file is meant to generate all the partial learning conditions in the
online least squares training. 

"""

import itertools
import pandas as pd
from typing import List, Tuple
import pickle

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
ramp_list_only_positive = [2.5, 5, 7.5, 10.0]
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
                  



condition_3_list = []

#For condition 3, we want to create the three tuples of levelground, ramp1, 
# ramp2 with the negative ramps too. Do this by taking 
ramp_2_combination_list = itertools.combinations(ramp_list_only_positive,2)

for ramp_2_conditions in ramp_2_combination_list:
    condition1 = (ramp_list_levelground[0],speed_list[0])
    condition2 = (ramp_2_conditions[0],speed_list[1])
    condition3 = (-ramp_2_conditions[0],speed_list[1])
    condition4 = (ramp_2_conditions[1],speed_list[2])
    condition5 = (-ramp_2_conditions[1],speed_list[2])
    
    condition_3_list.append((condition1,condition2,condition3,
                             condition4,condition5))



condition_4_list = []

#For condition 3, we want to create the three tuples of levelground, ramp1, 
# ramp2 with the negative ramps too. Do this by taking 
ramp_3_combination_list = itertools.combinations(ramp_list_only_positive,3)

for ramp_3_conditions in ramp_3_combination_list:
    
    
    condition1 = (ramp_list_levelground[0],speed_list[0])
    condition2 = (ramp_3_conditions[0],speed_list[1])
    condition3 = (-ramp_3_conditions[0],speed_list[1])
    condition4 = (ramp_3_conditions[1],speed_list[2])
    condition5 = (-ramp_3_conditions[1],speed_list[2])
    condition6 = (ramp_3_conditions[2],speed_list[1])
    condition7 = (-ramp_3_conditions[2],speed_list[1])
    
    
    condition_4_list.append((condition1,condition2,condition3,
                             condition4,condition5,condition6,
                             condition7))



condition_5_list = []

#For condition 3, we want to create the three tuples of levelground, ramp1, 
# ramp2 with the negative ramps too. Do this by taking 
ramp_4_combination_list = itertools.combinations(ramp_list_only_positive,4)

for ramp_4_conditions in ramp_4_combination_list:
    
    
    condition1 = (ramp_list_levelground[0],speed_list[0])
    condition2 = (ramp_4_conditions[0],speed_list[1])
    condition3 = (-ramp_4_conditions[0],speed_list[1])
    condition4 = (ramp_4_conditions[1],speed_list[2])
    condition5 = (-ramp_4_conditions[1],speed_list[2])
    condition6 = (ramp_4_conditions[2],speed_list[1])
    condition7 = (-ramp_4_conditions[2],speed_list[1])
    condition8 = (ramp_4_conditions[2],speed_list[1])
    condition9 = (-ramp_4_conditions[2],speed_list[0])

    
    condition_5_list.append((condition1,condition2,condition3,
                             condition4,condition5,condition6,
                             condition7,condition8,condition9))


#Do all the conditions
all_conditions = (tuple(itertools.product(ramp_list,speed_list)),)


#Aggregate all the condition lists
# task_data_constraint_list.extend(condition1_list)   #Done
# task_data_constraint_list.extend(condition2_list)   #Done
# task_data_constraint_list.extend(condition_3_list) #Done
# task_data_constraint_list.extend(condition_4_list)
# task_data_constraint_list.extend(condition_5_list)
task_data_constraint_list.extend(all_conditions)

#Aggregate all the condition lists
all_task_data_constraint_list =[]
all_task_data_constraint_list.extend(condition1_list)   #Done
all_task_data_constraint_list.extend(condition2_list)   #Done
all_task_data_constraint_list.extend(condition_3_list) #Done
all_task_data_constraint_list.extend(condition_4_list)
all_task_data_constraint_list.extend(condition_5_list)
all_task_data_constraint_list.extend(all_conditions)

with open('all_test_conditions.pickle','wb') as file: 
    pickle.dump(all_task_data_constraint_list, file)


#Display condition for debugging
log_condition = task_data_constraint_list


#Dfine the number of points in a condition
points_per_step = 150
    
def filter_data_for_condition(df:pd.DataFrame, 
                              condition_list:List[Tuple[float,float]],
                              num_steps:int=None):
    """
    This function will filter a dataset for a list of conditions
    
    Args:
    
    df (pd.DataFrame) - dataset to be filtered
    condition_list (List) - list of tuples in the form of (ramp, speed )
    num_steps - number of steps that will be in each condition
    """
    
    #Create a dataframe for every condition in the condition list
    df_list = []
    #Iterate through all the pairwise ramp and speed condition
    for condition in condition_list:
        #Get each condition
        ramp, speed = condition
        #Filter the dataframe 
        df_filtered = df[(df['ramp']==ramp) & (df['speed']==speed)]
        
        #Remove steps
        if num_steps is not None: 
            df_filtered = df_filtered.iloc[:num_steps*points_per_step]
        
        # and add it to the list of filtered dataframe
        df_list.append(df_filtered)
    
    #Expand the list to a pandas concatenate to aggregate all the filtered data
    return pd.concat(df_list)





if __name__ == '__main__':
    
    #If you run the function as the main code, print out the conditions that 
    # we are generating
    
    print(f"Number of conditoins is {len(log_condition)}")
    print("Conditions:")
    for cond in log_condition:
        print(cond)
        