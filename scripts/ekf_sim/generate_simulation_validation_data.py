"""
This file is meant to generate the data for the validation tests

It allows an user to specify the specific ramp and belt speed conditions 
it wants to test while also having a default condition and being able to 
randomize them

"""


import numpy as np 
import pandas as pd
import random
import itertools

from typing import List, Tuple



def generate_random_condition(num_conditions:int=0):
    """
    This function will generate a random list of (ramp,speed) tuples 
    that are contained in Kyle Embry's dataset. 

    Args:
        num_conditions (int, optional): The number of random. Defaults to 0.
        conditions that are desired. If set to 0 or lower, then it will 
        generate a list with all the conditions (27).
        
    Returns:
        condition_list_to_use (List[Tuple[ramp,speed]])
    """
    
    #Create a list of the conditions in the dataset
    ramp_list = [-10,-7.5,-5.0,-2.5,0,2.5,5.0,7.5,10.0]
    speed_list = [0.8,1.0,1.2]
    #Create the product of the two lists
    ramp_speed_list = list(itertools.product(ramp_list, speed_list))
    total_conditions = len(ramp_speed_list)
    if (num_conditions > 0):
        #Use only a subset of the list
        index_list = random.sample(range(total_conditions), num_conditions)
    if (num_conditions < 1):
        #Use all the dataset conditions
        index_list = random.sample(range(total_conditions), total_conditions)

    #Generate a random configuration
    condition_list_to_use = [ramp_speed_list[i] for i in index_list]
    
    return condition_list_to_use


def generate_data(subject:str,
                  state_list:List[str],
                  joint_list:List[str],
                  condition_list: List[Tuple] = None,
                  default_list: bool = None, 
                  random_list_size: int = None
                  ):
    """
    This function will generate a dataset based on the kyle embry data
    for the given dataset condition configuration that has been setup.
    
    Args:
        subject (str): subject to load the data for
        state_list (List[str]): list of strings to indicate the states in the
            dataset.
        joint_list (List[str]): list of strings that indiciate the joints to 
            use in the dataset.
        conditions_list (List[Tuple], optional): list of (ramp, belt speed) 
            tuples. 
            Defaults to None. 
        default_list (bool, optional): determines if the default condition 
            will be used. 
            Defaults to None.
        random_list (bool, optional): creates a random list of conditions 
            from all the dataset. If the int is greater than zero then it will
            use that many conditions. The max different conditions is 27.
            If greater than 27 then conditions will be repeated.
            Defaults to None.
            
    Returns:
        ground_truth_states (np.ndarray): Ground truth states for the 
            joints specified
        ground_truth_sensor_data (np.ndarray): Ground truth sensor data for
            the joints specified
    """
    
    #Number one priority is the specified list
    if (condition_list is not None):
        
        #Use the user given condition list
        condition_list_to_use = condition_list
    
    #Number two priority is the random list
    elif (random_list_size is not None):
        
        #Generate a random list from the above helper function
        condition_list_to_use = generate_random_condition(random_list_size)
        
    
    #Third priority is just using the default list
    elif (default_list is True):
         
        #Number three priority is the default list
        condition_list_to_use = [
                        (0.0, 0.8),
                        (0.0, 1.0),
                        (0.0, 1.2),
                        (-2.5,1.2),
                        (-5,1.2),
                        (-7.5,1.2),
                        (-10,0.8),
                        (-7.5,0.8),
                        (-5,1.2),
                        (-2.5,0.8),
                        (0.0, 0.8),
                        (2.5,1.2),
                        (5,1.2),
                        (7.5,1.2),
                        (10,0.8),
                        (7.5,0.8),
                        (5,1.2),
                        (2.5,0.8),
                        (0.0, 1.2),
                        (-7.5,0.8),
                        (10,0.8),
                        ]

    #If none of these conditions are met, the function is not being used 
    # correctly. Throw an error
    else:
        raise ValueError("You did not use the inputs correctly")
    
    
    ### Load the datasets
    #File relative imports
    # file_location = \
    # '../../data/r01_dataset/r01_Streaming_flattened_{}.parquet'
    file_location = ("../../data/flattened_dataport/validation/"
                     "dataport_flattened_validation_{}.parquet")

    #Get the file for the corresponding subject
    filename = file_location.format(subject)
    # print(f"Looking for {filename}")

    #Read in the parquet dataframe
    total_data = pd.read_parquet(filename)
    
    
    # print(total_data.columns)
    #Define the steps per conditoin
    num_steps_per_condition = 10
    points_per_step = 150
    points_per_condition = num_steps_per_condition * points_per_step
    num_trials = len(condition_list_to_use)

    #Calculate the total number of datapoints
    datapoints = points_per_condition * num_trials

    #Add velocity to the joint list
    joint_velocity_list = ['_'.join(joint.split('_')[:2] 
                               + ['dot'] 
                               + joint.split('_')[2:])
                      for joint
                      in joint_list]
    
    #Create a list that has both the joint and the joint velocity
    total_joint_list = joint_list + joint_velocity_list
    
    #Pre-allocate memory
    state_data_np = np.zeros((datapoints,len(state_list)))
    joint_data_np = np.zeros((datapoints,len(total_joint_list)))

    #Skip steps, don't do this by default
    skip = 0
    #this subject's data is very noisy in these conditions, skip
    if(subject == "AB02"):
        skip = 1

    #Create the data array based on the setup above
    for i,condition in enumerate(condition_list_to_use):
        
        #Get the desired incline and speed from the condition
        ramp, speed = condition

        #Get the filtered data based on the condition
        mask = (total_data['ramp'] == ramp) & (total_data['speed'] == speed)
        filtered_data = total_data[mask]

        #Verify that you have enough steps to validate this condition
        if (len(filtered_data) > points_per_condition):
                
            #Get the sensor data
            state_data_np[i*points_per_condition: 
                          (i+1)*points_per_condition,:] = \
                filtered_data[state_list]\
                    .values[skip*points_per_condition:
                            (skip+1)*points_per_condition,:]

            #Get the ground truth data
            joint_data_np[i*points_per_condition: 
                          (i+1)*points_per_condition,:] = \
                filtered_data[total_joint_list]\
                    .values[skip*points_per_condition:
                            (skip+1)*points_per_condition,:]
        
        #If you don't have enough data per condition, just repeat it
        else:
            #Get the number of times that the datset needs to be repeated
            len_filt_data = len(filtered_data)
            repeats = points_per_condition/len_filt_data
            
            #Get the integer part to loop over the set
            integer_repeat = int(repeats)
                    
            #Get the fractional part to fill the remaining step repeatitions
            fractional_part = repeats%1
            
            #Repeat sensor data
            repeated_state_data = np.concatenate(
                [np.repeat(filtered_data[state_list].values, 
                           integer_repeat,axis=0),
                filtered_data[state_list]\
                    .values[:int(fractional_part*len_filt_data),:]]
            ,axis=0)
            
            #Repeat State Data
            state_data_np[i*points_per_condition: 
                          (i+1)*points_per_condition,:] = repeated_state_data
                
                
            #Repeat ground truth data
            repeated_joint_data = np.concatenate(
                [np.repeat(filtered_data[total_joint_list].values,
                           integer_repeat,axis=0),
                filtered_data[total_joint_list]\
                    .values[:int(fractional_part*len_filt_data)]
                ]
            ,axis=0)
            
            joint_data_np[i*points_per_condition: \
                          (i+1)*points_per_condition,:] = repeated_joint_data

        
    return state_data_np, joint_data_np,\
           num_steps_per_condition, condition_list_to_use