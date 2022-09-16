"""
This file is meant to split the dataport dataset into training and validation
datasets to minimize the chance that the training data is used to 
validate the algorithm
"""

import pandas as pd
import numpy as np 
from itertools import product

#Get the relative file location
original_file_location = ("../../data/flattened_dataport/"
                          "dataport_flattened_partial_{}.parquet")
training_file_location = ("../../data/flattened_dataport/training/"
                          "dataport_flattened_training_{}.parquet")
validation_file_location = ("../../data/flattened_dataport/validation/"
                          "dataport_flattened_validation_{}.parquet")


#Generate a list of subjects to lead each file in
subject_list = [f"AB{i:02}" for i in range(1,11)]


#Establish the training percentage
training_percentage = 0.7

#Get the minimum steps in the validation step, start large so subsequents
# comparisons immediately overide it
min_steps_validation = float('inf')
#Create a dict to se overall min steps per person    
min_steps_per_person = {}

#For each subject, divide into training and validation
for subject_index, subject in enumerate(subject_list):
    
    #Get status feedback 
    print(f"Doing {subject}")
    
    #Get the file name
    file_name = original_file_location.format(subject)
    
    #Load the subject
    data = pd.read_parquet(file_name)
    
    #Get the unique ramp and speed combinations
    ramp_list = data.ramp.unique()
    speed_list = data.speed.unique()
    
    #Get all the permutations
    ramp_speed_permutation = list(product(ramp_list, speed_list))
    
    #Build the validation set
    training_set_list = []
    validation_set_list = []
    
    #Find the minimum steps for a person
    min_data_in_person = float('inf')
    
    #Iterate through the speeds to find the minimum conditions
    for ramp, speed in ramp_speed_permutation:
        #Get the subset of the data for these conditions
        filtered_data = data[(data.ramp == ramp) & (data.speed == speed)]
        
        #Get the minimum number
        min_data_in_person = min(min_data_in_person, len(filtered_data))
        
        #Print the subject and the minimum data in the condition
        # print(f"{subject} Ramp {ramp} Speed {speed} "
        #       f"Min steps {min_data_in_person*(1-training_percentage)/150}")
        
        #Print out how many steps a person has
        print(f"{subject} Ramp {ramp} Speed {speed} "
              f"Min Steps per Condition {min_data_in_person/150}")
        
        
    #Get the min steps per person for debugging only
    min_steps_per_person[subject] = min_data_in_person*(1-training_percentage)\
                                    /150
    
    
    #Calculate the filtered data set per condition
    for ramp, speed in ramp_speed_permutation:
        
        #Get the subset of the data for these conditions
        #Filter based on minimum step so all conditions are balanced
        # filtered_data = data[(data.ramp == ramp) & (data.speed == speed)]\
        #     [:min_data_in_person]
        
        #Not filtering based on minimum number of steps since we are storing
        # the number of steps in the condition to perform weighted least square
        filtered_data = data[(data.ramp == ramp) & (data.speed == speed)]
        
        #Get the index to split which is based on size of filtered data, 
        split_index = int(training_percentage*len(filtered_data))
        
        #Use modulus 150 to make sure we have complete steps
        split_index = split_index - split_index%150
        
        #Split data into training and validation
        training_split, validation_split = np.split(filtered_data, 
                                                    [split_index])

        #Add the amount of steps in the validation to perform weighted least 
        # squares. The steps do not need to be normalized
        if len(filtered_data) == None: 
            raise ValueError("filtered data has no length")
        
        #Verify if there are nan values in the dataset
        if training_split.isna().sum().sum() > 0:
            raise ValueError("Nans in the dataset")
        
        training_split['Steps in Condition'] = 1/len(filtered_data)

        #Print debug information
        # if subject_index == 0:
        #     print(f"Ramp {ramp}, Speed {speed}, df length {len(filtered_data)}"
        #           f", training length {len(training_split)/150}"
        #           f", validation length {len(validation_split)/150}")  
            
        #Calculate the minimum validation steps
        min_steps_validation = min(min_steps_validation, 
                                   len(validation_split)/150)
        
        #Add to the lists
        training_set_list.append(training_split)
        validation_set_list.append(validation_split)
    
    
    
    #First define what phase is
    phase_list = np.linspace(0,1,150)

    #Create combinations of phase and ramp
    phase_ramp_combinations = np.array(list(product(phase_list,ramp_list)))

    #Get the number of fake datapoints to use to generate other states later
    # This adds up to 9 steps of fake walking data (one per ramp condition)
    num_fake_datapoints = phase_ramp_combinations.shape[0]

    #Create the fake data frame to population additional fields
    fake_data_df = pd.DataFrame(phase_ramp_combinations,columns=['phase','ramp'])

    #Phase dot is not part of the model so set any value just so that it works
    fake_data_df['phase_dot'] = [0]*num_fake_datapoints
    #The data is meant so simulate when stride length is equal to 0, so set it to 0
    fake_data_df['stride_length'] = [0]*num_fake_datapoints 
    #Set the values of the joint angles
    fake_data_df['jointangles_thigh_x'] = [-24]*num_fake_datapoints
    fake_data_df['jointangles_shank_x'] = [-89]*num_fake_datapoints
    fake_data_df['jointangles_foot_x'] = fake_data_df['ramp']
    fake_data_df['Steps in Condition'] = 1/90.0
        
    #Add to the training set
    training_set_list.append(fake_data_df)
    
    #Create the training set by aggregating the lists
    training_set = pd.concat(training_set_list, ignore_index=True)
    
    #Create the validation set by aggregating the lists
    validation_set = pd.concat(validation_set_list, ignore_index=True)
    
    #Save the training set
    training_file_name = training_file_location.format(subject)
    training_set.to_parquet(training_file_name)
    
    #Save the validation set
    validation_file_name = validation_file_location.format(subject)
    validation_set.to_parquet(validation_file_name)
    
print(f"Minimum validation steps {min_steps_validation}")
print(min_steps_per_person)