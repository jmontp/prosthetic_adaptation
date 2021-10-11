#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:17:32 2021

@author: jmontp
"""

# %%

from os import remove
import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame
from functools import lru_cache


def get_column_name(column_string_list, num_last_keys):
    filter_strings = ['right', 'left']
    filtered_list = filter(
        lambda x: not x in filter_strings, column_string_list)
    column_string = '_'.join(filtered_list)
    return column_string


def get_end_points(d, out_dict, parent_key='', sep='/', num_last_keys=4):

    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, h5py._hl.group.Group):
            get_end_points(v, out_dict, new_key, sep=sep)
        # Where the magic happens when you reach an end point
        else:
            column_string_list = (parent_key+sep+k).split(sep)[-num_last_keys:]
            column_string = get_column_name(column_string_list, num_last_keys)

            if (column_string not in out_dict):
                out_dict[column_string] = [[new_key], [v]]
            else:
                out_dict[column_string][0].append(new_key)
                out_dict[column_string][1].append(v)

#Debug where the bad strides are in the datasets
def determine_zero_data_strides():
    pass
    #%%
    file_name = '../local-storage/InclineExperiment.mat'
    h5py_file = h5py.File(file_name)['Gaitcycle']

    # Iterate through all the subjects, make a file per subject to keep it RAM bound
    for subject in h5py_file.keys():
        if subject != 'AB05':
            continue

        # Initialize variables for the subject
        data = h5py_file[subject]
        save_name = '../local-storage/test/dataport_flattened_partial_{}.parquet'.format(
            subject)
        # Store all the end points
        columns_to_endpoint_list = {}
        get_end_points(data, columns_to_endpoint_list)
        
        for joint in columns_to_endpoint_list.keys():
            joint_data_trial_name = zip(*columns_to_endpoint_list[joint])
            total_datasets = len(columns_to_endpoint_list[joint][1])
            sum = 0
            bad_strides = 0
            for num_dataset,trial_name_dataset in enumerate(joint_data_trial_name):
                trial_name, dataset = trial_name_dataset
                total_rows = dataset.shape[0]
                for row_number, row in enumerate(dataset):
                    num_digits = np.count_nonzero(row)
                    if(num_digits == 0):
                        bad_strides += 1
                        if(row_number+1 != total_rows and "forceplate" not in joint):
                            print(subject + " " + joint + " dataset " + trial_name + " " + str(num_dataset) + "/" + str(total_datasets) + " bad row: " + str(row_number + 1) + "/" + str(total_rows))
                    sum += 1
            #print(subject + " " + joint + " total bad strides = " + str(bad_strides) + " Total strides: " + str(sum))

# This is a helper function to determine where trials have different strides
def determine_different_strides():
    pass
    #%%
    feature1 = 'jointangles_ankle_x'
    feature2 = 'jointmoment_knee_x'

    print("Comparing " + feature1 + " to " + feature2)

    file_name = '../local-storage/InclineExperiment.mat'
    h5py_file = h5py.File(file_name)['Gaitcycle']

    bad_datasets = []

    # Iterate through all the subjects, make a file per subject to keep it RAM bound
    for subject in h5py_file.keys():

        if(subject != "AB05"):
            continue

        # Initialize variables for the subject
        data = h5py_file[subject]

        # Store all the end points
        columns_to_endpoint_list = {}
        get_end_points(data, columns_to_endpoint_list)


        #Get the data for the features that we want
        data_feature1 = columns_to_endpoint_list[feature1]
        data_feature2 = columns_to_endpoint_list[feature2]

        bad_run_filter = lambda x: (x[:,:],[]) if (np.count_nonzero(x[-1,:]) or np.count_nonzero(x[-2,:]) == 0) else (x[:-1,:],[x.shape[0]-1])


        #Update the dataset based on the filter implementation
        data_feature1[1] = [bad_run_filter(x)[0] for x in data_feature1[1]]
        data_feature2[1] = [bad_run_filter(x)[0] for x in data_feature2[1]]


        #Create a mapping from trial to trial data shape
        #Initialzie to zero
        trial_length_dict_feature1 = {
            x.split('/')[0] + " " + x.split('/')[-3]: [] for x in data_feature1[0]}
        trial_length_dict_feature2 = {
            x.split('/')[0] + " " + x.split('/')[-3]: [] for x in data_feature2[0]}

        trial_to_dataset = {}

        #Initialize the dictionary taking into conisderation left and right legs
        for trial_long,data in zip(*data_feature1):
            trial = trial_long.split('/')[0] + " " + trial_long.split('/')[-3]
            trial_length_dict_feature1[trial].append(data)
        
        for trial_long,data in zip(*data_feature2):
            trial = trial_long.split('/')[0] + " " + trial_long.split('/')[-3]
            trial_length_dict_feature2[trial].append(data)


        sum_len1 = 0
        sum_len2 = 0

        #Verify each trial shape
        for trial in trial_length_dict_feature1.keys():
            
            trial_data_pair = zip(trial_length_dict_feature1[trial],
                                trial_length_dict_feature2[trial])

            for single_data_trial1, single_data_trial2 in trial_data_pair:
                
                len1 = single_data_trial1.shape[0]*single_data_trial1.shape[1]
                len2 = single_data_trial2.shape[0]*single_data_trial2.shape[1]


                if len1 != len2:
                    bad_datasets.append((single_data_trial1,single_data_trial2))
                    pass
                    print("!!!!!!!!!!!!!!!!! This trial does not match " + subject + " " + trial + " len1 " + str(len1) + " len2 " + str(len2))
                else:
                    pass
                    print("Good " + subject + " " + trial + " len1 " + str(len1) + " len2 " + str(len2))

    for dataset_pair in bad_datasets:
        print(np.count_nonzero(np.array(dataset_pair[1]).flatten()[-150:]))

    bad_datasets_np = [ (x[:,:], y[:,:]) for x,y in bad_datasets]
    bad_datasets_np.insert(0,(feature1,feature2))


    #Conclusion, there are datasets that have zero final stride inconsistently
    # I found a way to identify them
    # Need to implement into the dataset flattening technique
    # I think they are all on the left leg
#%%

def quick_flatten_dataport():
    pass
    # %%
    file_name = '../local-storage/InclineExperiment.mat'
    h5py_file = h5py.File(file_name)['Gaitcycle']

    # Iterate through all the subjects, make a file per subject to keep it RAM bound
    for subject in h5py_file.keys():
        
        #Uncomment if you want to debug a specific subject
        # if subject != "AB05":
        #     continue


        print("Flattening subject: " + subject)
        # Initialize variables for the subject
        data = h5py_file[subject]
        save_name = '../local-storage/test/dataport_flattened_partial_{}.parquet'.format(
            subject)
        # Store all the end points
        columns_to_endpoint_list = {}
        get_end_points(data, columns_to_endpoint_list)
        # This dictionary stores dataframes based on the amount of strides that
        # they have
        strides_to_dataframes_dict = {}
        # Which column will be used to get information about each row
        selected_column = 'jointangles_ankle_x'

        # Main loop - process each potential column
        for column_name, endpoint_list in columns_to_endpoint_list.items():

            # If the enpoints contain any of this, ignore the endpoint
            if('subjectdetails' in column_name or
                'cycles' in column_name or
                'stepsout' in column_name or
                'description' in column_name or
                'mean' in column_name or
                    'std' in column_name):
                #print(column_name + " " +       str(len(endpoint_list[1])) + " (ignored)")
                continue

            # Else: this is a valid column
            #print(column_name + " " + str(len(endpoint_list[1])))

            # Filter the endpoint list for bad 
            #This version removes elements just in the end
            bad_run_filter = lambda x: (x[:,:],[]) if (np.count_nonzero(x[-1,:]) or np.count_nonzero(x[-2,:]) == 0) else (x[:-1,:],[x.shape[0]-1])
            
            #This version removes elements in the middle 
            # def bad_run_filter(trial_dataset):
            #     remove_list = []
            #     for row_index,row in enumerate(trial_dataset):
            #         num_digits = np.count_nonzero(row)
            #         if(num_digits == 0):
            #             remove_list.append(row_index)
                
            #     #Remove elements
            #     # if remove list is empty, nothing is deleted
            #     return np.delete(trial_dataset[:,:], remove_list, axis=0), remove_list

            endpoint_list_filtered = [bad_run_filter(x)[0] for x in endpoint_list[1]]
        
            # Get the data to add it to a dataframe
            data_array = np.concatenate(endpoint_list_filtered, axis=0).flatten()


            # Calculate how many strides are in the dataframe
            len_arr = data_array.shape[0]
            len_key = len_arr/150.0

            # Add the key to the dataframe
            try:
                strides_to_dataframes_dict[len_key][column_name] = data_array
            except KeyError:
                strides_to_dataframes_dict[len_key] = DataFrame()
                strides_to_dataframes_dict[len_key][column_name] = data_array

        # All the dataframes have been created, add information about phase and task

        # Helper functions to get time, ramp to append task information to dataframe

        @lru_cache(maxsize=5)
        def get_time(trial, leg):
            return data[trial]['cycles'][leg]['time']

        @lru_cache(maxsize=5)
        def get_ramp(trial):
            return data[data[trial]['description'][1][1]][0][0]

        @lru_cache(maxsize=5)
        def get_speed(trial):
            return data[data[trial]['description'][1][0]][0][0]

        # Iterate by row to get phase information

        # Ugly but effective
        # We need to use the unfiltered version to get the remove_list again
        # This is used to filter the time column
        endpoint_list = columns_to_endpoint_list[selected_column]

        # Create lists to store all the phase dot and stride length information
        trials = []
        legs = []
        phase_dot_list = []
        stride_length_list = []

        # Iterate by trial to get time, speed
        for experiment_name, dataset in zip(*endpoint_list):
            
            filtered_dataset, remove_list = bad_run_filter(dataset)
            
            endpoint_split = experiment_name.split('/')

            trial = endpoint_split[0]
            leg = endpoint_split[-3]

            trials.extend([trial]*filtered_dataset.shape[0]*filtered_dataset.shape[1])
            legs.extend([leg]*filtered_dataset.shape[0]*filtered_dataset.shape[1])

            time = get_time(trial, leg)
            speed = get_speed(trial)

            #Filter out times that are not being used since there is no data
            time = np.delete(time,remove_list,axis=0)


            time_delta = (time[:, -1]-time[:, 0])
            phase_dot_list.append(np.repeat(1/time_delta, 150))
            stride_length_list.append(np.repeat(speed*time_delta, 150))

        # Get the corresponding dataframe to the selected column
        df = None
        for dataframe in strides_to_dataframes_dict.values():
            if selected_column in dataframe.columns:
                df = dataframe

        # print(len(trials))
        # print(df.shape[0])

        df['ramp'] = [get_ramp(trial) for trial in trials]
        df['speed'] = [get_speed(trial) for trial in trials]
        df['trial'] = trials
        # We don't want phase to reach one because 0=1 in terms of phase
        phase = np.linspace(0, (1-1/150), 150)
        df['leg'] = legs
        df['phase'] = np.tile(phase, int(df.shape[0]/150))
        df['phase_dot'] = np.concatenate(phase_dot_list, axis=0)
        df['stride_length'] = np.concatenate(stride_length_list, axis=0)
        print("Number of columns: " + str(len(df.columns)))
        print("Columns: " + df.columns)
        print("strides to length " + str([(strides,len(dataset.columns)) for strides, dataset in strides_to_dataframes_dict.items()]))
        # Comment out to not save
        df.to_parquet(save_name)

        # Uncomment break to just get one person
        #break
# %%


def add_global_shank_angle():
    pass
# %%
    # #Get the subjects
    subjects = [
        ('AB10', '../local-storage/test/dataport_flattened_partial_AB10.parquet')]
    for i in range(1, 10):
        subjects.append(
            ('AB0'+str(i), '../local-storage/test/dataport_flattened_partial_AB0'+str(i)+'.parquet'))

    for subject in subjects:
        df = pd.read_parquet(subject[1])
        print(df.columns)

        # Create the shank angles based on foot and ankle

        # df.drop(columns=['jointangle_shank_x','jointangle_shank_y','jointangle_shank_z'])
        df['jointangles_shank_x'] = df['jointangles_foot_x'] + \
            df['jointangles_ankle_x']
        df['jointangles_shank_y'] = df['jointangles_foot_y'] + \
            df['jointangles_ankle_y']
        df['jointangles_shank_z'] = df['jointangles_foot_z'] + \
            df['jointangles_ankle_z']

        # Create the thigh angle based on pelvis and ip
        df['jointangles_thigh_x'] = df['jointangles_pelvis_x'] + \
            df['jointangles_hip_x']
        df['jointangles_thigh_y'] = df['jointangles_pelvis_y'] + \
            df['jointangles_hip_y']
        df['jointangles_thigh_z'] = df['jointangles_pelvis_z'] + \
            df['jointangles_hip_z']

        # Calculate the derivative of foot dot manually
        shank_anles_cutoff = df['jointangles_shank_x'].values[:-1]
        shank_angles_future = df['jointangles_shank_x'].values[1:]
        phase_rate = df['phase_dot'].values[:-1]
        measured_shank_derivative = (
            shank_angles_future-shank_anles_cutoff)*(phase_rate)*150
        measured_shank_derivative = np.append(measured_shank_derivative, 0)
        df['jointangles_shank_dot_x'] = measured_shank_derivative

        # Calculate the derivative of foot dot manually
        foot_anles_cutoff = df['jointangles_foot_x'].values[:-1]
        foot_angles_future = df['jointangles_foot_x'].values[1:]
        measured_foot_derivative = (
            foot_angles_future-foot_anles_cutoff)*(phase_rate)*150
        measured_foot_derivative = np.append(measured_foot_derivative, 0)
        df['jointangles_foot_dot_x'] = measured_foot_derivative

        # Calculate the derivative of knee dot manually
        anles_cutoff = df['jointangles_knee_x'].values[:-1]
        angles_future = df['jointangles_knee_x'].values[1:]
        measured_foot_derivative = (
            angles_future-anles_cutoff)*(phase_rate)*150
        measured_foot_derivative = np.append(measured_foot_derivative, 0)
        df['jointangles_knee_dot_x'] = measured_foot_derivative

        # Calculate the derivative of hip dot manually
        anles_cutoff = df['jointangles_hip_x'].values[:-1]
        angles_future = df['jointangles_hip_x'].values[1:]
        measured_foot_derivative = (
            angles_future-anles_cutoff)*(phase_rate)*150
        measured_foot_derivative = np.append(measured_foot_derivative, 0)
        df['jointangles_hip_dot_x'] = measured_foot_derivative

        # Calculate the derivative of thigh dot manually
        anles_cutoff = df['jointangles_thigh_x'].values[:-1]
        angles_future = df['jointangles_thigh_x'].values[1:]
        measured_foot_derivative = (
            angles_future-anles_cutoff)*(phase_rate)*150
        measured_foot_derivative = np.append(measured_foot_derivative, 0)
        df['jointangles_thigh_dot_x'] = measured_foot_derivative

        df.to_parquet(subject[1])


# %%
if __name__ == '__main__':
    quick_flatten_dataport()
    add_global_shank_angle()
    #determine_different_strides()
    #determine_zero_data_strides()
    pass

# %%