#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:17:32 2021

@author: jmontp
"""

import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame
import threading
from functools import lru_cache
import os
from scipy.io import loadmat


def get_column_name(column_string_list,num_last_keys):
    filter_strings = ['right','left']
    filtered_list = filter(lambda x: not x in filter_strings, column_string_list)
    column_string = '_'.join(filtered_list)
    return column_string


def get_end_points(d,out_dict,parent_key='', sep='/',num_last_keys=4):

    for k,v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v,h5py._hl.group.Group):
            get_end_points(v,out_dict,new_key,sep=sep)
        #Where the magic happens when you reach an end point
        else:
            column_string_list = (parent_key+sep+k).split(sep)[-num_last_keys:]
            column_string = get_column_name(column_string_list,num_last_keys)
            
            if (column_string not in out_dict):
                out_dict[column_string] = [[new_key],[v]]
            else:
                out_dict[column_string][0].append(new_key)
                out_dict[column_string][1].append(v)
                
                
                
def quick_flatten_dataport(): 
    pass
    #%%
    file_name = '../local-storage/InclineExperiment.mat'
    h5py_file = h5py.File(file_name)['Gaitcycle']
    
    ### Iterate through all the subjects, make a file per subject to keep it RAM bound
    for subject in h5py_file.keys():
       
        #Initialize variables for the subject
        data = h5py_file[subject]
        save_name = '../local-storage/test/dataport_flattened_partial_{}.parquet'.format(subject)
        #Store all the end points
        columns_to_endpoint_list = {}
        get_end_points(data,columns_to_endpoint_list)
        #This dictionary stores dataframes based on the amount of steps that
        #they have        
        steps_to_dataframes_dict = {}
        #Which column will be used to get information about each row
        selected_column = 'jointangles_ankle_x'
        
       
        ### Main loop - process each potential column
        for column_name,endpoint_list in columns_to_endpoint_list.items():
            
            #If the enpoints contain any of this, ignore the endpoint
            if('subjectdetails' in column_name or\
                'cycles' in column_name or\
                'stepsout' in column_name or\
                'description' in column_name or\
                'mean' in column_name or\
                'std' in column_name):
                print(column_name + " " + str(len(endpoint_list[1])) + " (ignored)")
                continue
            
            #Else: this is a valid column
            print(column_name + " " + str(len(endpoint_list[1])))
            
            #Get the data to add it to a dataframe
            data_array = np.concatenate(endpoint_list[1],axis=0).flatten()
            #Calculate how many steps are in the dataframe
            len_arr = data_array.shape[0]
            len_key = len_arr/150.0
            
            #Add the key to the dataframe
            try:
                steps_to_dataframes_dict[len_key][column_name] = data_array
            except KeyError:
                steps_to_dataframes_dict[len_key] = DataFrame()
                steps_to_dataframes_dict[len_key][column_name] = data_array

            
        ### All the dataframes have been created, add information about phase and task 
        
        #Helper functions to get time, ramp to append task information to dataframe
        @lru_cache(maxsize=5)
        def get_time(trial,leg):
            return data[trial]['cycles'][leg]['time']
        @lru_cache(maxsize=5)
        def get_ramp(trial):
            return data[data[trial]['description'][1][1]][0][0]
        @lru_cache(maxsize=5)
        def get_speed(trial):
            return data[data[trial]['description'][1][0]][0][0]
        
        ## Iterate by row to get phase information
        
        #Ugly but effective  
        #The first list comprehension gives you a two-dimensional list with every trial 
        #the second comprehension flattens out the list into a single-dimensional list 
        endpoint_list = columns_to_endpoint_list[selected_column]
        #trials = [x for experiment_name, dataset in zip(*endpoint_list) for x in [experiment_name.split('/')[0]]*dataset.shape[0]*dataset.shape[1]]
        #legs = [x for experiment_name, dataset in zip(*endpoint_list) for x in [experiment_name.split('/')[-3]]*dataset.shape[0]*dataset.shape[1]]
       
        #Create lists to store all the phase dot and step length information
        trials = []
        legs = []
        phase_dot_list = []
        step_length_list = []
        
        #Iterate by trial to get time, speed
        for experiment_name, dataset in zip(*endpoint_list):
                endpoint_split = experiment_name.split('/')    
                
                trial = endpoint_split[0]
                leg = endpoint_split[-3]
                
                trials.extend([trial]*dataset.shape[0]*dataset.shape[1])
                legs.extend([leg]*dataset.shape[0]*dataset.shape[1])
                
                time = get_time(trial,leg)
                speed = get_speed(trial)
                
                time_delta = (time[:,-1]-time[:,0])
                phase_dot_list.append(np.repeat(1/time_delta, 150))
                step_length_list.append(np.repeat(speed*time_delta, 150))
        
        #Get the corresponding dataframe to the selected column
        df = None
        for dataframe in steps_to_dataframes_dict.values():
            if selected_column in dataframe.columns:
                df = dataframe
                
        print(len(trials))
        print(df.shape[0])
       
        df['ramp'] = [get_ramp(trial) for trial in trials]
        df['speed'] = [get_speed(trial) for trial in trials]
        df['trial'] = trials
        #We don't want phase to reach one because 0=1 in terms of phase
        phase = np.linspace(0,(1-1/150),150)
        df['leg'] = legs
        df['phase'] = np.tile(phase,int(df.shape[0]/150))
        df['phase_dot'] = np.concatenate(phase_dot_list, axis = 0)
        df['step_length'] = np.concatenate(step_length_list,axis=0)
        #Comment out to not save
        #df.to_parquet(save_name)
        
        #Uncomment break to just get one person
        break
#%%
def add_global_shank_angle():
    pass
    #%%
    # #Get the subjects
    subjects = [('AB10','../local-storage/test/dataport_flattened_partial_AB10.parquet')]
    for i in range(1,10):
        subjects.append(('AB0'+str(i),'../local-storage/test/dataport_flattened_partial_AB0'+str(i)+'.parquet'))

    for subject in subjects:
        df = pd.read_parquet(subject[1])
        print(df.columns)
        #df.drop(columns=['jointangle_shank_x','jointangle_shank_y','jointangle_shank_z'])
        df['jointangles_shank_x'] = df['jointangles_foot_x'] +  df['jointangles_ankle_x']
        df['jointangles_shank_y'] = df['jointangles_foot_y'] +  df['jointangles_ankle_y']
        df['jointangles_shank_z'] = df['jointangles_foot_z'] +  df['jointangles_ankle_z']
        
        #Calculate the derivative of foot dot manually
        shank_anles_cutoff = df['jointangles_shank_x'].values[:-1]    
        shank_angles_future = df['jointangles_shank_x'].values[1:]
        phase_rate = df['phase_dot'].values[:-1]
        measured_shank_derivative = (shank_angles_future-shank_anles_cutoff)*(phase_rate)*150
        measured_shank_derivative = np.append(measured_shank_derivative,0)
        df['jointangles_shank_dot_x'] = measured_shank_derivative
        
        #Calculate the derivative of foot dot manually
        foot_anles_cutoff = df['jointangles_foot_x'].values[:-1]    
        foot_angles_future = df['jointangles_foot_x'].values[1:]
        measured_foot_derivative = (foot_angles_future-foot_anles_cutoff)*(phase_rate)*150
        measured_foot_derivative = np.append(measured_foot_derivative,0)
        df['jointangles_foot_dot_x'] = measured_foot_derivative
        df.to_parquet(subject[1])


#%%            
if __name__ == '__main__':
    #quick_flatten_dataport()
    #add_global_shank_angle()
    pass