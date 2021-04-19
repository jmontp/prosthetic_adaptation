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
    
    #%%
    file_name = '../local-storage/InclineExperiment.mat'
    h5py_file = h5py.File(file_name)['Gaitcycle']
    for subject in h5py_file.keys():
       
        data = h5py_file[subject]
        save_name = '../local-storage/test/dataport_flattened_partial_{}.parquet'.format(subject)
        
        @lru_cache(maxsize=5)
        def get_speed(trial):
            return data[data[trial]['description'][1][1]][0][0]
        
        # out = flatten_list(data)
        # print(len(out))
        
        output_dict = {}
        get_end_points(data,output_dict)
        #print(output_dict)
        output_dataframe = DataFrame()
        
        data_frame_dict = {}
        
        first_column = ""
        for column_name,endpoint_list in output_dict.items():
            
            #Pick out the first column
            if(first_column == ""):
                first_column = column_name
            
            if('subjectdetails' in column_name or\
                'cycles' in column_name or\
                'stepsout' in column_name or\
                'description' in column_name or\
                'mean' in column_name or\
                'std' in column_name):
                print(column_name + " " + str(len(endpoint_list[1])) + " (ignored)")
            else:
                print(column_name + " " + str(len(endpoint_list[1])))
                
                #Pick an arbitraty columns to get information about all the rows
                if(column_name == 'jointangles_ankle_x'):
                    #Yikes...
                    #The first list comprehension gives you a double list 
                    #with every trial, the second comprehension flattens out the list
                    trials = [x for experiment_name, dataset in zip(*endpoint_list) for x in [experiment_name.split('/')[0]]*dataset.shape[0]*dataset.shape[1] ]
                    legs = [x for experiment_name, dataset in zip(*endpoint_list) for x in [experiment_name.split('/')[-3]]*dataset.shape[0]*dataset.shape[1] ]
                    #Yikes and then some
                    @lru_cache(maxsize=5)
                    def get_time(trial,leg):
                        return data[trial]['cycles'][leg]['time']
                    
                    phase_dot_list = []
                    step_length_list = []
                    for experiment_name in endpoint_list[0]:
                            endpoint_split = experiment_name.split('/')    
                            trial = endpoint_split[0]
                            leg = endpoint_split[-3]
                            
                            time = data[trial]['cycles'][leg]['time']
                            speed = get_speed(trial)
                            phase_dot_list.append(np.repeat(1/(time[:,-1]-time[:,0]), 150))
                            step_length_list.append(np.repeat(speed*(time[:,-1]-time[:,0]), 150))
                            
                arr = np.concatenate(endpoint_list[1],axis=0).flatten()
                len_arr = arr.shape[0]
                
                len_key = len_arr/150.0
                if(len_key not in data_frame_dict):
                    data_frame_dict[len_key] = DataFrame()
                    
                data_frame_dict[len_key][column_name] = arr
        
        for df in data_frame_dict.values():
            if "jointangles_ankle_x" in df.columns:
                
                print(len(trials))
                print(df.shape[0])
                #They way that this is formatted, there is a data pointer and 
                # the data itself. The first access to "data" gets the pointer
                # and the second gets the incline
                @lru_cache(maxsize=5)
                def get_ramp(trial):
                    return data[data[trial]['description'][0][1]][0][0]
                
                df['ramp'] = [ get_ramp(trial) for trial in trials]
                
                @lru_cache(maxsize=5)
                def get_speed(trial):
                    return data[data[trial]['description'][1][1]][0][0]
    
                df['speed'] = [get_speed(trial) for trial in trials]
                df['trial'] = trials
                phase = np.linspace(0,(1-1/150),150)
                df['leg'] = legs
                df['phase'] = np.tile(phase,int(df.shape[0]/150))
                df['phase_dot'] = np.concatenate(phase_dot_list, axis = 0)
                df['step_length'] = np.concatenate(step_length_list,axis=0)
                df.to_parquet(save_name)

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
        df.drop(columns=['jointangle_shank_x','jointangle_shank_y','jointangle_shank_z'])
        df['jointangles_shank_x'] = df['jointangles_foot_x'] +  df['jointangles_ankle_x']
        df['jointangles_shank_y'] = df['jointangles_foot_y'] +  df['jointangles_ankle_y']
        df['jointangles_shank_z'] = df['jointangles_foot_z'] +  df['jointangles_ankle_z']
        df.to_parquet(subject[1])


#%%            
if __name__ == '__main__':
    add_global_shank_angle()