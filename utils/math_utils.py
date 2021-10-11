#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:17:39 2021

@author: jmontp
"""
#Common imports
import numpy as np
import pandas as pd


test_pd = True
def assert_pd(matrix,name):
    if test_pd:
        try: 
            assert (matrix.shape[0] == matrix.shape[1])
            assert (len(matrix.shape)==2)
        except AssertionError:
            print(name + " NOT EVEN SQUARE: " + str(matrix.shape))
            print("Assertion on matrix: \n{}".format(matrix))
    
            raise AssertionError
    
        try:
            assert (np.linalg.norm(matrix-matrix.T) < 1e-7*np.linalg.norm(matrix))
        except AssertionError:
            print(name + " Error with norm: " + str(np.linalg.norm(matrix-matrix.T)))
            print("Assertion on matrix: \n{}".format(matrix))
    
            raise AssertionError
            
        try:
            for e in np.linalg.eigh(matrix)[0]:
                assert (e + 1e-8 > 0)
        except AssertionError:
            print(name + " Error with Evalue: " + str([e for e in np.linalg.eigh(matrix)[0]]))
            print("Assertion on matrix: \n{}".format(matrix))
            raise AssertionError
        else:
            return None 
    else:
        #print("Assertions turned off")  
        pass 



def get_mean_std_dev(np_array):
        point_per_stride = 150

        if (type(np_array) == np.ndarray):
            strides = int(np_array.shape[0]/point_per_stride)
            mean = np.mean(np_array.reshape(strides,-1),axis=0)
            std_dev = np.std(np_array.reshape(strides,-1),axis=0)
        else:
            strides = int(np_array.shape[0]/point_per_stride)
            mean = np.mean(np_array.values.reshape(strides,-1),axis=0)
            std_dev = np.std(np_array.values.reshape(strides,-1),axis=0)
        
        return mean, std_dev
    
    
def get_rmse(estimated_angles,measured_angles):
    
    # #If the shape is not the same, tile the estimation
    # if (estimated_angles.shape != measured_angles.shape):
    #    num_repeats = int(measured_angles.shape[0]/estimated_angles.shape[0])
    #    estimated_angles = np.tile(estimated_angles,num_repeats).reshape(-1)
    
    if (type(estimated_angles) == pd.core.series.Series):
        estimated_angles = estimated_angles.values[:,np.newaxis]
    
    if (type(measured_angles) == pd.core.series.Series):
        measured_angles = measured_angles.values[:,np.newaxis]
        
    return np.sqrt(np.mean(np.power((estimated_angles-measured_angles),2)))

def trial_to_string(trial,joint=None):
    sign = 1
    split = trial.split('i')
    if(len(split)==1):
        sign = -1
        split = trial.split('d')

    incline = sign*float(split[1].replace('x','.'))
    speed = float(split[0][1:].replace('x','.'))
    
    if(joint is not None):
        output = "{} Angle while walking at {:.1f}m/s with {:.1f}-deg incline".format(joint,speed,incline)
    else:
        output = "Walking at {:.1f}m/s with {:.1f}-deg incline".format(speed,incline)
    return output