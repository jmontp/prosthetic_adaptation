#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 23:09:56 2020

@author: jose
"""

#%% Please dont read the code below
### This legit breaks most of the code standards I believe in






#Getting data from dataset
import h5py


import numpy as np





#%% This section is dedicated to getting the data

#Set a reference to where the dataset is located
dataset_location = '../'
#Reference to the raw data filename
filename = 'InclineExperiment.mat'

#Import the raw data from the walking database 
#> README for the dataset can be found in 
#> https://ieee-dataport.org/open-access/
#> effect-walking-incline-and-speed-human-leg-kinematics-kinetics-and-emg
raw_walking_data = h5py.File(dataset_location+filename, 'r')


#Getting the subject info to play around with h5py dataset
#This will provide the each trial of walking, Subject 1 has 39 steps
left_hip = raw_walking_data['Gaitcycle']['AB01']['s0x8i10']\
                            ['kinematics']['jointangles']['left']['hip']['x']


#Set a dictionary to store all the fitting parameters per the different runs
parameters_per_walk_config=[]
parameter_list=[]






### Fine then, you have been warned




#This will return a concatenated array with all the trials for a subject
def get_trials_in_numpy(subject):
    #This retruns the numpy array that has all the trials.
    #This essentially hides the ugly generator call
    return np.concatenate(list(hip_angle_generator(subject)),axis=0)

#This will return a corresponding axis for the data in the model
def get_phi_from_numpy(array):
    #This retruns the numpy array that has all the axis.
    #This essentially hides the ugly generator call
    return np.concatenate(list(phi_generator(array.shape[0],150)),axis=0)

#This will return the step lengths for every trial
def get_step_length(subject):
    #This retruns the numpy array that has all the step lengths
    #This essentially hides the ugly generator call
    return np.concatenate(list(step_length_generator(subject)),axis=0)


#This will generate the angles 
def hip_angle_generator(subject):
    #Get all the trials
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        #Dont care about the subject details
        if trial == 'subjectdetails':
            continue
        #Return the numpy array for the trial
        yield raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['left']['hip']['x'][:]

#This will generate axis
def phi_generator(n,length):
    #We really just care about getting n copies
    for i in range(n):
        yield np.linspace(0,1,length).reshape(1,length)

#This will generate a step length for every set
def step_length_generator(subject):
    
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        
        #Reading the walking speed from the h5py file is cryptic
        #so I am hacking a method by reading the name
        if('s0x8' in trial):
            walking_speed=0.8
        elif('s1x2' in trial):
            walking_speed=1.2
        elif('s1' in trial):
            walking_speed=1
        else:
            raise ValueError("You dont have the speed on the name")  
        
        time_info=raw_walking_data['Gaitcycle'][subject][trial]['cycles']['left']['time']
       
        for step in time_info:        
            #Calculate the step length for the given walking config
            #Get delta time of step
            delta_time=step[149]-step[0]
            #Set the steplength for the 
            yield np.full((1,150),walking_speed*delta_time)
        
        
        
if __name__ == '__main__':
    test1=get_step_length('AB01')
    test2=get_step_length('AB02')
    test3=get_step_length('AB01')
    test3=get_step_length('AB02')
    