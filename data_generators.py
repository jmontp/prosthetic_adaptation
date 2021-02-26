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
dataset_location = './local-storage/'
#Reference to the raw data filename
filename = 'InclineExperiment.mat'

#Import the raw data from the walking database 
#> README for the dataset can be found in 
#> https://ieee-dataport.org/open-access/
#> effect-walking-incline-and-speed-human-leg-kinematics-kinetics-and-emg
raw_walking_data = h5py.File(dataset_location+filename, 'r')


#Example of the structure
#left_hip = raw_walking_data['Gaitcycle']['AB01']['s0x8i10']\
#                            ['kinematics']['jointangles']['left']['hip']['x']


#Set a dictionary to store all the fitting parameters per the different runs
parameters_per_walk_config=[]
parameter_list=[]


#This will return a list of the subjects names
def get_subject_names():
    return raw_walking_data['Gaitcycle'].keys()



#This will return a concatenated array with all the trials for a subject
def get_trials_in_numpy(subject,joint):
    #This retruns the numpy array that has all the trials.
    #This essentially hides the ugly generator call
    return np.concatenate(list(joint_angle_generator(subject,joint)),axis=0)

#This will return a corresponding axis for the data in the model
def get_phase_from_numpy(array):
    #This retruns the numpy array that has all the axis.
    #This essentially hides the ugly generator call
    return np.concatenate(list(phi_generator(int(array.shape[0]/150),150)),axis=0)

#This will return the step lengths for every trial
def get_step_length(subject):
    #This retruns the numpy array that has all the step lengths
    #This essentially hides the ugly generator call
    return np.concatenate(list(step_length_generator(subject)),axis=0)


def get_ramp(subject):
    #This retruns the numpy array that has all the step lengths
    #This essentially hides the ugly generator call
    return np.concatenate(list(ramp_generator(subject)),axis=0)

def get_phase_dot(subject):
    #This retruns the numpy array that has all the step lengths
    #This essentially hides the ugly generator call
    return np.concatenate(list(phase_dot_generator(subject)),axis=0)



#This will generate the angles 
def joint_angle_generator(subject,joint,left=True,right=True):
    #Get all the trials
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        #Dont care about the subject details
        if trial == 'subjectdetails':
            continue
        #Return the numpy array for the trial
        #Verify whether we want left or right
        #By default it gets both
        if(left == True):
            yield raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['left'][joint]['x'][:]
     

    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        #Dont care about the subject details
        if trial == 'subjectdetails':
            continue
        #Return the numpy array for the trial
        if(right == True):
            yield raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['right'][joint]['x'][:]


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
        
        #Get the h5py object pointer for the walking speed
        ptr = raw_walking_data['Gaitcycle'][subject][trial]['description'][1][0]

        walking_speed = raw_walking_data[ptr]
        
        time_info_left=raw_walking_data['Gaitcycle'][subject][trial]['cycles']['left']['time']
        
        time_info_right=raw_walking_data['Gaitcycle'][subject][trial]['cycles']['right']['time']

        for step_left in time_info_left:
            #Calculate the step length for the given walking config
            #Get delta time of step
            delta_time_left=step_left[149]-step_left[0]
            #Set the steplength for the 
            yield np.full((1,150),walking_speed*delta_time_left)

        for step_right in time_info_right:

            #Get delta time of step
            delta_time_right=step_right[149]-step_right[0]
            #Set the steplength for the 
            yield np.full((1,150),walking_speed*delta_time_right)
        


def ramp_generator(subject):       
    #Generate for the left leg
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        
        #Get the h5py object pointer for the walking speed
        ptr = raw_walking_data['Gaitcycle'][subject][trial]['description'][1][1]
        ramp = raw_walking_data[ptr]

        #This just gets the amount of ramps you need per step
        time_info=raw_walking_data['Gaitcycle'][subject][trial]['cycles']['left']['time']
       
        for step in time_info:        
            #Yield for the left leg
            yield np.full((1,150),ramp)
            #Yield for the right leg

    #Generate for the right leg
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        
        #Get the h5py object pointer for the walking speed
        ptr = raw_walking_data['Gaitcycle'][subject][trial]['description'][1][1]
        ramp = raw_walking_data[ptr]

        #This just gets the amount of ramps you need per step
        time_info=raw_walking_data['Gaitcycle'][subject][trial]['cycles']['right']['time']
       
        for step in time_info:        
            #Yield for the left leg
            yield np.full((1,150),ramp)
            #Yield for the right leg


def phase_dot_generator(subject):
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        
        #Get the h5py object pointer for the walking speed
        time_info_left= raw_walking_data['Gaitcycle'][subject][trial]['cycles']['left']['time']
        time_info_right = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['right']['time']
        phase_step = 1/150

        for step_left in time_info_left:        
            #Calculate the step length for the given walking config
            #Get delta time of step
            delta_time_left=step_left[1]-step_left[0]
            #Set the steplength for the 
            yield np.full((1,150),phase_step/delta_time_left)

        for step_right in time_info_right:        

            #Get delta time of step
            delta_time_right=step_right[1]-step_right[0]
            #Set the steplength for the 
            yield np.full((1,150),phase_step/delta_time_right)


        
#if __name__ == '__main__':
#    test1=get_step_length('AB01')
#    test2=get_step_length('AB02')
#    test3=get_step_length('AB01')
#    test3=get_step_length('AB02')
    