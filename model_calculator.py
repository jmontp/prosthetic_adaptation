#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:16:47 2020

@author: jose
"""

#%% Imports and setup

#My own janky imports
from data_generators import get_trials_in_numpy, get_phase_from_numpy, get_phase_dot, get_step_length, get_ramp
from model_generator import fourier_series, polynomial_function, kronecker_generator, least_squares, model_prediction
#General Purpose and plotting
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"
import numpy as np
#To get the data
import h5py


##Import data
#Set a reference to where the dataset is located
dataset_location = '../'
#Reference to the raw data filename
filename = 'InclineExperiment.mat'
#Get the walking dataset
raw_walking_data = h5py.File(dataset_location+filename, 'r')


##Setup model config
#The amount of datapoints we have per step
datapoints=150
#Amount of model parameters. Total parameters are 4n+2
num_params=12
#The list of all the fourier coefficients per patient
parameter_list=[]
#Amount of datapoints
G_total = 0
N_total = 0


##Visualization
#Create the figure element
fig=go.Figure()
#Paint the lines with corresponding colors
colors=['black','green','red','cyan','magenta','yellow','black','white',
        'cadetblue', 'darkgoldenrod', 'darkseagreen', 'deeppink', 'midnightblue']
color_index=0

##Create the model
phase_model = (fourier_series, 8)
phase_dot_model = (fourier_series, 2)
ramp_model = (polynomial_function, 2)
step_length_model = (polynomial_function, 3)
model,_ = kronecker_generator(phase_model, phase_dot_model, ramp_model, step_length_model)


##Calculate a model per subject
#Iterate through all the trials for a test subject
for subject in raw_walking_data['Gaitcycle'].keys():
    
    if(subject != "AB01"):
        continue
  
    #Get the data for the subject
    print("Doing subject: " + subject)
    left_hip_angle = get_trials_in_numpy(subject)
    phases = get_phase_from_numpy(left_hip_angle)
    phase_dots = get_phase_dot(subject)
    step_lengths = get_step_length(subject)
    ramps = get_ramp(subject)
    print("Obtained data for: " + subject)

  
    #Fit the model for the person
    ξ, R_p = least_squares(model, left_hip_angle.ravel(), phases.ravel(), phase_dots.ravel(), ramps.ravel(), step_lengths.ravel())
    G_total += R_p.T @ R_p
    N_total += R_p.shape[0]

  
    #Predict the average line 
    draw_left_hip_angle = left_hip_angle.mean(0)
    draw_phases = phases[0]
    draw_phase_dots = phase_dots.mean(0)
    draw_ramps = ramps.mean(0)
    draw_steps = step_lengths.mean(0)
    #Get prediction
    y_pred = model_prediction(model, ξ, draw_phases.ravel(), draw_phase_dots.ravel(), draw_ramps.ravel(), draw_steps.ravel())
  

    #Plot the result
    fig.add_trace(go.Scatter(x=draw_phases, y=draw_left_hip_angle,
                             line=dict(color=colors[color_index], width=4),
                             name=subject +' data'))
    fig.add_trace(go.Scatter(x=draw_phases, y=y_pred,
                             line=dict(color=colors[color_index], width=4, dash='dash'),
                             name=subject + ' predicted'))
    color_index=(color_index+1)%len(colors)

    #Add the the matrix of parameters (in the shape of a python list for now)
    parameter_list.append(ξ)

#This is equation eq:inner_regressor in the paper!
G = G_total/N_total

#Plot everything
fig.show()

#We need the amount of subjects to reshape the parameter matrix
amount_of_subjects=len(raw_walking_data['Gaitcycle'].keys())

#Create the parameter matrix based on the coefficients for all the models
np_parameters = np.array(parameter_list).reshape(amount_of_subjects,4*num_params-3)

#Save the information offline so that we do not have to recalculate every time
np.save('UM - fourier coefficient matrix', np_parameters)
np.save('UM - lambda Gram matrix', G)