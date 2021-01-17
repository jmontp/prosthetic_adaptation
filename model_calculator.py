#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:16:47 2020

@author: jose
"""

#%% Imports and setup

#My own janky imports
from data_generators import get_trials_in_numpy, get_phase_from_numpy, get_phase_dot, get_step_length, get_ramp
from model_framework import fourier_basis, polynomial_basis, kronecker_generator, least_squares, model_prediction, model_saver
#General Purpose and plotting
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"
import numpy as np
#To get the data
import h5py
#To get command line arguments
import sys


##Import data
#Set a reference to where the dataset is located
dataset_location = '../'
#Reference to the raw data filename
filename = 'InclineExperiment.mat'
#Get the walking dataset
raw_walking_data = h5py.File(dataset_location+filename, 'r')



#Change of basis conversions
def param_to_orthonormal(ξ):
    return Qinv @ ξ

def param_from_orthonormal(ξ):
    return Q @ ξ

def matrix_to_orthonormal(Ξ):
    return Ξ @ Qinv

def calculate_pca_axis(model, joint,save_file=None, vizualize=False):

    ##Setup model config
    #The amount of datapoints we have per step
    datapoints=150

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


    ##Calculate a model per subject
    #Iterate through all the trials for a test subject
    for subject in raw_walking_data['Gaitcycle'].keys():
   
        #Get the data for the subject
        print("Doing subject: " + subject)
        left_hip_angle = get_trials_in_numpy(subject,joint)
        phases = get_phase_from_numpy(left_hip_angle)
        phase_dots = get_phase_dot(subject)
        step_lengths = get_step_length(subject)
        ramps = get_ramp(subject)
        print("Obtained data for: " + subject)

        #Fit the model for the person
        ξ, R_p = least_squares(model, left_hip_angle.ravel(), phase_dots.ravel(), ramps.ravel(), step_lengths.ravel(),phases.ravel())
        G_total += R_p.T @ R_p
        N_total += R_p.shape[0]

        ##Vizualization
        #Predict the average line 
        draw_left_hip_angle = left_hip_angle.mean(0)
        draw_phases = phases[0]
        draw_phase_dots = phase_dots.mean(0)
        draw_ramps = ramps.mean(0)
        draw_steps = step_lengths.mean(0)
        #Get prediction
        y_pred = model_prediction(model, ξ, draw_phase_dots.ravel(), draw_ramps.ravel(), draw_steps.ravel(),draw_phases.ravel())

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


    #Plot everything
    fig.show()

    #This is equation eq:inner_regressor in the paper!
    G = G_total/N_total

    #We need the amount of subjects to reshape the parameter matrix
    amount_of_subjects=len(raw_walking_data['Gaitcycle'].keys())

    #Create the parameter matrix based on the coefficients for all the models
    Ξ = np.array(parameter_list).reshape(amount_of_subjects,model.size)

    #-----------------------------------------------------------------------------------------------------------------
    
    ##TODO: In theory this can be another function that calculates pca axis and explained variance based on the 
    ## G and \Xi
    #Verify that the G matrix is at least positive semidefinite
    #To be psd or pd, G=G^T
    assert(np.linalg.norm(G-G.T)<1e-7)

    #Diagonalize the matrix G as G = OVO
    eig, O = np.linalg.eigh(G)
    V = np.diagflat(eig)

    #Additionally, all the eigenvalues are true
    for e in eig:
        assert (e>=0)
        assert( e>0) # pd
    
    # Verify that it diagonalized correctly G = O (eig) O.T
    assert(np.linalg.norm(G - O @ V @ O.T)< 1e-7 * np.linalg.norm(G)) # passes
    # print(np.linalg.norm(G - O @ V @ O.T)) # good
    # print(np.linalg.norm(G - O.T @ V @ O)) # false
    # print(np.linalg.norm(G - sum([O[:,[i]]@ O[:,[i]].T * eig[i] for i in range(len(eig))]))) # good
    # print(np.linalg.norm(G)) # false

    #This is based on the equation in eq:Qdef
    # Q G Q = I
    Q       = sum([O[:,[i]] @ O[:,[i]].T * 1/np.sqrt(eig[i]) for i in range(len(eig))])
    Qinv    = sum([O[:,[i]] @ O[:,[i]].T * np.sqrt(eig[i]) for i in range(len(eig))])

    #Get the average coefficients
    ξ_avg = np.mean(Ξ, axis=0)

    #Substract the average coefficients
    Ξ0 = Ξ - ξ_avg

    ##Todo: The pca axis can also be obtained with pca instead of eigenvalue 
    ## decomposition
    #Calculate the coefficients in the orthonormal space
    Ξ0prime = matrix_to_orthonormal(Ξ0)

    #Get the covariance matrix for this
    Σ = Ξ0prime.T @ Ξ0prime / (Ξ0prime.shape[0]-1)

    #Calculate the eigendecomposition of the covariance matrix
    ψinverted, Uinverted = np.linalg.eigh(Σ)

    #Eigenvalues are obtained from smalles to bigger, make it bigger to smaller
    ψs = np.flip(ψinverted)
    Ψ = np.diagflat(ψs)
    
    #If we change the eigenvalues we also need to change the eigenvectors
    U = np.flip(Uinverted, axis=1)

    #Run tests to make sure that this is working
    assert(np.linalg.norm(Σ - U @ Ψ @ U.T)< 1e-7 * np.linalg.norm(Σ)) # passes
    for i in range(len(ψs)-1):
        assert(ψs[i] > ψs[i+1])

    #Define the amount principles axis that we want
    η = 6
    pca_axis_array = []

    #Convert from the new basis back to the original basis vectors
    for i in range (0,η):
        pca_axis_array.append(param_from_orthonormal(U[:,i]*np.sqrt(ψs[i])))

    #-----------------------------------------------------------------------------------------------------------------
    #Calculate the cumulative variance
    cumulative_variance = np.cumsum([0]+list((ψs[0:η])/sum(ψs)))

    return pca_axis_array, cumulative_variance

#I think the outputs should be the model, the optimal parameters and the percentage of variation per parameter


def H_generator(*models):
    
    def H_func(ξs,*inputs):
        result = []
        for model,ξ in zip(models,ξs):
            result.append(model(*inputs)@ξ)
        return np.array(result)

    return 


def dH_generator(*models):

    def dH_func(ξs,*inputs):
        result = []

        for model,ξ in zip(models,ξs):
            #Calculate a jacobean row in 
            jacobean_row = [model_partial_derivative(model,func.variable_name)(*inputs)@ξ for func in model.funcs]
            result.append(jacobean_row)

        return np.array(result)

    return dH


if __name__=='__main__':
    ##Create the model for the hip
    phase_model = fourier_basis(1,'phase')
    phase_dot_model = polynomial_basis(1, 'phase_dot')
    ramp_model = polynomial_basis(1, 'ramp')
    step_length_model = polynomial_basis(1,'step_length')
    model_hip = kronecker_generator(phase_dot_model, ramp_model, step_length_model,phase_model)

    ##Get the pca axis
    pca_axis_hip, cumulative_variance = calculate_pca_axis(model_hip, 'hip')


    ##Create the model for the ankle
    phase_model = fourier_basis(1,'phase')
    phase_dot_model = polynomial_basis(1, 'phase_dot')
    ramp_model = polynomial_basis(1, 'ramp')
    step_length_model = polynomial_basis(1,'step_length')
    model_ankle = kronecker_generator(phase_dot_model, ramp_model, step_length_model,phase_model)

    ##Get the pca axis
    pca_axis_ankle, cumulative_variance = calculate_pca_axis(model_ankle, 'ankle')

    H_func = H_generator((pca_axis_hip[0],pca_axis_ankle[0]),model_hip,model_ankle)

    dH_func = dH_generator((pca_axis_hip[0],pca_axis_ankle[0]),model_hip,model_ankle


    print(pca_axis)
    print(cumulative_variance)
    print('done')