#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:16:47 2020

@author: jose
"""

#Todo: Update visualization to include the standard deviation at each point
## do so by getting all the y values for each point in phase and then getting 
## the predicted y and compraring to actual y



#%% Imports and setup

#My own janky imports
from data_generators import get_trials_in_numpy, get_phase_from_numpy, get_phase_dot, get_step_length, get_ramp, get_subject_names
from model_framework import Fourier_Basis, Polynomial_Basis, Kronecker_Model, Measurement_Model, least_squares, model_prediction, model_saver, model_loader
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


##Get the names of all the subjects
subject_names = get_subject_names()


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
    for subject in subject_names:
   
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
    amount_of_subjects=len(subject_names)

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

    #Change of basis conversions
    def param_to_orthonormal(ξ):
        return Qinv @ ξ
    def param_from_orthonormal(ξ):
        return Q @ ξ
    def matrix_to_orthonormal(Ξ):
        return Ξ @ Qinv


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


if __name__=='__main__':

    if True:
        ##Create the model for the hip
        phase_model = Fourier_Basis(6,'phase')
        phase_dot_model = Polynomial_Basis(1, 'phase_dot')
        ramp_model = Polynomial_Basis(1, 'ramp')
        step_length_model = Polynomial_Basis(1,'step_length')
        
        model_hip = Kronecker_Model(phase_dot_model, ramp_model, step_length_model,phase_model)

        ##Get the pca axis
        pca_axis_hip, cumulative_variance = calculate_pca_axis(model_hip, 'hip')

        #Set the axis
        model_hip.set_pca_axis(pca_axis_hip)


        ##Create the model for the ankle
        phase_model = Fourier_Basis(6,'phase')
        phase_dot_model = Polynomial_Basis(1, 'phase_dot')
        ramp_model = Polynomial_Basis(1, 'ramp')
        step_length_model = Polynomial_Basis(1,'step_length')
        model_ankle = Kronecker_Model(phase_dot_model, ramp_model, step_length_model,phase_model)

        ##Get the pca axis
        pca_axis_ankle, cumulative_variance = calculate_pca_axis(model_ankle, 'ankle')

        #Set the axis
        model_ankle.set_pca_axis(pca_axis_ankle)

        m_model = Measurement_Model(model_hip, model_ankle)

        model_saver(m_model,'H_model.pickle')

    else: 
        m_model = model_loader('H_model.pickle')

    print(m_model.models)
    h_eval = m_model.evalulate_H_func(1,2,3,4,1,2,3,4,5,6)
    dh_eval = m_model.evaluate_dH_func(1,2,3,4,1,2,3,4,5,6)



    print(h_eval)
    print(dh_eval)

    print('done')