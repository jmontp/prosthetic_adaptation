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
from model_framework import Fourier_Basis, Polynomial_Basis, Kronecker_Model, Measurement_Model, least_squares, model_prediction, model_saver, model_loader, calculate_regression_matrix
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


def calculate_personalization_map(model, joint, subject_to_leave_out='', num_gait_fingerprints=6, visualize=False):

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

    #We can also leave one person out of Ξ and output his ξ directly
    leave_out_ξ = None
    leave_out_R = None
    expected_output = None

    ##Calculate a model per subject
    #Iterate through all the trials for a test subject
    for subject in subject_names:
   
        #Get the data for the subject
        #print("Doing subject: " + subject)
        left_hip_angle = get_trials_in_numpy(subject,joint)
        phases = get_phase_from_numpy(left_hip_angle)
        phase_dots = get_phase_dot(subject)
        step_lengths = get_step_length(subject)
        ramps = get_ramp(subject)
        #print("Obtained data for: " + subject)

        #Fit the model for the person
        ξ, R_p = least_squares(model, left_hip_angle.ravel(), phase_dots.ravel(), ramps.ravel(), step_lengths.ravel(),phases.ravel())
        
        #If we want to leave someone out store their parameters here
        if(subject == subject_to_leave_out):
            leave_out_ξ = ξ
            leave_out_R = R_p
            expected_output = left_hip_angle
            #Skip over this person
            continue

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
    if(visualize == True):
        fig.show()

    #This is equation eq:inner_regressor in the paper!
    G = G_total/N_total

    #We need the amount of subjects to reshape the parameter matrix
    amount_of_subjects=len(subject_names)

    #Create the parameter matrix based on the coefficients for all the models
    #If you leave that out, take that into consideration
    if(subject_to_leave_out == ''):
        Ξ = np.array(parameter_list).reshape(amount_of_subjects,model.size)
    else: 
        Ξ = np.array(parameter_list).reshape(amount_of_subjects-1,model.size)

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
    η = num_gait_fingerprints
    pca_axis_array = []

    #Convert from the new basis back to the original basis vectors
    for i in range (0,η):
        pca_axis_array.append(param_from_orthonormal(U[:,i]*np.sqrt(ψs[i])))

    #-----------------------------------------------------------------------------------------------------------------
    #Calculate the cumulative variance
    cumulative_variance = np.cumsum([0]+list((ψs[0:η])/sum(ψs)))

    return pca_axis_array, ξ_avg, cumulative_variance, leave_out_ξ, leave_out_R, expected_output

#I think the outputs should be the model, the optimal parameters and the percentage of variation per parameter


def calculate_gait_fingerprint(model, subject, personalization_map, avg_personalization_vector, joint, num_gait_fingerprints, R_personalization=None):

    ##Setup model config
    #The amount of datapoints we have per step
    datapoints=150

    #Dictionary for Subject to gait fingerprints
    gait_fingerprint = {}

    ##Calculate a model per subject
    #Get the data for the subject
    print("Personalizing subject: " + subject)
    joint_angle = get_trials_in_numpy(subject,joint)
    
    if(R_personalization is None):
        phases = get_phase_from_numpy(joint_angle)
        phase_dots = get_phase_dot(subject)
        step_lengths = get_step_length(subject)
        ramps = get_ramp(subject)
        #print("Obtained data for: " + subject)

        #Fit the model for the person
        R_personalization = calculate_regression_matrix(model, phase_dots.ravel(), ramps.ravel(), step_lengths.ravel(),phases.ravel())
        #print("R_personalization shape:" + str(R_personalization.shape))


    average_estimate = R_personalization @ avg_personalization_vector

    Y = joint_angle.ravel() - average_estimate
    A = R_personalization @ personalization_map

    c = np.linalg.solve(A.T @ A, A.T @ Y)
    
    subject_sum = avg_personalization_vector + personalization_map @ c
    
    return c, subject_sum



if __name__=='__main__':

    #Calculate two models
    if True:
        ##Create the model for the hip
        phase_model = Fourier_Basis(6,'phase')
        phase_dot_model = Polynomial_Basis(2, 'phase_dot')
        ramp_model = Polynomial_Basis(2, 'ramp')
        step_length_model = Polynomial_Basis(2,'step_length')
        
        model_hip = Kronecker_Model(phase_dot_model, ramp_model, step_length_model,phase_model)

        ##Get the pca axis
        pca_axis_hip, _, cumulative_variance,_,_,_ = calculate_personalization_map(model_hip, 'hip', '', )

        # #Set the axis
        # model_hip.set_pca_axis(pca_axis_hip)


        # ##Create the model for the ankle
        # phase_model = Fourier_Basis(6,'phase')
        # phase_dot_model = Polynomial_Basis(1, 'phase_dot')
        # ramp_model = Polynomial_Basis(1, 'ramp')
        # step_length_model = Polynomial_Basis(1,'step_length')
        # model_ankle = Kronecker_Model(phase_dot_model, ramp_model, step_length_model,phase_model)

        # ##Get the pca axis
        # pca_axis_ankle, _, cumulative_variance = calculate_personalization_map(model_ankle, 'ankle')

        # #Set the axis
        # model_ankle.set_pca_axis(pca_axis_ankle)

        # m_model = Measurement_Model(model_hip, model_ankle)

        # model_saver(m_model,'H_model.pickle')

    #Run to get the 'leave one out validation'
    elif False:
        ##Create the model for the hip
        phase_model = Fourier_Basis(6,'phase')
        phase_dot_model = Polynomial_Basis(1, 'phase_dot')
        ramp_model = Polynomial_Basis(1, 'ramp')
        step_length_model = Polynomial_Basis(1,'step_length')
        
        model_hip = Kronecker_Model(phase_dot_model, ramp_model, step_length_model,phase_model)

        for n in range(5,6):
            #Dictionary of subject to gait fingerprint
            gait_fingerprints = {}

            #Dictionary of the expected model parameters
            expected_model_parameters = {}

            #Save the personalization maps as well
            personalization_map_dict = {}

            #Save the regression matrix, why not
            regression_matrix_dict = {}

            #Average Xi map
            average_xi_map = {}

            #expected output
            output_map = {}

            for name in subject_names:
                ##Get the pca axis
                pca_axis_hip, ξ_avg, cumulative_variance, left_out_ξ, regression_matrix, expected_output = calculate_personalization_map(model_hip, 'hip', name, n)
                
                #Transpose in order to use it with least squares
                personalization_map = np.array(pca_axis_hip).T

                expected_model_parameters[name] = left_out_ξ
                personalization_map_dict[name] = personalization_map
                regression_matrix_dict[name] = regression_matrix

                #Get the gait fingerprint
                subject_c, estimated_ξ = calculate_gait_fingerprint(model_hip, name, personalization_map, ξ_avg, 'hip', n, regression_matrix)

                gait_fingerprints[name] = subject_c

                average_xi_map[name] = ξ_avg

                output_map[name] = expected_output

                #print('Subject' + name + ' optimal gait fingerprint: ' + str(left_out_ξ))
                #print('Subject' + name + ' is estimated gait fingerprint: ' + str(estimated_ξ))
                print("L2 Norm Between the two: " + str(np.linalg.norm(left_out_ξ - estimated_ξ, ord=2)))

            save_list = [gait_fingerprints, expected_model_parameters, personalization_map_dict, regression_matrix_dict, output_map, average_xi_map]

            file_name = 'gait_fingerprints_n' + str(n) + '.pickle'

            model_saver(save_list, file_name)

            print("Saved succesfully!: file name: " + file_name)