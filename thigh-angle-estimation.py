#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:16:47 2020

@author: jose
"""

#%% Imports and setup

#My own janky imports
from generators import get_trials_in_numpy, get_phi_from_numpy, get_step_length



#General Purpose and plotting
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"
import numpy as np

#To get the data
import h5py



#Set a reference to where the dataset is located
dataset_location = 'local-storage/'
#Reference to the raw data filename
filename = 'InclineExperiment.mat'

#Get the walking dataset
raw_walking_data = h5py.File(dataset_location+filename, 'r')

#Create the figure element
fig=go.Figure()


#https://stackoverflow.com/questions/4258106/how-to-calculate-a-fourier-series-in-numpy
#This implementation looks better ngl
# import numpy as np
# def cn(n):
#    c = y*np.exp(-1j*2*n*np.pi*time/period)
#    return c.sum()/c.size

#My implementation based on Grays least squares estimation
def get_fourier_coeff (x,y,z,n=0):
    #flatten all the data out
    #phi=x.reshape(-1)
    phi=2*np.pi*x.reshape(-1)
    step_length=y.reshape(-1)
    left_hip_angle=z.reshape(-1)
 
    R = [np.ones((len(phi),))]
    for i in range(n)[1:]:
        R.append(np.sin(i*phi))
        R.append(np.cos(i*phi))
        R.append(step_length*np.cos(i*phi))
        R.append(step_length*np.sin(i*phi))
    R = np.array(R).T
        
    return np.linalg.solve(R.T @ R, R.T @ left_hip_angle)
 
    
#Plot the fourier series
# def get_fourier_sum(a,x,y,Nh):
#     f = np.array([a[4*i+1]*np.cos(2*(i+1)*np.pi*x)\
#                 + a[4*i+2]*np.sin(2*(i+1)*np.pi*x)\
#                 + a[4*i+3]*y*np.cos(2*(i+1)*np.pi*x)\
#                 + a[4*i+4]*y*np.sin(2*(i+1)*np.pi*x)\
#                       for i in range(0,Nh-1)])
#     return a[0]+f.sum()

def get_fourier_sum(a,x,y,n):
    phi=2*np.pi*x.reshape(-1)
    step_length=y.reshape(-1)
 
    R = [np.ones((len(phi),))]
    for i in range(n)[1:]:
        R.append(np.sin(i*phi))
        R.append(np.cos(i*phi))
        R.append(step_length*np.cos(i*phi))
        R.append(step_length*np.sin(i*phi))
    R = np.array(R).T
    
    return R @ a


def get_fourier_prediction(fourier_coeff, phi_total, step_length_total, num_params):
    return np.array([get_fourier_sum(fourier_coeff, phi, step, num_params)\
              for phi,step in zip(phi_total[0], step_length_total.mean(0))]).reshape(-1)

#Paint the lines with corresponding colors
colors=['black','green','red','cyan','magenta','yellow','black','white',
        'cadetblue', 'darkgoldenrod', 'darkseagreen', 'deeppink', 'midnightblue']
color_index=0



#The amount of datapoints we have per step
datapoints=150
    
#Amount of model parameters. Total parameters are 4n+2
num_params=12


#The list of all the fourier coefficients per patient
parameter_list=[]

#Iterate through all the trials for a test subject
for subject in raw_walking_data['Gaitcycle'].keys():
    
    print("Doing subject: " + subject)
    left_hip_total=get_trials_in_numpy(subject)
    phi_total=get_phi_from_numpy(left_hip_total)
    step_length_total=get_step_length(subject)
    phi=phi_total[0]
    
    print("Got data for: " + subject)
    # This section is dedicated to regression based on fourier series
    #This is mostly from: https://stackoverflow.com/questions/52524919/fourier-series-fit-in-python
    fourier_coeff = get_fourier_coeff(phi_total,step_length_total,left_hip_total,num_params)
    y_pred = get_fourier_prediction(fourier_coeff, phi_total, step_length_total,num_params)
    
    #Plot the result
    fig.add_trace(go.Scatter(x=phi, y=left_hip_total.mean(0),
                             line=dict(color=colors[color_index], width=4),
                             name=subject +' data'))
    fig.add_trace(go.Scatter(x=phi, y=y_pred,
                             line=dict(color=colors[color_index], width=4, dash='dash'),
                             name=subject + ' predicted'))
    color_index=(color_index+1)%len(colors)
    parameter_list.append(fourier_coeff)

fig.show()
#%% This section is dedicated to PCA on the coefficients for the fourier 
### transform
#Performing the PCA
from sklearn.decomposition import PCA


#We need the amount of subjects to reshape the parameter matrix
amount_of_subjects=len(raw_walking_data['Gaitcycle'].keys())

#Create the parameter matrix based on the coefficients for all the models
np_parameters = np.array(parameter_list).reshape(amount_of_subjects,4*num_params-3)

#Normalize by substracting the mean and dividing by the variance
#TODO - verify if this is going this row weise
np_parameters -= np.mean(np_parameters)
np_parameters /= np.std(np_parameters)


#Ref:https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
pca=PCA(n_components=6)

pca.fit(np_parameters)

fourier_paramaters_pca = pca.transform(np_parameters)



# fig2 = px.scatter_3d(x=fourier_paramaters_pca[:,0], 
#                     y=fourier_paramaters_pca[:,1],
#                     z=fourier_paramaters_pca[:,2],
#                     labels={'x':'PCA 1',
#                             'y':'PCA 2',
#                             'z':'PCA 3'})
# fig2.show()

parameter_variation_axis=np.linspace(-50,50,100)
num=1
for axis in pca.components_:
    print('Doing Principal component: ' + str(num))
    num = num+1
    z_pred_pca=np.array([get_fourier_prediction(parameter_list[0]+x*axis,
                                                phi_total, step_length_total,num_params)
                         for x in parameter_variation_axis])
    
    
    fig3=go.Figure(data=[go.Surface(x=phi,y=parameter_variation_axis,z=z_pred_pca)])
    
    fig3.show()





