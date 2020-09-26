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
import matplotlib.pyplot as plt
import numpy as np

#To get the data
import h5py


#Set a reference to where the dataset is located
dataset_location = 'local-storage/'
#Reference to the raw data filename
filename = 'InclineExperiment.mat'

raw_walking_data = h5py.File(dataset_location+filename, 'r')




#https://stackoverflow.com/questions/4258106/how-to-calculate-a-fourier-series-in-numpy
#This implementation looks better ngl


#My implementation based on Grays least squares estimation
def get_fourier_coeff (x,y,z,n=0):
 #flatten all the data out
 phi=x.reshape(-1)
 #phi=2*np.pi*x.reshape(-1)
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
def get_fourier_sum(a,x, Nh):
   f = np.array([2*a[i]*np.exp(1j*2*i*np.pi*x) for i in range(1,Nh+1)])
   return f.sum()


#Paint the lines with corresponding colors
colors=['blue','green','red','cyan','magenta','yellow','black','white']
color_index=0



#The amount of datapoints we have per step
datapoints=150
    
#Amount of model parameters. Total parameters are 4n+2
num_params=8


#The list of all the fourier coefficients per patient
parameter_list=[]

#Iterate through all the trials for a test subject
for subject in raw_walking_data['Gaitcycle'].keys():
    
    print("Doing subject: " + subject)
    left_hip_total=get_trials_in_numpy(subject)
    phi_total=get_phi_from_numpy(left_hip_total)
    step_length_total=get_step_length(subject)   
    
    print("Got data for: " + subject)
    # This section is dedicated to regression based on fourier series
    #This is mostly from: https://stackoverflow.com/questions/52524919/fourier-series-fit-in-python
    fourier_coeff = get_fourier_coeff(phi_total,step_length_total,left_hip_total,num_params)
    y_pred = get_fourier_sum(fourier_coeff, phi_total[0], num_params)
    
    #Plot the result
    plt.plot(phi_total[0], left_hip_total[0], color=colors[color_index])
    plt.plot(phi_total[0], y_pred, color=colors[color_index], ls=':')
    color_index=(color_index+1)%len(colors)
    plt.show
    
    parameter_list += fourier_coeff
    
    
#%% This section is dedicated to PCA on the coefficients for the fourier 
### transform
#Performing the PCA
from sklearn.decomposition import PCA


#We need the amount of subjects to reshape the parameter matrix
amount_of_subjects=len(raw_walking_data['Gaitcycle'].keys())

#Create the parameter matrix based on the coefficients for all the models
np_parameters = np.array(parameter_list).reshape(amount_of_subjects,4*num_params+3)

#Normalize by substracting the mean and dividing by the variance
#TODO - verify if this is going this row weise
np_parameters -= np.mean(np_parameters)
np_parameters /= np.std(np_parameters)


#Ref:https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
pca=PCA(n_components=3)

pca.fit(np_parameters)

fourier_paramaters_pca = pca.transform(np_parameters)



import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

fig = px.scatter_3d(x=fourier_paramaters_pca[:,0], 
                    y=fourier_paramaters_pca[:,1],
                    z=fourier_paramaters_pca[:,2],
                    labels={'x':'PCA 1',
                            'y':'PCA 2',
                            'z':'PCA 3'})
fig.show()






