#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:16:47 2020

@author: jose
"""
#%% This section is dedicated to getting the data
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

#Set a reference to where the dataset is located
dataset_location = 'local-storage/'
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


#%% This section is dedicated to plotting the data

#Lets set up an X axis that will serve as our phase variable phi
#> Well use a linspace to setup a linear interpolation from 0-1
#> 150 samples will be used since that is the amount of samples taken per step
phi = np.linspace(0,1,150)


#This will plot every step in term of the phase variable phi
#> H5py datasets iterate over the first axis which makes this very easy
for step in left_hip:
    plt.plot(phi,step)
    
#Set the labels for the axis
plt.ylabel('Left hip angle')
plt.xlabel('Phase variable phi')

#Plot it
plt.show



#%% This section is dedicated for regression based on GP

#Define the kernel, here we are going to use the RBF kernel to start
kernel = RBF(length_scale=0.01, length_scale_bounds=(1e-3, 1e1))

#Need to reshape de data in order to fit to gaussian process
phi_reshape = phi[:, np.newaxis]
left_hip_reshape = np.array(left_hip[0])[:, np.newaxis]
predict_phi = np.linspace(0,1,1000)


gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
fit_model = gp.fit(phi_reshape,left_hip[0])

y_mean, y_cov = fit_model.predict(predict_phi[:, np.newaxis], return_cov=True)
y_mean.reshape(-1,1)
y_cov.reshape(-1,1)


plt.plot(predict_phi, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(predict_phi, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.5, color='k')

plt.show()
