#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:16:47 2020

@author: jose
"""

#%% Imports and setup

#Getting data from dataset
import h5py

#General Purpose and plotting
import matplotlib.pyplot as plt
import numpy as np

#Gaussian process regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

#Fourier Regression
from symfit import parameters, variables, sin, cos, Fit

def fourier_series (x,f,n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    #Make the paremeter objects for all the terms
    a0, *cos_a=parameters(','.join(['a{}'.format(i) for i in range(0,n+1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series???
    series = a0 + sum(ai*cos(i*f*x)+bi*sin(i*f*x) for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

#Performing the PCA

#%% This section is dedicated to getting the data

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


#Flatten the list so that we have all the data in one axis


#%% This section is dedicated to setting up the data

#Lets set up an X axis that will serve as our phase variable phi
#> Well use a linspace to setup a linear interpolation from 0-1
#> 150 samples will be used since that is the amount of samples taken per step
phi = np.linspace(0,1,150)

#This has the amounts of step data per person
amount_of_steps=left_hip.shape[0]


#This variable will store all the runs for one person in a flattened np array
left_hip_total = np.empty((0,0))
phi_total=np.empty((0,0))


#This will plot every step in term of the phase variable phi
#> H5py datasets iterate over the first axis which makes this very easy
for step in left_hip:
    plt.plot(phi,step)
    np.append(left_hip_total,step)
    np.append(phi_total,phi)
    pass
    
#Set the labels for the axis
plt.ylabel('Left hip angle')
plt.xlabel('Phase variable phi')

#Plot it
#plt.show




# This section is dedicated to regression based on fourier series
#This is mostly from: https://stackoverflow.com/questions/52524919/fourier-series-fit-in-python


 
x, y = variables('x, y')
w, = parameters('w')
model_dict = {y: fourier_series(x, f=w, n=3)}

fit = Fit(model_dict, x=phi, y=left_hip[0])
fit_result = fit.execute()

print(fit_result)

# Plot the result
#plt.plot(phi, left_hip[0])
plt.plot(phi, fit.model(x=phi, **fit_result.params).y, color='green', ls=':')


#%% This section is dedicated to PCA on the coefficients for the fourier 
### transform
















#%% This section is dedicated for regression based on GP

# #Define the kernel, here we are going to use the RBF kernel to start
# kernel = RBF(length_scale=0.01, length_scale_bounds=(1e-3, 1e1))

# #Need to reshape de data in order to fit to gaussian process
# phi_reshape = phi[:, np.newaxis]    #(150,) -> (150,1)
# left_hip_reshape = np.array(left_hip[0])[:, np.newaxis]
# predict_phi = np.linspace(0,1,1000)


# gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
# fit_model = gp.fit(phi_reshape,left_hip[0])

# y_mean, y_cov = fit_model.predict(predict_phi[:, np.newaxis], return_cov=True)
# y_mean.reshape(-1,1)
# y_cov.reshape(-1,1)


# plt.plot(predict_phi, y_mean, 'k', lw=3, zorder=9)
# plt.fill_between(predict_phi, y_mean - np.sqrt(np.diag(y_cov)),
#                  y_mean + np.sqrt(np.diag(y_cov)),
#                  alpha=0.5, color='k')


# plt.plot(predict_phi, y_mean - np.sqrt(np.diag(y_cov)),alpha=0.5, color='k')
# plt.plot(predict_phi, y_mean + np.sqrt(np.diag(y_cov)),alpha=0.5, color='k')

# plt.show()



