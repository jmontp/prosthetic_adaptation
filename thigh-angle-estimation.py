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

#Ref:https://stackoverflow.com/questions/52524919/fourier-series-fit-in-python
def fourier_series (x,y,f,n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    #Make the paremeter objects for all the terms
    a0, *cos_a=parameters(','.join(['a{}'.format(i) for i in range(0,n+1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    c0, *cos_c=parameters(','.join(['c{}'.format(i) for i in range(0,n+1)]))
    sin_d = parameters(','.join(['d{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series???
    series = a0 + sum(ai*cos(i*f*x)+bi*sin(i*f*x) for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    series += c0*y + sum(y*ci*cos(i*f*x)+y*di*sin(i*f*x) for i, (ci, di) in enumerate(zip(cos_c, sin_d), start=1))
    return series

#Performing the PCA
from sklearn.decomposition import PCA


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


#Set a dictionary to store all the fitting parameters per the different runs
parameters_per_walk_config=[]
parameter_list=[]
header= ['a0','a1','a2','a3','b1','b2','b3','w']


for trial in raw_walking_data['Gaitcycle']['AB01'].keys():
    
    if trial == 'subjectdetails':
        continue          
    
    left_hip = raw_walking_data['Gaitcycle']['AB01'][trial]\
                        ['kinematics']['jointangles']['left']['hip']['x']
    
    
    time_info = raw_walking_data['Gaitcycle']['AB01'][trial]\
                         ['cycles']['left']['time']
                         
                         
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
                        
    #%% This section is dedicated to setting up the data
    
    
    #The amount of datapoints we have per step
    datapoints=150
    
    #Amount of model parameters. Total parameters are 4n+2
    num_params=3
    
    #Lets set up an X axis that will serve as our phase variable phi
    phi = np.linspace(0,1,datapoints)
    
    #This has the amounts of step data per person
    amount_of_steps=left_hip.shape[0]
    
    #This variable will store all the runs for one person in a flattened np array
    left_hip_total=np.array(left_hip)
    phi_total=np.repeat(phi.reshape(1,150),amount_of_steps,axis=0)
    step_length_total=np.empty((amount_of_steps,datapoints))
       
    #Build the dataset for the regression model based on all the steps
    #for a given walking configuration
    for step in range(left_hip.shape[0]):

        #Calculate the step length for the given walking config
        #Get delta time of step
        delta_time=time_info[step][149]-time_info[step][0]
        #Set the steplength for the 
        step_length_total[step]= np.full((150,),walking_speed*delta_time)
    
    
    # This section is dedicated to regression based on fourier series
    #This is mostly from: https://stackoverflow.com/questions/52524919/fourier-series-fit-in-python
    
    x, y, z = variables('x, y, z')
    w, = parameters('w')
    model_dict = {z: fourier_series(x, y, f=w, n=num_params)}
    
    fit = Fit(model_dict, x=phi_total, y=step_length_total, z=left_hip_total)
    fit_result = fit.execute()
        
    print(fit_result)
    
    #Plot the result
    plt.plot(phi, left_hip[0])
    plt.plot(phi, fit.model(x=phi, y=step_length_total[0],
                            **fit_result.params).z, color='green', ls=':')
    plt.show
    
    parameters_per_walk_config += [fit_result.params.copy()]
    parameter_list += fit_result.params.copy().values()
    
    
    #parameters_per_walk_config[-1]['walking_config'] = walking_configuration
    
#%% This section is dedicated to PCA on the coefficients for the fourier 
### transform

# amount_of_walking_configurations=len(raw_walking_data['Gaitcycle']['AB01'])-1


# np_parameters = np.array(parameter_list)\
#     .reshape(amount_of_walking_configurations,4*num_params+3)
# np_parameters-= np.mean(np_parameters)
# np_parameters/= np.std(np_parameters)


# #Ref:https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# pca=PCA(n_components=3)

# pca.fit(np_parameters)

# fourier_paramaters_pca = pca.transform(np_parameters)



# import plotly.express as px
# import plotly.io as pio
# pio.renderers.default = "browser"

# fig = px.scatter_3d(x=fourier_paramaters_pca[:,0], 
#                     y=fourier_paramaters_pca[:,1],
#                     z=fourier_paramaters_pca[:,2],
#                     labels={'x':'PCA 1',
#                            'y':'PCA 2',
#                            'z':'PCA 3'})
# fig.show()






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



