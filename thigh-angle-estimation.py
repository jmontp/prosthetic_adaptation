#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:16:47 2020

@author: jose
"""

#%% Imports and setup

#My own janky imports
from generators import *



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



#Paint the lines with corresponding colors
colors=['blue','green','red','cyan','magenta','yellow','black','white']
color_index=0



#The amount of datapoints we have per step
datapoints=150
    
#Amount of model parameters. Total parameters are 4n+2
num_params=8




#Iterate through all the trials for a test subject
for subject in raw_walking_data['Gaitcycle'].keys():
    
    print("Doing subject: " + subject)
    #%% This section is dedicated to setting up the data
    left_hip_total=get_trials_in_numpy(subject)
    phi_total=get_phi_from_numpy(left_hip_total)
    step_length_total=get_step_length(subject)   
    
    print("Got data for: " + subject)
    # This section is dedicated to regression based on fourier series
    #This is mostly from: https://stackoverflow.com/questions/52524919/fourier-series-fit-in-python
    
    x, y, z = variables('x, y, z')
    w, = parameters('w')
    model_dict = {z: fourier_series(x, y, f=w, n=num_params)}
    
    fit = Fit(model_dict, x=phi_total, y=step_length_total, z=left_hip_total)
    fit_result = fit.execute()
        
    print(fit_result)
    
    #Plot the result
    plt.plot(phi_total[0], left_hip_total[0], color=colors[color_index])
    plt.plot(phi_total[0], fit.model(x=phi_total[0], y=step_length_total[0],
                            **fit_result.params).z, color=colors[color_index], ls=':')
    color_index=(color_index+1)%len(colors)
    plt.show
    
    parameters_per_walk_config += [fit_result.params.copy()]
    parameter_list += fit_result.params.copy().values()
    
    
    #parameters_per_walk_config[-1]['walking_config'] = walking_configuration
    
#%% This section is dedicated to PCA on the coefficients for the fourier 
### transform


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








#Change model fitting to grays algorithm
#Visualizations