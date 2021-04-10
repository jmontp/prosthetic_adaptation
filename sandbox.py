#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:55:34 2021

@author: jmontp
"""

from model_fitting.model_framework import Polynomial_Basis, Fourier_Basis, Kronecker_Model
import pandas as pd
import numpy as np 
import h5py
import matplotlib.pyplot as plt



#Initialize the model that we are going to base the regressor on
phase_model = Fourier_Basis(9,'phase')
phase_dot_model = Polynomial_Basis(5, 'phase_dot')
ramp_model = Polynomial_Basis(5, 'ramp')
#No step length implemented for now
step_length_model = Polynomial_Basis(5,'step_length')
model_hip = Kronecker_Model('jointangles_hip_x', phase_dot_model,ramp_model,phase_model,step_length_model)


subjects = [('AB10','./local-storage/test/dataport_flattened_partial_AB10.parquet')]
for i in range(1,10):
	subjects.append(('AB0'+str(i),'local-storage/test/dataport_flattened_partial_AB0'+str(i)+'.parquet'))

print("Adding subjects")
model_hip.add_subject(subjects)
print("Fitting?")
model_hip.fit_subjects()

model_hip.calculate_gait_fingerprint()

#%%
#Plot cumulative variance
x = range(1,11)

plt.plot(x,np.cumsum(model_hip.pca_result.explained_variance_ratio_))
plt.plot(3,np.cumsum(model_hip.pca_result.explained_variance_ratio_)[2], 'ro')
#plt.xlabel("Number of PCA Components")
#plt.ylabel("Variance Explained")
plt.savefig('Cumulative sum of explained variance.svg',format='svg')
plt.show()
print(np.cumsum(model_hip.pca_result.explained_variance_ratio_)[2])

#%%
#Plot predicted output
step = 150
x = np.linspace(0,1,150)
predicted_output = model_hip.evaluate_subject("AB01",model_hip.subjects["AB01"]['dataframe'][step:step+150])
plt.plot(x,predicted_output)
expected_output = model_hip.subjects["AB01"]['dataframe']['jointangles_hip_x'][step:step+150].values.reshape(-1,1)
plt.plot(x,expected_output)
#plt.xlabel("Phase")
#plt.ylabel("Hip Angle")
plt.legend(["Model", "Actual"])
plt.savefig('Actual hig angle vs. Predicted hip angle.svg',format='svg')
rmse = np.sqrt(np.mean(np.power((expected_output-predicted_output),2)))
print(rmse)
plt.show()
#%%

cols = ['jointangles_hip_x', 'phase']

ab01_data = pd.read_parquet('local-storage/test/dataport_flattened_partial_AB09.parquet',columns=cols)
ab02_data = pd.read_parquet('local-storage/test/dataport_flattened_partial_AB02.parquet',columns=cols)
ab03_data = pd.read_parquet('local-storage/test/dataport_flattened_partial_AB03.parquet',columns=cols)
ab04_data = pd.read_parquet('local-storage/test/dataport_flattened_partial_AB04.parquet',columns=cols)

ab01_data[ab01_data['phase'] == 1-1/150] = np.nan

plt.plot(ab01_data['phase'][:1500], ab01_data['jointangles_hip_x'][:1500])
plt.xlabel("Phase")
plt.ylabel("AB09 Hip Angle (Deg)")
plt.show()


# ab02_data[ab02_data['phase'] == 1-1/150] = np.nan

# plt.plot(ab02_data['phase'][:1500], ab02_data['jointangles_hip_x'][:1500])
# plt.xlabel("Phase")
# plt.ylabel("AB02 Hip Angle (rad)")
# plt.show()


# ab03_data[ab03_data['phase'] == 1-1/150] = np.nan

# plt.plot(ab03_data['phase'][:1500], ab03_data['jointangles_hip_x'][:1500])
# plt.xlabel("Phase")
# plt.ylabel("AB03 Hip Angle (rad)")
# plt.show()


# ab04_data[ab04_data['phase'] == 1-1/150] = np.nan

# plt.plot(ab04_data['phase'][:1500], ab04_data['jointangles_hip_x'][:1500])
# plt.xlabel("Phase")
# plt.ylabel("AB04 Hip Angle (rad)")
# plt.show()