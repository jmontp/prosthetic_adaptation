#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:55:34 2021

@author: jmontp
"""

from model_framework import Polynomial_Basis, Fourier_Basis, Kronecker_Model
import pandas as pd
import dask.dataframe as dd
from dask import delayed
from dask.distributed import Client
import numpy as np 
import h5py


#Initialize the model that we are going to base the regressor on
phase_model = Fourier_Basis(6,'phase')
phase_dot_model = Polynomial_Basis(3, 'phase_dot')
ramp_model = Polynomial_Basis(3, 'incline')
#No step length implemented for now
#step_length_model = Polynomial_Basis(3,'step_length')
model_hip = Kronecker_Model('kinematics_jointangles_ankle_x', phase_dot_model, ramp_model,phase_model)


subjects = [('AB10','local-storage/test/InclineExperiment_AB10.par')]
for i in range(1,10):
	subjects.append(('AB0'+str(i),'local-storage/test/InclineExperiment_AB0'+str(i)+'.par'))

print("Adding subjects")
model_hip.add_subject(subjects)
print("Fitting?")
model_hip.fit_subjects()
