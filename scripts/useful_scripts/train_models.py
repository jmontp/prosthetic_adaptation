#Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


#Relative Imports
from context import kmodel
from kmodel.kronecker_model import KroneckerModel, model_loader, model_saver, calculate_cross_model_p_map
from kmodel.function_bases import FourierBasis, PolynomialBasis

#Determine if you want to save the models
# set false to debug
save_models = True

save_model_dir = '../../data/kronecker_models/'

def train_models():
    pass
    #%%
    
    #Determine the phase models
    phase_model = FourierBasis(8,'phase')
    phase_dot_model = PolynomialBasis(1,'phase_dot')
    stride_length_model = PolynomialBasis(2,'stride_length')
    ramp_model = PolynomialBasis(6,'ramp')

    num_gait_fingerprints = 5
    
    # #Get the subjects
    left_out = [('AB10','../../data/flattened_dataport/dataport_flattened_partial_AB10.parquet')]
    subjects = []
    for i in range(1,10):
        subjects.append(('AB0'+str(i),'../../data/flattened_dataport/dataport_flattened_partial_AB0'+str(i)+'.parquet'))

    feature_list = ['jointangles_hip_x', 'jointangles_knee_x', 'jointangles_thigh_x', 
                    'jointangles_hip_dot_x', 'jointangles_knee_dot_x', 'jointangles_thigh_dot_x',
                    'jointmoment_hip_x', 'jointmoment_knee_x']

    # personalization_list = ['jointangles_hip_x', 'jointangles_knee_x', 'jointangles_thigh_x', 
    #                 'jointangles_hip_dot_x', 'jointangles_knee_dot_x', 'jointangles_thigh_dot_x']

    personalization_list = feature_list #have torque also be personalized 

    model_list = []
    personalization_models = []

    for feature in feature_list:
        #Create the hip model
        model = KroneckerModel(feature,phase_model,
                                phase_dot_model,stride_length_model,
                                ramp_model,subjects=subjects,
                                num_gait_fingerprint=num_gait_fingerprints)
        
        model.add_left_out_subject(left_out)
        model_list.append(model)

        if feature in personalization_list:
            personalization_models.append(model)

    #Need to run this before they are saved
    print("Calculating cross models")
    calculate_cross_model_p_map(personalization_models)

    #Save the models
    if save_models == True:
        for feature_name, feature_model in zip(feature_list, model_list):
            model_saver(feature_model, save_model_dir + 'model_{}.pickle'.format(feature_name))
        
      
def calculate_cross_model():

    model_knee = model_loader('model_knee.pickle')
    model_knee_dot = model_loader('model_knee_dot.pickle')
    model_hip = model_loader('model_hip.pickle')
    model_hip_dot = model_loader('model_knee_dot.pickle')
    model_thigh = model_loader('model_thigh.pickle')
    model_thigh_dot = model_loader('model_thigh_dot.pickle')

    models = [model_knee, model_knee_dot, model_hip, model_hip_dot, model_thigh, model_thigh_dot]

    calculate_cross_model_p_map(models)
        

    
    

#%%
if __name__=='__main__':
    pass
    train_models()
    #calculate_cross_model()