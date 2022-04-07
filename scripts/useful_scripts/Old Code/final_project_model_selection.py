#Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


#Relative Imports
from context import kmodel
from kmodel.kronecker_model import KroneckerModel, model_loader, model_saver, calculate_cross_model_p_map
from kmodel.function_bases import FourierBasis, HermiteBasis, ChebyshevBasis

#Determine if you want to save the models
# set false to debug
save_models = False

save_model_dir = '../../data/kronecker_models/'

def train_models():
    pass
    #%%
    
    #Determine the phase models
    phase_model = FourierBasis(3,'phase')
    phase_dot_model = HermiteBasis(3,'phase_dot')
    stride_length_model = HermiteBasis(2,'stride_length')
    ramp_model = HermiteBasis(3,'ramp')


    num_gait_fingerprints = 9
    
    # #Get the subjects
    subject_10 = ('AB10','../../data/flattened_dataport/dataport_flattened_partial_AB10.parquet')
    left_out = []
    subjects = []
    for i in range(1,10):
        subjects.append(('AB0'+str(i),'../../data/flattened_dataport/dataport_flattened_partial_AB0'+str(i)+'.parquet'))

    subjects.append(subject_10)

    feature = 'jointangles_knee_x'

    residuals_sum = [0]*10
    ls_residual_sum = 0
    ls_count = 0
    count = 0


    #Find the model that does not overfit
    MODEL_FIT = True

    #Vary specific paremeters
    VARY_PARAMTERS = False

    if MODEL_FIT == True:
        model = KroneckerModel(feature,phase_model,
                                    phase_dot_model,stride_length_model,
                                    ramp_model,subjects=subjects,
                                    num_gait_fingerprint=num_gait_fingerprints)
        


        for subject_name, subject_dict in model.subjects.items():
            ls_residual_sum += subject_dict['gp_residual']
            ls_count += 1
            print(f"{subject_name} rmse error {subject_dict['residual']} avg subject rmse error {subject_dict['avg_subject_residual']}")

        print(f"Average rmse error {ls_residual_sum/ls_count}")    


    if VARY_PARAMTERS == True:
        for i in range(10):

            #Remove one subject
            
            left_out = [subjects.pop()]

            print(f"Left out subject {left_out}")

            model = KroneckerModel(feature,phase_model,
                                    phase_dot_model,stride_length_model,
                                    ramp_model,subjects=subjects,
                                    num_gait_fingerprint=num_gait_fingerprints)
            
            model.add_left_out_subject(left_out)

            

            #Average out the one-left-out cross validation residuals 
            # per amount of gait fingerprints
            current_residuals = model.one_left_out_subjects[left_out[0][0]]['gf_residual']
            for i in range(len(current_residuals)):
                residuals_sum[i] += current_residuals[i]
            count += 1


            for subject_name, subject_dictionary in model.subjects.items():
                ls_residual_sum += subject_dictionary['residual']
                ls_count += 1

            #Reinsert at the top
            subjects.insert(0,left_out[0])

        mean_residuals = [f"{float(i/count):.4f}" for i in residuals_sum]

        print(f"Residuals per gf number are {' & '.join(mean_residuals)}")
        print(f"Mean residual from least squares is {ls_residual_sum/ls_count}")
        pass

#%%
if __name__=='__main__':
    pass
    train_models()
