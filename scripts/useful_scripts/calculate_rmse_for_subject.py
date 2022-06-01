#Normal imports 
import numpy as np 
import pandas as pd

#Custom imports
from context import kmodel
from kmodel.personalized_model_factory import PersonalizedKModelFactory


#Desired joint angle
joint = 'jointangles_thigh_x'

##Import the personalized model 
factory = PersonalizedKModelFactory()

#Define a list for all the models
subject_list = [f"AB{i:02}" for i in range(1,11)]

for subject_model in subject_list:
    model_dir = f'../../data/kronecker_models/left_one_out_model_{subject_model}.pickle'
    #load the model
    model = factory.load_model(model_dir)

    #Get the subject data
    file_location = "../../data/flattened_dataport/dataport_flattened_partial_{}.parquet"
    #Get the file for the corresponding subject
    filename = file_location.format(subject_model)
    #Read in the parquet dataframe
    total_data = pd.read_parquet(filename)

    #Apply a zero ramp data mask
    mask = total_data['ramp']==0.0
    total_data = total_data[mask]

    #State ordering to get numpy array
    states = ['phase','phase_dot','stride_length']
    numpy_data = total_data[states].values

    #Get ground truth data
    true_joint_angles = total_data[joint].values.reshape(-1,1)

    #Evaluate with the average model and residual
    average_model_joint_angles = model.evaluate(numpy_data,use_average_fit=True)
    average_model_residual = np.sqrt(np.mean(np.power(average_model_joint_angles - true_joint_angles,2)))


    #Evaluate with subject specific model and residual
    ls_gf_model_joint_angles = model.evaluate(numpy_data,use_personalized_fit=True)
    ls_gf_model_residual = np.sqrt(np.mean(np.power(ls_gf_model_joint_angles - true_joint_angles,2)))

    #Print out in latex style
    print(f"{subject_model} & {average_model_residual:.3} & {ls_gf_model_residual:.3} \\\\")
    print("\hline")
