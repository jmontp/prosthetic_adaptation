import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
import numpy as np 
import pickle
import os, sys


#Importing the janky way since its too hard to do it the right way
PACKAGE_PARENT = '../../model_fitting/'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
new_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
print(new_path)
new_path = "/home/jmontp/workspace/prosthetic_adaptation/model_fitting"
sys.path.insert(1,new_path)


from kronecker_model import Kronecker_Model, model_loader, get_rmse


dataset_name = '../../local-storage/test/dataport_flattened_partial_{}.parquet'
model_name = '../../model_fitting/foot_model.pickle'


with open(model_name, "rb") as pickle_file:
    model_foot = pickle.load(pickle_file)



subject_names = ['AB01','AB02','AB03']


joint = 'jointangles_foot_x'

x_axis = np.linspace(0,1,150)

ax = plt.subplot(111)


for name in subject_names:
    file_name = dataset_name.format(name)
    
    file_data = pd.read_parquet(file_name, 
                                columns = [joint, 'phase', 'phase_dot', 'ramp', 'step_length'])
    
    file_data_cropped = file_data.head(150)
    
    
    model_output_no_personalization = model_foot.evaluate_pandas(file_data_cropped)
    
    
    
    
    model_output = model_output_no_personalization @ \
        (model_foot.cross_model_personalization_map@model_foot.subjects['AB01']['cross_model_gait_coefficients_unscaled'] + \
         model_foot.cross_model_inter_subject_average)
    
    file_numpy = file_data_cropped[joint].values

    default_cycler = (cycler(color=['r', 'g', 'b', 'c','m','y']) + \
                      cycler(alpha=[1.0, 0.0, 0.0, 0.4, 0.4, 0.4]))

    plt.rc('lines', linewidth=4)
    plt.rc('axes', prop_cycle=default_cycler)

    ax.plot(x_axis,file_numpy)



#Now plot the model
name = 'AB01'

file_name = dataset_name.format(name)
    
file_data = pd.read_parquet(file_name, 
                            columns = [joint, 'phase', 'phase_dot', 'ramp', 'step_length'])

file_data_cropped = file_data.head(150)


model_output_no_personalization = model_foot.evaluate_pandas(file_data_cropped)

gait_fingerprint = model_foot.subjects['AB01']['cross_model_gait_coefficients_unscaled']
personalization_map = model_foot.cross_model_personalization_map
average_fit = model_foot.cross_model_inter_subject_average


model_output = model_output_no_personalization @ (personalization_map @ gait_fingerprint + average_fit)


average_output = model_output_no_personalization @ average_fit


ax.plot(x_axis, average_output)
ax.plot(x_axis, model_output)


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
    
plt.axvspan(0.35,0.65, alpha=0.5, color='yellow')

plt.savefig('slide4.png', transparent=True)


plt.show()