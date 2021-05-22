import sys
import os

PACKAGE_PARENT = '../model_fitting/'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
new_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
print(new_path)
sys.path.insert(1,new_path)


import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from function_bases import Polynomial_Basis, Fourier_Basis 
from kronecker_model import Kronecker_Model
from measurement_model import Measurement_Model
from dynamic_model import Gait_Dynamic_Model


class Extended_Kalman_Filter:

    def __init__(self,initial_state,initial_covariance,dynamic_model,process_noise,measurement_model,observation_noise):
        self. dynamic_model = dynamic_model
        self.measurement_model = measurement_model
        self.x = initial_state
        self.P = initial_covariance
        self.Q = process_noise
        self.R = observation_noise


    #Calculate the next estimate of the kalman filter
    def calculate_next_estimates(self, time_step, control_input_u, sensor_measurements):
        
        predicted_state, predicted_covariance = self.preditction_step(time_step)
        updated_state, updated_covariance = self.update_step(predicted_state, predicted_covariance, sensor_measurements)

        self.x = updated_state
        self.P = updated_covariance

        return updated_state, updated_covariance


    #Call this function to get the next state 
    def preditction_step(self, time_step,control_input_u=0):
        #Calculate the new step with the prediction function
        new_state = self.dynamic_model.f_function(self.x, time_step)

        #Calculate the jacobian of f_function
        F = self.dynamic_model.f_jacobean(self.x, time_step)

        #Get the new measurements
        new_covariance = F @ self.P @ F.T + self.Q

        return (new_state, new_covariance)

    
    #This function will calculate the new state based on the predicted state
    def update_step(self, predicted_state, predicted_covariance, sensor_measurements):
        #print("sensor measurements" + str(sensor_measurements))
        #Calculate the expected measurements (z)
        expected_measurements = self.measurement_model.evaluate_h_func(predicted_state)
        #print("Expected Measurements" + str(expected_measurements))

        #Calculate the innovation
        #print("Sensor shape " + str(sensor_measurements.shape) + "Expected shape " + str(expected_measurements.shape))
        y_hat = sensor_measurements - expected_measurements
    
        #Get the jacobian for the measurement function
        H = self.measurement_model.evaluate_dh_func(predicted_state)

        #Calculate the innovation covariance
        S = H @ predicted_covariance @ H.T + self.R

        #Calculate the Kalman Gain
        K = predicted_covariance @ H.T @ np.linalg.inv(S)

        #Calculate the updated state
        predicted_state_vector = predicted_state.values.T
        
        updated_state_vector = predicted_state_vector + K @ y_hat

        #print("ekf: Updated state vector" + str(updated_state_vector))
        #copy the column names
        #updated_state = pd.DataFrame(updated_state_vector.T,columns=predicted_state.columns)
        predicted_state.iloc[0] = updated_state_vector.ravel()
        #Calculate the updated covariance
        updated_covariance = predicted_covariance - K @ H @ predicted_covariance

        return predicted_state, updated_covariance






################################################################################################
################################################################################################
#Unit testing
#Save the model so that you can use them later
def model_saver(model,filename):
    with open(filename,'wb') as file:
        pickle.dump(model,file)

#Load the model from a file
def model_loader(filename):
    with open(filename,'rb') as file:
        return pickle.load(file)

def ekf_unit_test():
    pass
    #%%
    #Phase, Phase Dot, Ramp, Step Length, 4
    initial_state_dict = {'phase': [0],
                    'phase_dot': [1],
                    'step_length': [1],
                    'ramp': [0],
                    'gf1': [0],
                    'gf2': [0],
                    'gf3': [0],
                    'gf4': [0]}

    initial_state = pd.DataFrame(initial_state_dict)
    
    train_models = False
    if train_models == True:
        #Determine the phase models
        phase_model = Fourier_Basis(8,'phase')
        phase_dot_model = Polynomial_Basis(3,'phase_dot')
        step_length_model = Polynomial_Basis(3,'step_length')
        ramp_model = Polynomial_Basis(3,'ramp')
    
        # #Get the subjects
        subjects = [('AB10','../local-storage/test/dataport_flattened_partial_AB10.parquet')]
        for i in range(1,10):
            subjects.append(('AB0'+str(i),'../local-storage/test/dataport_flattened_partial_AB0'+str(i)+'.parquet'))

        model_foot = Kronecker_Model('jointangles_foot_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=4)
        model_saver(model_foot,'foot_model.pickle')
        
        model_shank = Kronecker_Model('jointangles_shank_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=4)
        model_saver(model_shank,'shank_model.pickle')
    
        model_foot_dot = Kronecker_Model('jointangles_foot_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=4,time_derivative=True)
        model_saver(model_foot_dot,'foot_dot_model.pickle')
        
        model_shank_dot = Kronecker_Model('jointangles_shank_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=4,time_derivative=True)
        model_saver(model_shank_dot,'shank_dot_model.pickle')
        
    else:
        model_foot = model_loader('foot_model.pickle')
        model_shank = model_loader('shank_model.pickle')
        model_foot_dot = model_loader('foot_dot_model.pickle')
        model_shank_dot = model_loader('shank_dot_model.pickle')

    models = [model_foot,model_shank,model_foot_dot,model_shank_dot]
    state_names = list(initial_state_dict.keys())

    measurement_model = Measurement_Model(state_names,models)

    n = len(state_names)
    num_outputs = len(models)
    initial_state_covariance = np.zeros((n,n))

    R = np.eye(num_outputs)

    Q = np.eye(n)

    d_model = Gait_Dynamic_Model()

    ekf = Extended_Kalman_Filter(initial_state,initial_state_covariance, d_model, Q, measurement_model, R)

    time_step = 0.5

    #Really just want to prove that we can do one interation of this
    #Dont really want to pove much more than this since we would need actual data for that
    
    control_input_u = 0 

    sensor_measurements = np.array([[1,1,1,1]]).T

    state_history = initial_state.copy()

    try:
        for i in range(100):
            state_history = state_history.append(ekf.calculate_next_estimates(time_step, control_input_u, sensor_measurements)[0],ignore_index=True)
    except KeyboardInterrupt:
        pass
        
    print(state_history['gf1'])
    plt.plot(state_history['gf1'])
    plt.show()

#%%
if(__name__=='__main__'):
    ekf_unit_test()