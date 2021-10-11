#Standard Imports
import numpy as np 

#Import from same folder
from .measurement_model import MeasurementModel
from .dynamic_model import GaitDynamicModel

#Get relative imports
from .context import math_utils 

#Set this to false to speed up calculations but not test if ekf is PSD
math_utils.test_pd = True

class Extended_Kalman_Filter:

    def __init__(self,initial_state,initial_covariance,dynamic_model,process_noise,measurement_model,observation_noise):
        self.dynamic_model = dynamic_model
        self.measurement_model = measurement_model
        self.x = initial_state
        self.P = initial_covariance
        self.Q = process_noise
        self.R = observation_noise
        self.calculated_measurement_ = None
        self.num_states = initial_state.shape[0]


    #Calculate the next estimate of the kalman filter
    def calculate_next_estimates(self, time_step, control_input_u, sensor_measurements):
        
        predicted_state, predicted_covariance = self.preditction_step(time_step)
        self.predicted_state = predicted_state
        self.predicted_covariance = predicted_covariance
        
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
        
        #print("Dynamic model jacobean F: {}".format(F))
        
        #Get the new measurements
        new_covariance = F @ self.P @ F.T + self.Q
        
        math_utils.assert_pd(new_covariance-self.Q,"Updated covariance")
        
        return (new_state, new_covariance)

    
    #This function will calculate the new state based on the predicted state
    def update_step(self, predicted_state, predicted_covariance, sensor_measurements):

        #Calculate the expected measurements (z)
        expected_measurements = self.measurement_model.evaluate_h_func(predicted_state)
        self.calculated_measurement_ = expected_measurements

        #Calculate the innovation
        y_tilde = (sensor_measurements - expected_measurements)
        self.y_tilde = y_tilde
        
        #Get the jacobian for the measurement function
        H = self.measurement_model.evaluate_dh_func(predicted_state)

        #Calculate the innovation covariance
        S = H @ predicted_covariance @ H.T + self.R
        
        #Verify if S is PD
        math_utils.assert_pd(S-self.R, "S-R")
        
        #Calculate the Kalman Gain
        K = predicted_covariance @ H.T @ np.linalg.inv(S)
        
        #Calculate the updated state
        self.delta_state = K @ y_tilde
        updated_state = predicted_state + self.delta_state

        #Calculate the updated covariance
        I = np.eye(self.num_states)
        updated_covariance = (I - K @ H) @ predicted_covariance
        
        #Verify that updated covariance is done
        math_utils.assert_pd(updated_covariance, "Updated Covariance")
        
        return updated_state, updated_covariance
