#Standard Imports
import numpy as np
from sympy import uppergamma 

#Import from same folder
from .measurement_model import MeasurementModel
from .dynamic_model import GaitDynamicModel

#Get relative imports
from .context import math_utils 

#Set this to false to speed up calculations but not test if ekf is PSD
math_utils.test_pd = True

class Extended_Kalman_Filter:

    def __init__(self,initial_state: np.ndarray, initial_covariance: np.ndarray , 
                 dynamic_model: GaitDynamicModel,process_noise: np.ndarray,
                 measurement_model: MeasurementModel,observation_noise: np.ndarray,
                 output_model: MeasurementModel = None, 
                 lower_state_limit: np.ndarray = None, 
                 upper_state_limit: np.ndarray = None):
        """
        Create the extended Kalman filter object

        Keyword Arguments
        initial_state: initial state of the extended kalman filter
                       shape(num_states,1)

        initial_covariance: initial convariance of the extended kalman filter
                            shape(num_states, num_states)

        dynamic_model: Dynamic model that generates the predicted states

        process noise: noise that is applied to the dynamic model
                       shape(num_states, num_states)
        
        measurement_model: Measurement model that generates the expected measurements for a predicted state
                           
        observation_noise: Noise that is applied to the measurement model
                           shape(num_measurements, num_measurements)
        
        output_model: Optional, calculates an output based on the current state

        lower_state_limit: sets a lower bound on the states of the system
                           shape(num_states, 1)

        upper_state_limit: sets an upper bound on the states of the system
                           shape(num_states, 1)
        """

        #Assign internal variables
        self.dynamic_model = dynamic_model
        self.measurement_model = measurement_model
        self.x = initial_state
        self.P = initial_covariance
        self.Q = process_noise
        self.R = observation_noise
        self.calculated_measurement_ = None
        self.num_states = initial_state.shape[0]
        
        #Optinal, output model
        self.output_model = output_model
        
        #Calculate output based on initial conditions if it is defined
        if self.output_model is not None: 
            self.output = self.output_model.evaluate_h_func(initial_state)
        else:
            self.output = None

        #Set saturation limits
        #upper limit of infinity
        if (upper_state_limit is not None):
            if upper_state_limit.shape != initial_state.shape:
                raise Exception(f"Upper state limit does not have the same shape as the initial state {upper_state_limit.shape} vs {initial_state.shape}")
            self.upper_state_limit = upper_state_limit
        else: 
            self.upper_state_limit = np.ones(initial_state.shape) + np.inf
        #lower limit of minus infinity
        if (lower_state_limit is not None):
            if lower_state_limit.shape != initial_state.shape:
                raise Exception(f"Upper state limit does not have the same shape as the initial state {lower_state_limit.shape} vs {initial_state.shape}")
            self.lower_state_limit = lower_state_limit
        else: 
            self.lower_state_limit = np.ones(initial_state.shape) - np.inf
    
    
    
    #Getter for output
    def get_output(self):
        return self.output


    #Calculate the next estimate of the kalman filter
    def calculate_next_estimates(self, time_step, sensor_measurements, control_input_u=0):
        
        #Run the prediction step
        predicted_state, predicted_covariance = self.preditction_step(time_step)

        #Saturate the predicted state
        predicted_state = np.clip(predicted_state, self.lower_state_limit, self.upper_state_limit)

        #Store predicted state for debugging purpose
        self.predicted_state = predicted_state
        self.predicted_covariance = predicted_covariance
        
        #Run the measurement step
        updated_state, updated_covariance = self.update_step(predicted_state, predicted_covariance, sensor_measurements)

        #Saturate the updated state
        updated_state = np.clip(updated_state, self.lower_state_limit, self.upper_state_limit)

        #Store the updated state and covariance
        self.x = updated_state
        self.P = updated_covariance

        # Calculate the output
        if (self.output_model is not None):
            self.output = self.output_model.evaluate_h_func(updated_state)

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




# TODO: Update the default model to use the new kronecker model stuff
# def default_ekf(R=None,Q=None):
#     from .context import kmodel
#     from kmodel.personalized_model_factory import PersonalizedKModelFactory    

#     #Phase, Phase Dot, Ramp, Step Length, 5 gait fingerprints
#     state_names = ['phase', 'phase_dot', 'stride_length', 'ramp',
#                     'gf1', 'gf2','gf3', 'gf4', 'gf5']

#     ## Load the measurement and output models

#     #Initialize the measurement model
#     #Define the joints that you want to import 
#     joint_names = ['jointangles_hip_dot_x','jointangles_hip_x',
#                     'jointangles_knee_dot_x','jointangles_knee_x',
#                     'jointangles_thigh_dot_x','jointangles_thigh_x']

#     #Import the personalized model 
#     factory = PersonalizedKModelFactory()

#     subject_model = "AB01"

#     model_dir = f'../../data/kronecker_models/left_one_out_model_{subject_model}.pickle'

#     model = factory.load_model(model_dir)
    
#     measurement_model = MeasurementModel(model)

#     #Initialize the output model 
#     #Get the torque models from 
#     # torque_names = ['jointmoment_hip_x', 'jointmoment_knee_x']
#     # torque_models = [model_loader(model_dir.format(torque)) for torque in torque_names]

#     # output_model = MeasurementModel(state_names, torque_models)


#     #Initialize gait fingerprint to all zeros
#     initial_gait_fingerprint = np.array([[0.0,0.0,0.0,0.0,0.0]]).T    

#     #Setup initial states to zero 
#     #Phase, Phase, Dot, Stride_length, ramp
#     initial_state_partial= np.array([[0.0,0.0,1.0,0.0]]).T
#     initial_state = np.concatenate((initial_state_partial,initial_gait_fingerprint))

#     #Generate the initial covariance as being very low
#     #TODO - double check with gray if this was the strategy that converged or not
#     cov_diag = 1e-5
#     initial_state_diag = [cov_diag,cov_diag,cov_diag,cov_diag,
#                           cov_diag,cov_diag,cov_diag,cov_diag,cov_diag]
#     initial_state_covariance = np.diag(initial_state_diag)


#     #Measurement covarience, Innovation
#     r_diag = [3000,25,3000,25,3000,25]
#     R_default = np.diag(r_diag)

#     #Verify if input is none
#     if(R is None):
#         R = R_default

#     #Process noise
#     #Phase, Phase, Dot, Stride_length, ramp, gait fingerprints

#     q_diag = [0,3e-7,5e-8,1e-8,
#             1e-8,1e-8,1e-8,1e-8,1e-8]
#     Q_default = np.diag(q_diag)

#     if(Q is None):
#         Q = Q_default

#     ###################

#     d_model = GaitDynamicModel()

#     ekf_instance = Extended_Kalman_Filter(initial_state,initial_state_covariance, d_model, Q, measurement_model, R)

#     return ekf_instance

