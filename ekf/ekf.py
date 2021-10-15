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

    def __init__(self,initial_state,initial_covariance,dynamic_model,process_noise,measurement_model,observation_noise,output_model=None):
        self.dynamic_model = dynamic_model
        self.measurement_model = measurement_model
        self.x = initial_state
        self.P = initial_covariance
        self.Q = process_noise
        self.R = observation_noise
        self.calculated_measurement_ = None
        self.num_states = initial_state.shape[0]
        
        self.output_model = output_model
        
        if self.output_model is not None: 
            self.output = self.output_model.evaluate_h_func(initial_state)
        else:
            self.output = None


    #Getter for output
    def get_output(self):
        return self.output


    #Calculate the next estimate of the kalman filter
    def calculate_next_estimates(self, time_step, sensor_measurements, control_input_u=0):
        
        #Run the prediction step
        predicted_state, predicted_covariance = self.preditction_step(time_step)
        self.predicted_state = predicted_state
        self.predicted_covariance = predicted_covariance
        
        #Run the measurement step
        updated_state, updated_covariance = self.update_step(predicted_state, predicted_covariance, sensor_measurements)
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





def default_ekf(R=None,Q=None):
    from .context import kmodel
    from kmodel.kronecker_model import model_loader


    #Phase, Phase Dot, Ramp, Step Length, 5 gait fingerprints
    state_names = ['phase', 'phase_dot', 'stride_length', 'ramp',
                    'gf1', 'gf2','gf3', 'gf4', 'gf5']

    ## Load the measurement and output models

    #Initialize the measurement model
    #Define the joints that you want to import 
    joint_names = ['jointangles_hip_dot_x','jointangles_hip_x',
                    'jointangles_knee_dot_x','jointangles_knee_x',
                    'jointangles_thigh_dot_x','jointangles_thigh_x']

    model_dir = '../../data/kronecker_models/model_{}.pickle'

    models = [model_loader(model_dir.format(joint)) for joint in joint_names]
    
    measurement_model = MeasurementModel(state_names,models)

    #Initialize the output model 
    #Get the torque models from 
    torque_names = ['jointmoment_hip_x', 'jointmoment_knee_x']
    torque_models = [model_loader(model_dir.format(torque)) for torque in torque_names]

    output_model = MeasurementModel(state_names, torque_models)


    #Initialize gait fingerprint to all zeros
    initial_gait_fingerprint = np.array([[0.0,0.0,0.0,0.0,0.0]]).T    

    #Setup initial states to zero 
    #Phase, Phase, Dot, Stride_length, ramp
    initial_state_partial= np.array([[0.0,0.0,1.0,0.0]]).T
    initial_state = np.concatenate((initial_state_partial,initial_gait_fingerprint))

    #Generate the initial covariance as being very low
    #TODO - double check with gray if this was the strategy that converged or not
    cov_diag = 1e-5
    initial_state_diag = [cov_diag,cov_diag,cov_diag,cov_diag,
                          cov_diag,cov_diag,cov_diag,cov_diag,cov_diag]
    initial_state_covariance = np.diag(initial_state_diag)


    #Measurement covarience, Innovation
    r_diag = [25,25,25,3000,3000,3000]
    R_default = np.diag(r_diag)

    #Verify if input is none
    if(R is None):
        R = R_default

    #Process noise
    #Phase, Phase, Dot, Stride_length, ramp, gait fingerprints

    q_diag = [0,3e-7,5e-8,1e-8,
            1e-8,1e-8,1e-8,1e-8,1e-8]
    Q_default = np.diag(q_diag)

    if(Q is None):
        Q = Q_default

    ###################

    d_model = GaitDynamicModel()

    ekf_instance = Extended_Kalman_Filter(initial_state,initial_state_covariance, d_model, Q, measurement_model, R, output_model)

    return ekf_instance

