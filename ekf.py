

import numpy as np 

from model_framework import model_loader
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
	def update_step(self, predicted_state, predicted_covariance, measurements):

		#Calculate the expected measurements (z)
		expected_measurements = self.measurement_model.evaluate_h_func(*predicted_state)

		#Calculate the innovation
		y_hat = measurements - expected_measurements

		#Get the jacobian for the measurement function
		H = self.measurement_model.evaluate_dh_func(*predicted_state)

		#Calculate the innovation covariance
		S = H @ predicted_covariance @ H.T + self.R

		#Calculate the Kalman Gain
		K = predicted_covariance @ H.T @ np.linalg.inv(S)

		#Calculate the updated state
		updated_state = predicted_state + K @ y_hat

		#Calculate the updated covariance
		updated_covariance = predicted_covariance - K @ H @ predicted_covariance

		return updated_state, updated_covariance






################################################################################################
################################################################################################
#Unit testing


def ekf_unit_test():
	
	#Phase, Phase Dot, Ramp, Step Length, 6 
	state = np.array([0,1,0,0,
					  0,0,0,0,0,0]).T

	n = state.shape[0]
	state_covariance = np.zeros((n,n))

	R = np.eye(2)

	Q = np.eye(n)

	m_model = model_loader('H_model.pickle')
	d_model = Gait_Dynamic_Model()

	ekf = Extended_Kalman_Filter(state,state_covariance, d_model, Q, m_model, R)

	time_step = 0.5


	#Really just want to prove that we can do one interation of this
	#Dont really want to pove much more than this since we would need actual data for that
	
	next_state = ekf.calculate_next_estimates(time_step, 0, (1,1))

	print(next_state)


if(__name__=='__main__'):
	ekf_unit_test()