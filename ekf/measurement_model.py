import numpy as np
from .context import kmodel
from kmodel.personal_measurement_function import PersonalMeasurementFunction

class MeasurementModel():
    """
    This class wraps the PersonalMeasurementFunction and adds the
    numerical differentiation required for the extended kalman filter
    """



    def __init__(self,personal_model : PersonalMeasurementFunction):
        
        #The initialization method only requires the PersonalMeasurementFunction
        self.personal_model = personal_model
        
        
        
    def evaluate_h_func(self,current_state):
        """
        Evaluate the measurement function

        Keyword Arguments
        current_state: current state of the extended kalman filter
                       shape(num_states, 1)
        
        Returns
        measurement_value: Value of the measurement function
                           shape(num_outputs, 1)

        """


        #Measurment models receive their input as row vectors and the 
        # ekf inputs column vectors
        # therefore, transpose both input and output
        return self.personal_model.evaluate(current_state.T).T



    def evaluate_dh_func(self,current_state):
        """
        Calculate the derivative at this current point in time

        Currently only support numerical differentiation

        Keyword Arguments
        current_state: current state of the extended kalman filter
                       shape(num_states, 1)
        
        Returns
        measurement_value: Value of the derivative of the measurement function
                           shape(num_outputs, 1)
        """
        #Use numerical method to calculate the jacobean 
        result = self.numerical_jacobean(current_state)

        return result

    def numerical_jacobean(self, current_state):

        """
        Numerical differentiation algorithm

        Keyword Arguments
        current_state: current state of the extended kalman filter
                       shape(num_states, 1)
        
        Returns
        measurement_value: Value of the derivative of the measurement function
                           shape(num_outputs, 1)
        """
        num_states = current_state.shape[0]

        #create buffer for the derivative
        manual_derivative = np.zeros((self.personal_model.num_kmodels,num_states))
        
        #We are going to fill it element-wise
        #col represents the current state
        f_state = self.evaluate_h_func(current_state)

        #Create buffer to store the result
        state_plus_delta = current_state.copy()
        delta = 1e-6


        for col in range(num_states):
            #Increment the state in the appropriate dimension
            state_plus_delta[col,0] += delta

            #Eliminate the previous increment
            if(col != 0):
                state_plus_delta[col-1,0] -= delta

            #Evaluate the model and extract it 
            f_delta = self.evaluate_h_func(state_plus_delta)

            #Get the derivative for the column
            manual_derivative[:,col] = ((f_delta-f_state)/(delta)).T
            
        return manual_derivative