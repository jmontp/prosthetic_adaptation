import numpy as np
from .context import kmodel
from kmodel.model_definition.personal_measurement_function import PersonalMeasurementFunction

class MeasurementModel():
    """
    This class wraps the PersonalMeasurementFunction and adds the
    numerical differentiation required for the extended kalman filter
    """



    def __init__(self,personal_model : PersonalMeasurementFunction,
                      calculate_output_derivative: bool):
        
        #The initialization method only requires the PersonalMeasurementFunction
        self.personal_model = personal_model
        
        #Add property for calculating output derivative
        self.calculcate_output_derivative = calculate_output_derivative

        #Set the otuput names based on wheter we add time derivatives or not
        if calculate_output_derivative == True:
            
            #If we are calculating the derivatives, modify the output names to
            #  contain the time derivatives names
            output_names_with_dot = ['_'.join(name.split('_')[:-1] + ['dot'] + [name.split('_')[-1]]) for name in personal_model.output_names]
            self.output_names = personal_model.output_names + output_names_with_dot

        else:
            self.output_names = personal_model.output_names

        #Consistent differentiation step across all methods
        self.delta = 1e-7



    def evaluate_h_func(self,current_state:np.ndarray):
        """
        Evaluate the measurement function

        Keyword Arguments
        current_state: current state of the extended kalman filter
                       shape(num_states, 1)
        use_subject_average_fit: determines if the subject average fit is used in the 
                                 evaluation process of the measurement model
        use_least_squares_gf: determines if the least squares gf fit is used in the 
                                 evaluation process of the measurement model
        Returns
        measurement_value: Value of the measurement function
                           shape(num_outputs, 1)

        """


        #Measurment models receive their input as row vectors and the 
        # ekf inputs column vectors
        # therefore, transpose both input and output

        output = self.personal_model.evaluate(current_state.T)

        if (self.calculcate_output_derivative == True):
            #Calculate the time derivative
            output_time_derivative = self.output_time_derivative(current_state, output)
            
            #Append to the output
            output = np.concatenate([output,output_time_derivative], axis=1)

        return output.T



    def evaluate_dh_func(self,current_state:np.ndarray):
        """
        Calculate the derivative at this current point in time

        Currently only support numerical differentiation

        Keyword Arguments
        current_state: current state of the extended kalman filter
                       shape(num_states, 1)
        use_subject_average_fit: determines if the subject average fit is used in the 
                                 evaluation process of the measurement model
        use_least_squares_gf: determines if the least squares gf fit is used in the 
                                 evaluation process of the measurement model
        Returns
        measurement_value: Value of the derivative of the measurement function
                           shape(num_outputs, 1)
        """
        #Use numerical method to calculate the jacobean 
        result = self.numerical_jacobean(current_state)

        return result

    def numerical_jacobean(self, current_state:np.ndarray):

        """
        Numerical differentiation algorithm

        Keyword Arguments
        current_state: current state of the extended kalman filter
                       shape(num_states, 1)
        use_subject_average_fit: determines if the subject average fit is used in the 
                                 evaluation process of the measurement model
        use_least_squares_gf: determines if the least squares gf fit is used in the 
                                 evaluation process of the measurement model
        Returns
        measurement_value: Value of the derivative of the measurement function
                           shape(num_outputs, 1)
        """
        #Get the number of states
        num_states = current_state.shape[0]

        #Get the number of outputs
        num_outputs = self.personal_model.num_kmodels

        #Duplicate output if time derivative is enabled
        if (self.calculcate_output_derivative == True):
            num_outputs = num_outputs * 2

        #create buffer for the derivative
        manual_derivative = np.zeros((num_outputs,num_states))
        
        #We are going to fill it element-wise
        #col represents the current state
        f_state = self.evaluate_h_func(current_state)

        #Create buffer to store the result
        state_plus_delta = current_state.copy()


        for col in range(num_states):
            #Increment the state in the appropriate dimension
            state_plus_delta[col,0] += self.delta

            #Eliminate the previous increment
            if(col != 0):
                state_plus_delta[col-1,0] -= self.delta

            #Evaluate the model and extract it 
            f_delta = self.evaluate_h_func(state_plus_delta)

            #Get the derivative for the column
            manual_derivative[:,col] = ((f_delta-f_state)/(self.delta)).T
            
        return manual_derivative



    def output_time_derivative(self, current_state: np.ndarray,
                               eval_at_current_state: np.ndarray):
        """
        This function will get the time derivative of the output functions 

        Keyword Arguments: 
        current_state: Current state of the extended kalman filter
        It is assumed that phase is the first column and phase_dot is in the second
        column. 
        shape(num_states, 1)            

        Returns:
        measurement_value: Value of the derivative of the measurement function
                        shape(num_outputs, 1)
        """

        #Create an offset vector to calculate the derivative
        delta_vector = np.zeros(current_state.shape)
        delta_vector[0] += self.delta

        #Calculate new vector
        delta_state = current_state + delta_vector
        
        #Get the evaluation at the new phase
        eval_at_delta_state = self.personal_model.evaluate(delta_state.T)

        #Calculate the phase derivative
        phase_derivative = (eval_at_delta_state - eval_at_current_state)/self.delta

        #Get phase_dot 
        phase_dot = current_state[1]

        #Calculate the time derivative
        time_derivative = phase_derivative*phase_dot

        return time_derivative


            