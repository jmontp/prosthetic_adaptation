import numpy as np


class MeasurementModel():
    def __init__(self,state_names,models):
        self.models = models
        self.states = state_names
        self.len_models = len(models)
        self.len_states = len(state_names)
        self.num_states = len(state_names)
        self.num_outputs = len(models)
        print('states!' + str(state_names))
        
        
        
    def evaluate_h_func(self,current_state):
        #get the output
        result = [model.evaluate_gait_fingerprint_cross_model_numpy(current_state) for model in self.models]
        return np.array(result).reshape(self.num_outputs,-1)



    def evaluate_dh_func(self,current_state):
        # result = np.zeros((self.len_models,self.len_states))
        # for i,model in enumerate(self.models):
        #     state_derivatives = [model.evaluate_gait_fingerprint_numpy(current_state,self.partial_derivatives[state_name]) for state_name in self.states]
        #     result[i,:] = state_derivatives
        #return np.array(result).reshape(self.num_outputs,-1)
        
        result = self.numerical_jacobean(current_state)
        return result

    def numerical_jacobean(self, current_state):
        #create buffer for the derivative
        manual_derivative = np.zeros((self.len_models,self.len_states))
        
        #We are going to fill it element-wise
        #col represents the current state
        for col in range(self.len_states):
            state_plus_delta = current_state.copy()
            delta = 1e-6
            state_plus_delta[col,0] += delta
            
            #Evaluate the model and extract it 
            f_state = self.evaluate_h_func(current_state)
            f_delta = self.evaluate_h_func(state_plus_delta)

            manual_derivative[:,col] = ((f_delta-f_state)/(delta)).T
            
        return manual_derivative