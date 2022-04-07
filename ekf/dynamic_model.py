import numpy as np


class GaitDynamicModel():

    def __init__(self):

        #Small increase in phase dot every time step
        self.phase_dot_scale = 1
        pass

    #The gait dynamics are an integrator for phase that take into consideration 
    # the phase_dot and the timestep. It is assumed that the phase_dot is the 
    # second element in the state vector
    def f_jacobean(self, current_state, time_step):
        
        amount_of_states = current_state.shape[0]

        #All the states will stay the same except for the first one
        jacobean = np.eye(amount_of_states)

        #Add the integrator corresponding to phase dot
        jacobean[0,1] = time_step
        jacobean[1,1] = self.phase_dot_scale
        return jacobean

    #This is a linar function based on the jacobean its pretty straighforward 
    #to calculate
    def f_function(self, current_state, time_step):

        current_state[0,0] = current_state[0,0] + current_state[1,0]*time_step
        
        integer_part = np.floor(np.abs(current_state[0,0]))
        
        #Reset if you get over one
        if (current_state[0,0] >= 1.0):
            current_state[0,0] -= integer_part
        
        #Prevent negative phase_dot
        #This can occur with negative phase dot    
        if (current_state[0,0] < 0.0):
            current_state[0,0] += integer_part + 1

        #Small phase_dot increase per state
        current_state[1,0] = current_state[1,0]*self.phase_dot_scale
        
        return current_state