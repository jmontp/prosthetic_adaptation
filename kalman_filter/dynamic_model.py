
import numpy as np
import matplotlib.pyplot as plt


class Gait_Dynamic_Model():

    def __init__(self):
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

        return jacobean

    #This is a linar function based on the jacobean its pretty straighforward 
    #to calculate
    def f_function(self, current_state, time_step):

        current_state[0,0] = current_state[0,0] + current_state[1,0]*time_step
        
        integer_part = np.floor(np.abs(current_state[0,0]))
        
        #Reset if you get over one
        if (current_state[0,0] >= 1.0):
            current_state[0,0] -= integer_part
        
        #This can occur with negative phase dot    
        if (current_state[0,0] < 0.0):
            current_state[0,0] += integer_part + 1
        
        return current_state





#This model test does not work anymore

def dynamic_model_unit_test():
    import pandas as pd
    #Set the integration timestep
    time_step = 0.1

    #Set an initial state
    initial_state_dict = {'phase': [0],
                    'phase_dot': [1],
                    'step_length': [1],
                    'ramp': [0],
                    'gf1': [0],
                    'gf2': [0],
                    'gf3': [0],
                    'gf4': [0]}
    
    initial_state = pd.DataFrame.from_dict(initial_state_dict)
    print(initial_state)
    gait_model = Gait_Dynamic_Model()

    state_history = initial_state.copy()
    
    current_state = initial_state.copy()

    #Iterate to see how the state evolves over time
    for i in range(100):
        current_state = gait_model.f_function(current_state,time_step)
        state_history = state_history.append(current_state, ignore_index=True)

    time = np.linspace(0,time_step*100,101)

    print(state_history)
    plt.plot(state_history['phase'])
    plt.show()


if __name__ == '__main__':
    dynamic_model_unit_test()
    
    
##TODO Log