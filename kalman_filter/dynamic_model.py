
#Idea: intialize state with dictionary, but then how to do covariance? double dictionary? That actually makes sense. 

#However that sounds like a pain in the ass to update. However, that can be done internally.

#I guess that the measurement function should also output expected measurements in a dictionary and have it compare to 
# dictionary inputs




#I think that everything should have the key-value pairs assigned to them

#I also think that I should take some time to design this right


#Large features to implement:
## Make everything dictionary based
## This makes sure that there is no fat fingering of inputs where they dont belong.
### Should be easy to implement with functions that convert 2d and 2d dicts and np arrays back and forth between them



import numpy as np
import matplotlib.pyplot as plt


class Dynamic_Model:

    #Initial States is a dictionary that will contain a variable name
    # for every state and the initial value for that state
    def __init__(self):
        pass

    def get_predicted_state(self,current_state,time_step,control_input_u):
        pass



class Gait_Dynamic_Model(Dynamic_Model):

    def __init__(self):
        pass

    #The gait dynamics are an integrator for phase that take into consideration 
    # the phase_dot and the timestep. It is assumed that the phase_dot is the 
    # second element in the state vector
    def f_jacobean(self, current_state, time_step):
        #print(current_state)
        amount_of_states = current_state.shape[0]
        #print(amount_of_states)

        #All the states will stay the same except for the first one
        jacobean = np.eye(amount_of_states)


        #Add the integrator corresponding to phase dot
        jacobean[0,1] = time_step

        return jacobean

    #This is a linar function based on the jacobean its pretty straighforward to calculate
    def f_function(self, current_state, time_step):
        
        #Essentially we want to calculate dot x = Ax
        #jacobean = self.f_jacobean(current_state,time_step)

        #current_state_vector = current_state.values.T
        
        #Copy the column name
        #result = current_state.copy


        #Set the result
        #result[0] = jacobean @ current_state_vector

        #compact implementation
        #result = current_state.copy()

        #result['phase'] = result['phase'] + result['phase_dot']*time_step
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