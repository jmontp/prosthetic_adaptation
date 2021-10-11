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