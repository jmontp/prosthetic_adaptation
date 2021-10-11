import matplotlib.pyplot as plt


################################################################################################
################################################################################################
#Unit testing

def ekf_unit_test():
    pass
    #%%
    #Phase, Phase Dot, Ramp, Step Length, 4 gait fingerprints
    initial_state_dict = {'phase': [0],
                    'phase_dot': [1],
                    'step_length': [1],
                    'ramp': [0],
                    'gf1': [0],
                    'gf2': [0],
                    'gf3': [0],
                    'gf4': [0],
                    'gf5': [0]}

    # initial_state = pd.DataFrame(initial_state_dict)
    #Phase, Phase, Dot, Step_length, ramp
    initial_state = np.array([[0.5,0.5,0.5,0.5,
                                0.5,0.5,0.5,0.5,0.5]]).T
    train_models = False
    if train_models == True:
        #Determine the phase models
        phase_model = Fourier_Basis(5,'phase')
        phase_dot_model = Polynomial_Basis(3,'phase_dot')
        step_length_model = Polynomial_Basis(3,'step_length')
        ramp_model = Polynomial_Basis(3,'ramp')
    
        # #Get the subjects
        subjects = [('AB10','../local-storage/test/dataport_flattened_partial_AB10.parquet')]
        for i in range(1,10):
            subjects.append(('AB0'+str(i),'../local-storage/test/dataport_flattened_partial_AB0'+str(i)+'.parquet'))

        model_foot = Kronecker_Model('jointangles_foot_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=4)
        model_saver(model_foot,'foot_model.pickle')
        
        model_shank = Kronecker_Model('jointangles_shank_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=4)
        model_saver(model_shank,'shank_model.pickle')
    
        model_foot_dot = copy.deepcopy(model_foot)
        model_foot_dot.time_derivative = True
        model_saver(model_foot_dot,'foot_dot_model.pickle')
        
        model_shank_dot = copy.deepcopy(model_shank)
        model_shank_dot.time_derivative = True
        model_saver(model_shank_dot,'shank_dot_model.pickle')
        
    else:
        model_foot = model_loader('foot_model.pickle')
        model_shank = model_loader('shank_model.pickle')
        model_foot_dot = model_loader('foot_dot_model.pickle')
        model_shank_dot = model_loader('shank_dot_model.pickle')

    models = [model_foot,model_shank,model_foot_dot,model_shank_dot]
    state_names = list(initial_state_dict.keys())

    measurement_model = Measurement_Model(state_names,models)

    num_states = len(state_names)
    num_outputs = len(models)
    initial_state_covariance = np.eye(num_states)*1e-7

    R = np.eye(num_outputs)
    
    Q = np.eye(num_states)
    
    Q[4,4] *= 2
    Q[5,5] *= 2
    Q[6,6] *= 2
    Q[7,7] *= 2

    d_model = Gait_Dynamic_Model()

    ekf = Extended_Kalman_Filter(initial_state,initial_state_covariance, d_model, Q, measurement_model, R)

    time_step = 0.001

    #Really just want to prove that we can do one interation of this
    #Dont really want to pove much more than this since we would need actual data for that
    
    control_input_u = 0 

    sensor_measurements = np.array([[1,1,1,1]]).T

    iterations = 100 
    state_history = np.zeros((iterations,len(initial_state)))

    try:
        for i in range(iterations):
            state_history[i,:] = ekf.calculate_next_estimates(time_step, control_input_u, sensor_measurements)[0].T
    except KeyboardInterrupt:
        pass
        
    print(state_history[:,0])
    plt.plot(state_history[:,:])
    plt.show()



def ekf_unit_test_simple_model():
    #%%
    #Mass spring system
    
    #state = [x,xdot]
    #state_dot = [xdot, xddot]
    
    #state_k+1 = R*state, R = rotation matrix with det 1 
    #(you are not adding or subtracting energy from the system)
    
    
    
    class SimpleDynamicModel():
        
        def __init__(self):
            #Rotation matrix to represent state dynamics
            self.R = lambda theta: np.array([[np.cos(theta),np.sin(theta)],
                                             [-np.sin(theta),np.cos(theta)]])
            #Rotational velocity in radians per sec
            self.omega = 2
        
        def f_jacobean(self, current_state, time_step):
            return self.R(self.omega*time_step)
            
        def f_function(self, current_state, time_step):
            return self.R(self.omega*time_step) @ current_state    
            
            
            
    class SimpleMeasurementModel():
        
        def evaluate_h_func(self,current_state):
            return np.eye(2) @ current_state
        
        def evaluate_dh_func(self,current_state):
            return np.eye(2)    

    #Setup simulation
    initial_state = np.array([[0,1]]).T
    initial_state_covariance = np.eye(2)*1e-7
    
    d_model = SimpleDynamicModel()
    measurement_model = SimpleMeasurementModel()
    
    #Sensor noise    
    R = np.eye(2)
    
    #Process noise
    Q = np.eye(2)*1e-2
    
    ekf = Extended_Kalman_Filter(initial_state,initial_state_covariance, d_model, Q, measurement_model, R)


    actual_state = np.array([[1,0]]).T
    
    #Setup time shenanigans
    iterations = 1001
    total_time = 10    
    iterator = np.linspace(0,total_time,iterations)
    time_step = iterator[1] - iterator[0]

    #Setup state history tracking
    state_history = np.zeros((iterations,2*len(initial_state)))

    for i,t in enumerate(iterator):
        
        actual_state = d_model.f_function(actual_state, time_step)
        
        predicted_state,_ = ekf.calculate_next_estimates(time_step, 0, actual_state)
        
        state_history[i,:2] = predicted_state.T
        state_history[i,2:] = actual_state.T
        
    #%matplotlib qt
    plt.plot(state_history)
    plt.legend(["Predicted Position", "Predicted Velocity", 
                "Actual Position", "Actual Velocity"])
    plt.show()
    #%%


def profiling():
    pass
#%%
    import pstats
    from pstats import SortKey
    p = pstats.Stats('profile.txt')
    p.sort_stats(SortKey.CUMULATIVE).print_stats(10)


#%%
if(__name__=='__main__'):
    #ekf_unit_test()
    ekf_unit_test_simple_model()