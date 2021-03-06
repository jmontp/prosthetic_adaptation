import numpy as np

##LOOK HERE 
##There is a big mess with how the measurement model is storing the gait fingerprint coefficients
##They should really just be part of the state vector, the AXIS should be stored internally since that is 
## fixed


#Measurement Model = []



class Measurement_Model():
    def __init__(self,state_names,models):
        self.models = models
        self.states = state_names
        self.len_models = len(models)
        self.len_states = len(state_names)
        self.num_states = len(state_names)
        self.num_outputs = len(models)
        print('states!' + str(state_names))
        self.partial_derivatives = {state : models[0].get_partial_derivatives([state]) for state in state_names}
        
        
        
    def evaluate_h_func(self,current_state):
        #get the output
        #print("Measurement Model: Current State" + str(current_state))
        result = [model.evaluate_gait_fingerprint_cross_model_numpy(current_state) for model in self.models]
        return np.array(result).reshape(self.num_outputs,-1)



    def evaluate_dh_func(self,current_state):
        # result = np.zeros((self.len_models,self.len_states))
        # for i,model in enumerate(self.models):
        #     state_derivatives = [model.evaluate_gait_fingerprint_numpy(current_state,self.partial_derivatives[state_name]) for state_name in self.states]
        #     result[i,:] = state_derivatives
        result = self.numerical_jacobean(current_state)
        return result
        #return np.array(result).reshape(self.num_outputs,-1)


    def numerical_jacobean(self, current_state):
        #create buffer for the derivative
        manual_derivative = np.zeros((self.len_models,self.len_states))
        #We are going to fill it element-wise
        #col represents the current state
        for col in range(self.len_states):
            state_plus_delta = current_state.copy()
            delta = 1e-4
            state_plus_delta[col,0] += delta
            #print("State" + str(state))
            #print("State +" + str(state_plus_delta))
            #Evaluate the model and extract it 
            f_state = self.evaluate_h_func(current_state)
            f_delta = self.evaluate_h_func(state_plus_delta)

            manual_derivative[:,col] = ((f_delta-f_state)/(delta)).T
            
        return manual_derivative

def unit_test():
    pass
    #%%
    import os,sys
    PACKAGE_PARENT = '../model_fitting/'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    new_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
    print(new_path)
    sys.path.insert(0,new_path)
        
    from function_bases import Polynomial_Basis, Fourier_Basis 
    from kronecker_model import Kronecker_Model, model_saver, model_loader
    
    
    train_models = False
    if train_models == True:
        #Determine the phase models
        phase_model = Fourier_Basis(10,'phase')
        phase_dot_model = Polynomial_Basis(1,'phase_dot')
        step_length_model = Polynomial_Basis(3,'step_length')
        ramp_model = Polynomial_Basis(4,'ramp')
    
        # #Get the subjects
        subjects = [('AB10','../local-storage/test/dataport_flattened_partial_AB10.parquet')]
        for i in range(1,10):
            subjects.append(('AB0'+str(i),'../local-storage/test/dataport_flattened_partial_AB0'+str(i)+'.parquet'))

        model_foot = Kronecker_Model('jointangles_foot_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=5)
        model_saver(model_foot,'foot_model.pickle')
        
        model_shank = Kronecker_Model('jointangles_shank_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=5)
        model_saver(model_shank,'shank_model.pickle')
    
        model_foot_dot = Kronecker_Model('jointangles_foot_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=5,time_derivative=True)
        model_saver(model_foot_dot,'foot_dot_model.pickle')
        
        model_shank_dot = Kronecker_Model('jointangles_shank_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=5,time_derivative=True)
        model_saver(model_shank_dot,'shank_dot_model.pickle')
        
    else:
        model_foot = model_loader('../model_fitting/foot_model.pickle')
        model_shank = model_loader('../model_fitting/shank_model.pickle')
        model_foot_dot = model_loader('../model_fitting/foot_dot_model.pickle')
        model_shank_dot = model_loader('../model_fitting/shank_dot_model.pickle')
        
    
    initial_state_dict = {'phase': [0],
                    'phase_dot': [1],
                    'step_length': [1],
                    'ramp': [0],
                    'gf1': [0],
                    'gf2': [0],
                    'gf3': [0],
                    'gf4': [0],
                    'gf5':[0]}

    models = [model_foot,model_shank,model_foot_dot,model_shank_dot]
    state_names = list(initial_state_dict.keys())
    
    num_states = len(state_names)
    num_models = len(models)
    measurement_model = Measurement_Model(state_names,models)
    
    state = (np.array([[1.0,1.0,1.0,1.0,
                        1.0,1.0,1.0,1.0,1.0]]).T)*0.5
    
    manual_derivative = np.zeros((num_models,num_states))
    for row in range(num_models):
        for col in range(num_states):
            
            state_plus_delta = state.copy()
            delta = 1e-14
            state_plus_delta[col,0] += delta
            #print("State" + str(state))
            #print("State +" + str(state_plus_delta))
            f_state = measurement_model.evaluate_h_func(state)[row]
            f_delta = measurement_model.evaluate_h_func(state_plus_delta)[row]

            manual_derivative[row,col] = (f_delta-f_state)/(delta)


    
    print("Manual" + str(manual_derivative))
    print("Rank manual rank: {}".format(np.linalg.matrix_rank(manual_derivative)))

    expected_derivative = measurement_model.evaluate_dh_func(state)
    
    print("Expected" + str(expected_derivative))
    print("Rank expected rank: {}".format(np.linalg.matrix_rank(expected_derivative)))
    
    
    print("Difference" + str(manual_derivative - expected_derivative))
    
    print("Norm of expected - manual: {}".format(np.linalg.norm(expected_derivative-manual_derivative)))
    
    #%%
if __name__=='__main__':
    unit_test()