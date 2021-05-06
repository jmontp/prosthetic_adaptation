#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 22:15:17 2021

@author: jmontp
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
#Model Object:
# list of basis objects
# string description
# model_size
# pca_axis - list of np array of model_size length
# coefficient_list - list of coefficients for each pca_axis
#--------------------------
class Kronecker_Model:
    def __init__(self, output_name,*funcs,time_derivative=False,personalization_map=None,subjects=None,num_gait_fingerprint=5):
        self.funcs = funcs

        #Calculate the size of the parameter array
        #Additionally, pre-allocate arrays for kronecker products intermediaries 
        # to speed up results
        self.output_name = output_name
        self.alocation_buff = []
        self.order = []
        size = 1
        for func in funcs:
            #Since we multiply left to right, the total size will be on the left 
            #and the size for the new row will be on the right
            print((str(size), str(func.size)))
            self.alocation_buff.append(np.zeros((size, func.size)))

            size = size * func.size

            self.order.append(func.var_name)


        self.size = size
        self.num_states = len(funcs)
        self.subjects = {}
        self.one_left_out_subjects = {}
        self.time_derivative = time_derivative
        self.num_gait_fingerprint = num_gait_fingerprint
        self.gait_fingerprint_names = ["gf"+str(i) for i in range(1,num_gait_fingerprint+1)]
        self.no_derivatives = {val:0 for val in (self.order+self.gait_fingerprint_names)}
        print(self.no_derivatives)
        #Todo: Add average pca coefficient
        
        if(personalization_map == None and subjects is not None):
            self.add_subject(subjects)
            self.fit_subjects()
            self.calculate_gait_fingerprint(n=num_gait_fingerprint)
        
    
    def add_subject(self,subjects):
        for subject,filename in subjects:
            self.subjects[subject] = \
                {'filename': filename, \
                 'dataframe': pd.read_parquet(filename, columns=[self.output_name,*self.order]), \
                 'optimal_xi': [], \
                 'least_squares_info': [], \
                 'pca_axis': [], \
                 'pca_coefficients': [] \
             }

    # #Evaluate the models at the function inputs that are received
    # #The function inputs are expected in the same order as they where defined
    # #Alternatively, you can also input a dictionary with the var_name as the key and the 
    # # value you want to evaluate the function as the value
    # def evaluate(self, *function_inputs,partial_derivative=None, result_buffer=None):
        
    #     result = np.array([1])
        
    #     for i in range(self.num_states):

    #         #Verify if we want to take the partial derivative of this function
    #         if(partial_derivative is not None and curr_func.var_name in partial_derivative):
    #             apply_derivative = True
    #         else: 
    #             apply_derivative = False


    #         result=fast_kronecker(result,self.funcs[i].evaluate_conditional(function_inputs[i],False))#, self.alocation_buff[i], False)
            
    #     return result.copy()

    def evaluate_pandas(self, dataframe, partial_derivatives=None):
        #Todo: Implement partial derivatives
        #Can be done by creating a new list of functions and adding partial derivatives when needed
        # if (self.time_derivative == True):
        #     if partial_derivatives is not None:
        #         partial_derivatives.append('time')
        #         partial_derivatives.append('phase')
        #     else:
        #         partial_derivatives = ['time','phase']
                
        return pandas_kronecker(dataframe,self.funcs)#,partial_derivatives)
    
    
    def evaluate_subject_optimal_pandas(self,subject, dataframe):
        regressor = self.evaluate_pandas(dataframe)
        print(regressor.shape)
        xi = self.subjects[subject]['optimal_xi']
        print(regressor.shape)
        return regressor @ xi
    
    def evaluate_subject_optimal_numpy(self,subject, np_array):
        regressor = self.evaluate_numpy(np_array)
        xi = self.subjects[subject]['optimal_xi']
        return regressor @ xi
    
    def evaluate_gaint_fingerprint(self,dataframe,partial_derivatives=None):
            
        #Get the row function with its corresponding partial derivatives
        row_function = self.evaluate_pandas(dataframe,partial_derivatives)
        
        #Get the gait fingerprints and then sort them
        gait_fingerprint = dataframe[self.gait_fingerprint_names]
        #gait_fingerprint = gait_fingerprint.reindex(sorted(gait_fingerprint.columns), axis=1)
        
        if partial_derivatives is not None:
            for element in partial_derivatives:
                if element in gait_fingerprint.columns:
                    gait_fingerprint[element] = 1
        
        gait_finterprint_vector = gait_fingerprint.values.T
        
        xi = self.personalization_map_scaled @ gait_finterprint_vector
        
        return row_function @ xi

    def evaluate_numpy(self,states,partial_derivatives=None):
        rows = states.shape[1]
        #Copy the dict so that we dont modify the global one
        if(partial_derivatives == None):
            local_partial_derivatives = dict(self.no_derivatives)
        else:
            local_partial_derivatives = dict(partial_derivatives)
        
        #Make sure we only have the amount of states that we want
        phase_states = states[:self.num_gait_fingerprint,:]
        
        partial_phase_dot = False
        if self.time_derivative == True:
            if (local_partial_derivatives['phase_dot'] > 0):
                partial_phase_dot = True

            local_partial_derivatives['phase']+=1
            #print(local_partial_derivatives)

        output = np.array(1).reshape(1,1,1)

        for func,state in zip(self.funcs,phase_states):
            state_t = (state.T)[:,np.newaxis]
            eval_value = func.evaluate_derivative(state_t,local_partial_derivatives[func.var_name]).reshape(rows,-1,1)
            output = (output[:,np.newaxis,:]*eval_value).reshape(rows,-1)
            #print("I'm alive, size = " + str(output.shape))
    
        if self.time_derivative == True:
            #print("Phase dot: " + str(phase_states[1]))
            #print(phase_states[1,:].reshape(-1,1))
            output = phase_states[1,:].reshape(-1,1)*output
            
            if(partial_phase_dot == True):
                local_partial_derivatives['phase_dot'] -= 1
                
                assert local_partial_derivatives['phase_dot'] >= 0
                assert local_partial_derivatives['phase'] > 0
                
                added_output = np.array(1).reshape(1,1,1)
                for func,state in zip(self.funcs,phase_states):
                    state_t = (state.T)[:,np.newaxis]
                    eval_value = func.evaluate_derivative(state_t,local_partial_derivatives[func.var_name]).reshape(rows,-1,1)
                    added_output = (added_output[:,np.newaxis,:]*eval_value).reshape(rows,-1)
                output += added_output
                
        return output

    def get_partial_derivatives(self,state_names):
        output = {}
        for key in self.no_derivatives:
            output[key] = 0

        for name in state_names:
            output[name] += 1

        return output

    def evaluate_gait_fingerprint_numpy(self,states,partial_derivatives=None):
        
        phase_states =  states[:self.num_states].copy()
        gait_fingerprints = states[self.num_states:].copy()
        
        derive = False


        if(partial_derivatives is None):
            local_partial_derivatives = self.no_derivatives
        else:
            local_partial_derivatives = partial_derivatives.copy()
            for name in self.gait_fingerprint_names:
                if local_partial_derivatives[name] > 0:
                    if(derive==True):
                        gait_fingerprints[int(name[-1])-1] = 0
                        break
                    else:
                        for name_2 in self.gait_fingerprint_names:
                            gait_fingerprints[int(name_2[-1])-1] = 0
                        gait_fingerprints[int(name[-1])-1] = 1
                        derive = True

        #print(self.inter_subject_average_fit.shape)
        #print((self.personalization_map @ gait_fingerprints).shape)
        if(derive == False):
            xi = (self.personalization_map @ gait_fingerprints) + self.inter_subject_average_fit
        else:
            xi = (self.personalization_map @ gait_fingerprints)
            
        row_vector = self.evaluate_numpy(phase_states,partial_derivatives)

        return row_vector @ xi


    
    #Future optimizations
    #@numba.jit(nopython=True, parallel=True)
    def least_squares(self,dataframe,output,splits=50):

        RTR = np.zeros((self.size,self.size))
        yTR = np.zeros((1,self.size))
        RTy = np.zeros((self.size,1))
        yTy = 0
        num_rows = len(dataframe.index)
        
        for sub_dataframe in np.array_split(dataframe,splits):
            R = self.evaluate_pandas(sub_dataframe)
            #nans = sub_dataframe[output].isnull().sum()
            #print(nans)
            #print(sub_dataframe.shape[0])
            y = sub_dataframe[output].values[:,np.newaxis]
            RTR_ = R.T @ R
        
            print("RTR rank: {} RTR shape {}".format(np.linalg.matrix_rank(RTR,hermitian=True),RTR.shape))
            
            RTR += RTR_
            yTR += y.T @ R
            RTy += R.T @ y
            yTy += y.T @ y
        
        # try:

        #     assert (np.linalg.norm(RTR-RTR.T) < 1e-7)
            
        #     eval_, evec_ = np.linalg.eigh(RTR)
        #     for e in eval_:
        #         #print("Eigenvalue is {}".format(e))
        #         assert(e >= 1e-2)
            
        
        # except AssertionError:
        #     print("Assertion Error on RTR \n {}".format(RTR))
        #     print("R = {}".format(R))

        #     raise AssertionError
        
        return np.linalg.solve(RTR, RTy), num_rows, RTR, RTy, yTR, yTy
       


    def fit_subjects(self):
        
        for name,subject_dict in self.subjects.items():
            print("Doing " + name)
            print(subject_dict['filename'])
            data = subject_dict['dataframe']
            output = self.least_squares(data,self.output_name)
            subject_dict['optimal_xi'] = output[0]
            subject_dict['num_rows'] = output[1]
            subject_dict['least_squares_info'] = output[2:]
        
    def scaled_pca(self):
        
        num_subjects = len(self.subjects)
        Ξ = np.array([subject['optimal_xi'] for subject in self.subjects.values()]).reshape(num_subjects,-1)
        G_total = 0
        N_total = 0
        
        for name,subject_dict in self.subjects.items():
            G_total += subject_dict['least_squares_info'][0]
            N_total += subject_dict['num_rows']
	
        #This is equation eq:inner_regressor in the paper!
       	G = G_total/N_total
        #G = G_total/1


        #Verify we are positive semidefinite
        assert(np.linalg.norm(G-G.T)<1e-7)

        #Diagonalize the matrix G as G = OVO
        eig, O = np.linalg.eigh(G)
        V = np.diagflat(eig)
        #print("Gramian {}".format(G))
        #Additionally, all the eigenvalues are true
        for e in eig:
            #print("Eigenvalue: {}".format(e))
            assert (e >= 0)
            assert( e > 0) # pd

        # Verify that it diagonalized correctly G = O (eig) O.T
        assert(np.linalg.norm(G - O @ V @ O.T)< 1e-7 * np.linalg.norm(G)) # passes

        #This is based on the equation in eq:Qdef
        # Q G Q = I
        Q       = sum([O[:,[i]] @ O[:,[i]].T * 1/np.sqrt(eig[i]) for i in range(len(eig))])
        Qinv    = sum([O[:,[i]] @ O[:,[i]].T * np.sqrt(eig[i]) for i in range(len(eig))])

        #Change of basis conversions
        def param_to_orthonormal(ξ):
            return Qinv @ ξ
        def param_from_orthonormal(ξ):
            return Q @ ξ
        def matrix_to_orthonormal(Ξ):
            return Ξ @ Qinv

        #Get the average coefficients
        ξ_avg = np.mean(Ξ, axis=0)
        
        #Save the intersubject average model
        self.inter_subject_average_fit = ξ_avg[:,np.newaxis]
        
        #Substract the average coefficients
        Ξ0 = Ξ - ξ_avg

        ##Todo: The pca axis can also be obtained with pca instead of eigenvalue 
        ## decomposition
        #Calculate the coefficients in the orthonormal space
        Ξ0prime = matrix_to_orthonormal(Ξ0)

        #Get the covariance matrix for this
        Σ = Ξ0prime.T @ Ξ0prime / (Ξ0prime.shape[0]-1)

        #Calculate the eigendecomposition of the covariance matrix
        ψinverted, Uinverted = np.linalg.eigh(Σ)

        #Eigenvalues are obtained from smalles to bigger, make it bigger to smaller
        ψs = np.flip(ψinverted)
        self.scaled_pca_eigenvalues = ψs
        Ψ = np.diagflat(ψs)

        #If we change the eigenvalues we also need to change the eigenvectors
        U = np.flip(Uinverted, axis=1)

        #Run tests to make sure that this is working
        assert(np.linalg.norm(Σ - U @ Ψ @ U.T)< 1e-7 * np.linalg.norm(Σ)) # passes
        for i in range(len(ψs)-1):
            assert(ψs[i] > ψs[i+1])

        #Define the amount principles axis that we want
        #η = num_gait_fingerprints
        η=len(self.subjects)#self.num_gait_fingerprint
        pca_axis_array = []

        #Convert from the new basis back to the original basis vectors
        for i in range (0,η):
            pca_axis_array.append(param_from_orthonormal(U[:,i]*np.sqrt(ψs[i])))
            
        self.scaled_pca_components = np.array(pca_axis_array).T
        #Return the personalization map
        return self.scaled_pca_components[:,:self.num_gait_fingerprint]


    def estimate_xi(self,gait_finterprint_vector):
        return self.personalization_map_scaled*gait_finterprint_vector

    def calculate_gait_fingerprint(self,n=None,cum_var=0.95):
        num_subjects = len(self.subjects.values())
        #Get all the gait fingerprints into a matrix
        XI = np.array([subject['optimal_xi'] for subject in self.subjects.values()]).reshape(num_subjects,-1)
        XI_mean = XI.mean(axis=0).reshape(1,-1)
        XI_0 = XI - XI_mean
        pca = PCA(n_components=num_subjects)
        pca.fit(XI_0) 
        self.pca_result = pca
        
        self.personalization_map = (pca.components_[:n,:]).T
        self.personalization_map_scaled = self.scaled_pca()
        
        #Calculate gait fingerprints for every individual
        for subject_dict in self.subjects.values():
            RTR, RTy, yTR, yTy = subject_dict['least_squares_info']
            RTR_prime = (self.personalization_map_scaled.T) @ RTR @ self.personalization_map_scaled
            RTy_prime = (self.personalization_map_scaled.T) @ RTy-(self.personalization_map_scaled.T) @ RTR @ self.inter_subject_average_fit
            
            subject_dict['gait_coefficients'] = np.linalg.solve(RTR_prime,RTy_prime)
            
            RTR, RTy, yTR, yTy = subject_dict['least_squares_info']
            RTR_prime = (self.personalization_map.T) @ RTR @ self.personalization_map
            RTy_prime = (self.personalization_map.T) @ RTy-(self.personalization_map.T) @ RTR @ self.inter_subject_average_fit
            
            subject_dict['gait_coefficients_unscaled'] = np.linalg.solve(RTR_prime,RTy_prime)

            
    def add_left_out_subject(self,subjects):
        for subject,filename in subjects:
            self.one_left_out_subjects[subject] = \
                {'filename': filename, \
                 'dataframe': pd.read_parquet(filename, columns=[self.output_name,*self.order]), \
                 'optimal_xi': [], \
                 'least_squares_info': [], \
                 'pca_axis': [], \
                 'pca_coefficients': [] \
             }
                    
            subject_dict = self.one_left_out_subjects[subject]        
            print("One left out fit: " + subject)
            data = subject_dict['dataframe']
            
            output = self.least_squares(data,self.output_name)
            subject_dict['optimal_xi'] = output[0]
            subject_dict['num_rows'] = output[1]
            subject_dict['least_squares_info'] = output[2:]

            RTR, RTy, yTR, yTy = subject_dict['least_squares_info']
            RTR_prime = (self.personalization_map_scaled.T) @ RTR @ self.personalization_map_scaled
            RTy_prime = (self.personalization_map_scaled.T) @ RTy-(self.personalization_map_scaled.T) @ RTR @ self.inter_subject_average_fit
            
            subject_dict['gait_coefficients'] = np.linalg.solve(RTR_prime,RTy_prime)
            
            RTR, RTy, yTR, yTy = subject_dict['least_squares_info']
            RTR_prime = (self.personalization_map.T) @ RTR @ self.personalization_map
            RTy_prime = (self.personalization_map.T) @ RTy-(self.personalization_map.T) @ RTR @ self.inter_subject_average_fit
            
            subject_dict['gait_coefficients_unscaled'] = np.linalg.solve(RTR_prime,RTy_prime)



    def __str__(self):
        output = ''
        for func in self.funcs:
            func_type = type(func).__name__
            if(func_type == 'Polynomial_Basis'):
                basis_identifier = 'P'
            elif (func_type == 'Fourier_Basis'):
                basis_identifier = 'F'
            elif (func_type == 'Bernstein_Basis'):
                basis_identifier = 'B'
            else:
                raise TypeError("This is not a basis")

            output += func.var_name + '-' + str(func.n)+ basis_identifier + '--'
        
        return output

    def get_order(self):
        return self.order


#@vectorize(nopython=True)
def pandas_kronecker(dataframe,funcs,partial_derivatives=None):
    rows = dataframe.shape[0]
    output = np.array(1).reshape(1,1,1)
        
    for func in funcs:
        apply_derivative = False
        num_derivatives = 0
        
        if partial_derivatives is not None and func.var_name in partial_derivatives:
            apply_derivative = True
            num_derivatives = partial_derivatives.count(func.var_name)
            
        output = (output[:,np.newaxis,:]*func.evaluate_conditional(dataframe[func.var_name].values[:,np.newaxis],apply_derivative,num_derivatives)[:,:,np.newaxis]).reshape(rows,-1)
        #print("I'm alive, size = " + str(output.shape))
    
    if partial_derivatives is not None and 'time' in partial_derivatives:
        output = (dataframe['phase_dot'].values)[:,np.newaxis]*output
    
    
    return output


#Save the model so that you can use them later
def model_saver(model,filename):
    with open(filename,'wb') as file:
        pickle.dump(model,file)

#Load the model from a file
def model_loader(filename):
    with open(filename,'rb') as file:
        return pickle.load(file)
####################################################################################
#//////////////////////////////////////////////////////////////////////////////////#
####################################################################################


def train_models():
    pass
    #%%
    from function_bases import Fourier_Basis, Polynomial_Basis
    import copy
    
    train_models = True
    if train_models == True:
        #Determine the phase models
        phase_model = Fourier_Basis(8,'phase')
        phase_dot_model = Polynomial_Basis(1,'phase_dot')
        step_length_model = Polynomial_Basis(2,'step_length')
        ramp_model = Polynomial_Basis(3,'ramp')
    
        # #Get the subjects
        left_out = [('AB10','../local-storage/test/dataport_flattened_partial_AB10.parquet')]
        subjects = []
        for i in range(1,10):
            subjects.append(('AB0'+str(i),'../local-storage/test/dataport_flattened_partial_AB0'+str(i)+'.parquet'))

        model_foot = Kronecker_Model('jointangles_foot_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=5)
        model_foot.add_left_out_subject(left_out)
        model_saver(model_foot,'foot_model.pickle')
        
        model_shank = Kronecker_Model('jointangles_shank_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=5)
        model_shank.add_left_out_subject(left_out)
        model_saver(model_shank,'shank_model.pickle')
    
        model_foot_dot = copy.deepcopy(model_foot)
        model_foot_dot.time_derivative = True
        model_saver(model_foot_dot,'foot_dot_model.pickle')
        
        model_shank_dot = copy.deepcopy(model_shank)
        model_shank_dot.time_derivative = True
        model_saver(model_shank_dot,'shank_dot_model.pickle')
        
 #%%
def get_mean_std_dev(np_array):
        point_per_step = 150

        if (type(np_array) == np.ndarray):
            steps = int(np_array.shape[0]/point_per_step)
            mean = np.mean(np_array.reshape(steps,-1),axis=0)
            std_dev = np.std(np_array.reshape(steps,-1),axis=0)
        else:
            steps = int(np_array.shape[0]/point_per_step)
            mean = np.mean(np_array.values.reshape(steps,-1),axis=0)
            std_dev = np.std(np_array.values.reshape(steps,-1),axis=0)
        
        return mean, std_dev
    
    
def get_rmse(arr1,arr2):
    if (type(arr1) == pd.core.series.Series):
        arr1 = arr1.values[:,np.newaxis]
    
    if (type(arr2) == pd.core.series.Series):
        arr2 = arr2.values[:,np.newaxis]
        
    return np.sqrt(np.mean(np.power((arr1-arr2),2)))

def trial_to_string(trial,joint=None):
    sign = 1
    split = trial.split('i')
    if(len(split)==1):
        sign = -1
        split = trial.split('d')

    incline = sign*float(split[1].replace('x','.'))
    speed = float(split[0][1:].replace('x','.'))
    
    if(joint is not None):
        output = "{} Angle while walking at {:.1f}m/s with {:.1f}-deg incline".format(joint,speed,incline)
    else:
        output = "Walking at {:.1f}m/s with {:.1f}-deg incline".format(speed,incline)
    return output
#%%
def plot_model():
    #%%
    #%matplotlib qt

    model_foot = model_loader('foot_model.pickle')
    model_shank = model_loader('shank_model.pickle')
    
    #Variables
    ##########################################################################
    subject = 'AB01'
    model = model_foot
    joint_angle = 'jointangles_foot_x'
    joint_str = joint_angle.split('_')[1].capitalize()
    #trials = ['s0x8i0','s0x8i10','s1x2i0']
    trials = ['s0x8i0']
    #trials = list(df.trial.unique())[:]
    mean_plots = False
    ##########################################################################
    
    try:
        subject_dict = model.subjects[subject]
    except KeyError:
        print("Not in subject dict, checking left out dict")
        subject_dict = model.one_left_out_subjects[subject]
        print("In left out dict")

    
    #Get constants
    inter_subject_average_fit = model.inter_subject_average_fit
    personalization_map_scaled = model.personalization_map_scaled
    bad_personalization_map_scaled = model.personalization_map
    gait_fingerprints = subject_dict['gait_coefficients']
    bad_gait_fingerprints = subject_dict['gait_coefficients_unscaled']
    optimal_fit = subject_dict['optimal_xi']
    
    df = pd.read_parquet(subject_dict['filename'])
    
    
    
    points_per_step = 150
    x = np.linspace(0,1+1/points_per_step,points_per_step)
    
    fig, ax = plt.subplots(1,len(trials), sharex='all',sharey ='all')
    
    if (len(trials)==1):
        ax = [ax]
    
    for i,trial in enumerate(trials):
        #Level ground walking
        #Get measured data
        trial_df = df[df['trial'] == trial]
        
        #Uncomment to get rmse for all trials
        #trial_df = df
        measured_angles = trial_df[joint_angle]
        foot_mean, foot_std_dev= get_mean_std_dev(measured_angles)
        
        #Get regressor rows
        if(mean_plots == True):
            foot_angle_evaluated = model.evaluate_pandas(trial_df)
        else:
            measured_angles_total = measured_angles
            measured_angles = measured_angles[:150]
            foot_angle_evaluated = model.evaluate_pandas(trial_df)[:150]
        
        
        print(trial)
        #Optimal fit
        optimal_estimate = foot_angle_evaluated @ optimal_fit
        optimal_mean, optimal_std_dev = get_mean_std_dev(optimal_estimate)
        optimal_rmse = get_rmse(optimal_estimate,measured_angles)
        print("Optimal rmse {}".format(optimal_rmse))

        #Intersubject fit
        inter_subject_estimate = foot_angle_evaluated @ inter_subject_average_fit
        inter_subject_mean,  inter_subject_std_dev = get_mean_std_dev(inter_subject_estimate)
        inter_subject_rmse = get_rmse(inter_subject_estimate,measured_angles)
        print("Inter subject average rmse {}".format(inter_subject_rmse))

        
        #Gait fingerprint fit
        gait_fingerprint_estimate = foot_angle_evaluated @ (inter_subject_average_fit + personalization_map_scaled @ gait_fingerprints)
        gait_fingerprint_mean, gait_fingerprint_std_dev = get_mean_std_dev(gait_fingerprint_estimate)
        gait_fingerprint_rmse = get_rmse(gait_fingerprint_estimate,measured_angles)
        print("Gait fingerprint rmse {}".format(gait_fingerprint_rmse))

        
        #Bad gait fingerprint fit
        bad_gait_fingerprint_estimate = foot_angle_evaluated @ (inter_subject_average_fit + bad_personalization_map_scaled @ bad_gait_fingerprints)
        bad_gait_fingerprint_mean, bad_gait_fingerprint_std_dev = get_mean_std_dev(bad_gait_fingerprint_estimate)
        bad_gait_fingerprint_rmse = get_rmse(bad_gait_fingerprint_estimate,measured_angles)
        print("Bad gait fingerprint rmse {}".format(bad_gait_fingerprint_rmse))
        
        
        clrs = cm.get_cmap('tab20').colors
            
        if(mean_plots == True):
            #Measured
            #Mean plots with width 
            ax[i].plot(x, foot_mean,label='Measured Foot Angle', c=clrs[0], linestyle = 'solid')
            ax[i].fill_between(x, foot_mean-foot_std_dev, foot_mean+foot_std_dev ,alpha=0.3, facecolor=clrs[0])
            #Optimal
            ax[i].plot(x, optimal_mean,label='Optimal Fit RMSE:{:.2f}'.format(optimal_rmse), c=clrs[1])
            ax[i].fill_between(x, optimal_mean-optimal_std_dev, optimal_mean+optimal_std_dev ,alpha=0.3, facecolor=clrs[1])
            #Inter subject average
            ax[i].plot(x, inter_subject_mean,label='Inter-Subject Averate Fit RMSE:{:.2f}'.format(inter_subject_rmse), c=clrs[2])
            ax[i].fill_between(x, inter_subject_mean-inter_subject_std_dev, inter_subject_mean+inter_subject_std_dev ,alpha=0.3, facecolor=clrs[2])
            #Gait fingerprint
            ax[i].plot(x, gait_fingerprint_mean,label='Gait Fingerprint Fit RMSE:{:.2f}'.format(gait_fingerprint_rmse), c=clrs[3])
            ax[i].fill_between(x, gait_fingerprint_mean-gait_fingerprint_std_dev, gait_fingerprint_mean+gait_fingerprint_std_dev ,alpha=0.3, facecolor=clrs[3])
            #Bad Gait fingerprint
            ax[i].plot(x, bad_gait_fingerprint_mean,label='Bad Gait Fingerprint Fit RMSE:{:.2f}'.format(bad_gait_fingerprint_rmse), c=clrs[4])
            ax[i].fill_between(x, bad_gait_fingerprint_mean-bad_gait_fingerprint_std_dev, bad_gait_fingerprint_mean+bad_gait_fingerprint_std_dev ,alpha=0.3, facecolor=clrs[4])
        
        else:
            
            line_width = 6
            # Individual line plots
            step_data = measured_angles_total.values.reshape(-1,150)
            for k in range (0,step_data.shape[0],3):
                if (150 - np.count_nonzero(step_data[k,:]) > 20):
                    continue
                if k == 0:
                    ax[i].plot(x, step_data[k,:],label='Measured Foot Angle', linestyle = 'solid', alpha=0.2, linewidth=5, c='darkgrey')
                else:
                    ax[i].plot(x, step_data[k,:], linestyle = 'solid', alpha=0.3, linewidth=5, c='darkgrey')
            
            #Inter subject average
            ax[i].plot(x, inter_subject_estimate,label='Inter-Subject Averate Fit RMSE:{:.2f}'.format(inter_subject_rmse),
                       linewidth=line_width, c=clrs[6])#, linestyle=(0, (1, 1)), alpha=0.8) #Densely dotted
           
            #Bad Gait fingerprint
            # ax[i].plot(x, bad_gait_fingerprint_estimate,label='Bad Gait Fingerprint Fit RMSE:{:.2f}'.format(bad_gait_fingerprint_rmse),
            #            linewidth=line_width,c=clrs[4],linestyle=(0,(6,1,1,1)), alpha=0.8)
            
            #Optimal fit
            ax[i].plot(x, optimal_estimate,label='Optimal Fit RMSE:{:.2f}'.format(optimal_rmse),
                       linewidth=line_width, c=clrs[2])#, linestyle=(0, (6, 1)), alpha=0.8) #Densely dash dot dotted
            
            #Gait fingerprint
            ax[i].plot(x, gait_fingerprint_estimate,label='Gait Fingerprint Fit RMSE:{:.2f}'.format(gait_fingerprint_rmse),
                       linewidth=line_width, c=clrs[0], linestyle='solid', alpha=0.8) 
            
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)
            ax[i].title.set_text(trial_to_string(trial,joint_str))
            ax[i].legend()
            
#%%
def plot_cumulative_variance():
    pass
#%%
    model_foot = model_loader('foot_model.pickle')
    model_shank = model_loader('shank_model.pickle')
    model_foot_dot = model_loader('foot_dot_model.pickle')
    model_shank_dot = model_loader('shank_dot_model.pickle')
    
    
    clrs = cm.get_cmap('tab20').colors

    model = model_foot
    
    pca_values = model.scaled_pca_eigenvalues
    
    pca_values_sum= np.sum(pca_values)
    marker_on = [5]
    pca_cum_sum = np.cumsum(pca_values)/pca_values_sum
    
    plt.plot(pca_cum_sum[:11], '-o', markevery=marker_on, linewidth=7, markersize=15, mfc = 'r', mec='r',c=clrs[0])
    plt.xticks(np.arange(0, 11, 1.0))
    plt.show()
    
#%%
def validate_velocity_derivative():
    pass
#%% 
    
    trial = 's1x2i7x5'
    subject = 'AB01'
    joint = 'jointangles_shank_x'
    filename = '../local-storage/test/dataport_flattened_partial_{}.parquet'.format(subject)
    
    df = pd.read_parquet(filename)
    trial_df = df[df['trial'] == trial]
    
    model_foot_dot = model_loader('foot_dot_model.pickle')
    model_foot = model_loader('foot_model.pickle')    
    
    states = ['phase', 'phase_dot', 'step_length', 'ramp']
    states_data = trial_df[states].values.T
    
    
    foot_angle_evaluated = model_foot.evaluate_numpy(states_data)
    foot_angle_dot_evaluated = model_foot_dot.evaluate_numpy(states_data)

    
    #Calculate the derivative of foot dot manually
    foot_anles_cutoff = trial_df[joint].values[:-1]    
    foot_angles_future = trial_df[joint].values[1:]
    phase_rate = trial_df['phase_dot'].values[:-1]
    
    measured_foot_derivative = (foot_angles_future-foot_anles_cutoff)*(phase_rate)*150
    calculated_foot_derivative = foot_angle_dot_evaluated @ model_foot_dot.subjects[subject]['optimal_xi']
    
    measured_foot_angle = trial_df[joint]
    calculated_foot_angles = foot_angle_evaluated @ model_foot.subjects[subject]['optimal_xi']
    
    points_per_step = 150
    start_step = 40
    num_steps = 3 + start_step
    x = np.linspace(0,1+1/(num_steps-start_step)*points_per_step,(num_steps-start_step)*points_per_step)
    fig, axs = plt.subplots(2,1)
    axs[0].plot(x,measured_foot_derivative[start_step*points_per_step:num_steps*points_per_step])
    axs[0].plot(x,calculated_foot_derivative[start_step*points_per_step:num_steps*points_per_step])
    axs[0].legend(['measured','calculated'])
    axs[0].grid(True)

    
    axs[1].plot(x, measured_foot_angle[start_step*points_per_step:num_steps*points_per_step])
    axs[1].plot(x, calculated_foot_angles[start_step*points_per_step:num_steps*points_per_step])
    axs[1].legend([ 'measured foot angle', 'calculated foot angle'])
    axs[1].grid(True)
    plt.show()
    
#%%
if __name__=='__main__':
    pass
    #train_models()
    #validate_velocity_derivative()