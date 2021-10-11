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


#Relative Imports
from .context import math_utils


#Set test assert_pd for speed boost
math_utils.test_pd = False
    
#Model Object:
# The kronecker model will take in multiple models and then calculate the 
# kronecker product of all of them in runtime
#--------------------------
class KroneckerModel:
    def __init__(self, output_name,*funcs,subjects=None,num_gait_fingerprint=5):
        self.funcs = funcs

        #Calculate the size of the parameter array
        #Additionally, pre-allocate arrays for kronecker products intermediaries 
        # to speed up results
        self.output_name = output_name
        self.order = []
        size = 1
        for func in funcs:
            #Since we multiply left to right, the total size will be on the left 
            #and the size for the new row will be on the right
            print((str(size), str(func.size)))

            size = size * func.size

            self.order.append(func.var_name)


        self.size = size
        self.num_states = len(funcs)
        self.subjects = {}
        self.one_left_out_subjects = {}
        self.num_gait_fingerprint = num_gait_fingerprint
        self.gait_fingerprint_names = ["gf"+str(i) for i in range(1,num_gait_fingerprint+1)]
        #Todo: Add average pca coefficient
        self.cross_model_personalization_map = None
        self.cross_model_inter_subject_average = None        
        
        if(subjects is not None):
            self.add_subject(subjects)
            self.fit_subjects()
            self.calculate_gait_fingerprint(n=num_gait_fingerprint)
        
    
    def add_subject(self,subjects):

        import os

        print("CWD is: " + os.getcwd())


        for subject,filename in subjects:
            self.subjects[subject] = \
                {'filename': filename, \
                 'dataframe': pd.read_parquet(filename, columns=[self.output_name,*self.order]), \
                 'optimal_xi': [], \
                 'least_squares_info': [], \
                 'pca_axis': [], \
                 'pca_coefficients': [] \
             }

    
    
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
    

    def evaluate_gait_fingerprint(self,dataframe):
            
        #Get the row function 
        row_function = self.evaluate_pandas(dataframe)
        
        #Get the gait fingerprints from the dataframe as numpy array 
        gait_fingerprint = dataframe[self.gait_fingerprint_names].values.T          
        
        #Get the personalization map for the person
        xi = self.personalization_map_scaled @ gait_fingerprint
        
        return row_function @ xi

    #@vectorize(nopython=True)
    def evaluate_pandas(self, dataframe):
        rows = dataframe.shape[0]
        output = np.array(1).reshape(1,1,1)
            
        for func in self.funcs:
            
            output = (output[:,np.newaxis,:]*func.evaluate(dataframe[func.var_name].values[:,np.newaxis])[:,:,np.newaxis]).reshape(rows,-1)
            
        return output

    def evaluate_numpy(self,states):
        rows = states.shape[1]
        
        #Make sure we only have the amount of states that we want
        phase_states = states[:self.num_gait_fingerprint,:]

        output = np.array(1).reshape(1,1,1)

        for func,state in zip(self.funcs,phase_states):
            state_t = (state.T)[:,np.newaxis]
            
            eval_value = func.evaluate(state_t).reshape(rows,-1,1)
            
            output = (output[:,np.newaxis,:]*eval_value).reshape(rows,-1)
                
        return output

    def evaluate_gait_fingerprint_numpy(self,states):
        
        phase_states =  states[:self.num_states]
        
        gait_fingerprints = states[self.num_states:]
    
        xi = (self.personalization_map @ gait_fingerprints) + self.inter_subject_average_fit            
        
        row_vector = self.evaluate_numpy(phase_states)

        return row_vector @ xi

    
    def evaluate_gait_fingerprint_cross_model_numpy(self,states):
        
        phase_states =  states[:self.num_states]
        
        gait_fingerprints = states[self.num_states:]
        
        xi = (self.cross_model_personalization_map @ gait_fingerprints) + self.cross_model_inter_subject_average
          
        row_vector = self.evaluate_numpy(phase_states)

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
           
            y = sub_dataframe[output].values[:,np.newaxis]
            RTR_ = R.T @ R
        
            print("RTR rank: {} RTR shape {}".format(np.linalg.matrix_rank(RTR,hermitian=True),RTR.shape))
            
            RTR += RTR_
            yTR += y.T @ R
            RTy += R.T @ y
            yTy += y.T @ y
        
        try:
            result = (np.linalg.solve(RTR, RTy), num_rows, RTR, RTy, yTR, yTy)
        except:
            print('Singular RTR in optimal fit least squares')
            result = (0, num_rows, RTR, RTy, yTR, yTy)

        return result
       


    def fit_subjects(self):
        
        for name,subject_dict in self.subjects.items():
            print("Doing " + name)
            print(subject_dict['filename'])
            data = subject_dict['dataframe']
            output = self.least_squares(data,self.output_name)
            subject_dict['optimal_xi'] = output[0]
            subject_dict['num_rows'] = output[1]
            subject_dict['least_squares_info'] = output[2:]
        
    def scaled_pca_single_model(self, XI=None):
        
        num_subjects = len(self.subjects)
        Ξ = np.array([subject['optimal_xi'] for subject in self.subjects.values()]).reshape(num_subjects,-1)
        G_total = 0
        N_total = 0
        
        for name,subject_dict in self.subjects.items():
            G_total += subject_dict['least_squares_info'][0]
            N_total += subject_dict['num_rows']
	
        #This is equation eq:inner_regressor in the paper!
       	G = G_total/N_total
           
        #This function verifies positive semidefinite, so we dont have to
        personalization_map_scaled, pca_info = scaled_pca(Ξ,G)
                
        self.inter_subject_average_fit = pca_info['inter_subject_average_fit']
        
        self.scaled_pca_eigenvalues = pca_info['eigenvalues']
        
        return personalization_map_scaled


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
        self.personalization_map_scaled = self.scaled_pca_single_model()
        
        #Calculate gait fingerprints for every individual
        for subject_dict in self.subjects.values():

            #Get least squares info
            RTR, RTy, yTR, yTy = subject_dict['least_squares_info']
            
            #Get scaled personalization map gait fingerprint
            pmap = self.personalization_map_scaled
            avg_fit = self.inter_subject_average_fit
            
            RTR_prime = (pmap.T) @ RTR @ pmap
            RTy_prime = (pmap.T) @ RTy-(pmap.T) @ RTR @ avg_fit
            subject_dict['gait_coefficients'] = np.linalg.solve(RTR_prime,RTy_prime)
            
            
            
            #Get bad (naive pca) personalization map gait fingerprint 
            pmap = self.personalization_map
            avg_fit = self.inter_subject_average_fit
            
            RTR_prime = (pmap.T) @ RTR @ pmap
            RTy_prime = (pmap.T) @ RTy-(pmap.T) @ RTR @ avg_fit
            
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

            #Temporary addition to try to fit with only walking data
            #data = data[data['ramp'] == 0.0]
            
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
            #print(f'asset')            
            
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


def scaled_pca(Ξ,G,num_gait_fingerprint=5):
        
    
        math_utils.assert_pd(G, 'G in scaled pca')
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
        Q = np.zeros((O.shape[0],O.shape[0]))
        Qinv = np.zeros((O.shape[0],O.shape[0]))
        for i in range(len(eig)):
            Q += O[:,[i]] @ O[:,[i]].T * 1/np.sqrt(eig[i])
            Qinv += O[:,[i]] @ O[:,[i]].T * np.sqrt(eig[i])
        # Q       = sum([O[:,[i]] @ O[:,[i]].T * 1/np.sqrt(eig[i]) for i in range(len(eig))])
        # Qinv    = sum([O[:,[i]] @ O[:,[i]].T * np.sqrt(eig[i]) for i in range(len(eig))])

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
        inter_subject_average_fit = ξ_avg[:,np.newaxis]
        
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
        scaled_pca_eigenvalues = ψs
        Ψ = np.diagflat(ψs)

        #If we change the eigenvalues we also need to change the eigenvectors
        U = np.flip(Uinverted, axis=1)

        #Run tests to make sure that this is working
        assert(np.linalg.norm(Σ - U @ Ψ @ U.T)< 1e-7 * np.linalg.norm(Σ)) # passes
        for i in range(len(ψs)-1):
            assert(ψs[i] > ψs[i+1])

        #Define the amount principles axis that we want
        η = Ξ.shape[1]
        pca_axis_array = []

        #Convert from the new basis back to the original basis vectors
        for i in range (0,η):
            pca_axis_array.append(param_from_orthonormal(U[:,i]*np.sqrt(ψs[i])))
            
        scaled_pca_components = np.array(pca_axis_array).T
        
        pca_info = {'all_components': scaled_pca_components,
                    'eigenvalues':scaled_pca_eigenvalues,
                    'inter_subject_average_fit':inter_subject_average_fit}
        #Return the personalization map
        return scaled_pca_components[:,:num_gait_fingerprint], pca_info

def calculate_cross_model_p_map(models):
    
    num_models = len(models)
    
    subjects = models[0].subjects.keys()
    
    XI_list = [models[0].subjects[subject]['optimal_xi'] for subject in subjects]
    
    for model in models[1:]: 
        for i,subject in enumerate(subjects):    
            XI_list[i] = np.concatenate((XI_list[i], model.subjects[subject]['optimal_xi']), axis=0)
            
    XI = np.array(XI_list)
    XI = XI.reshape(XI.shape[:2])
    
    G_total = [0 for i in range(num_models)]
    N_total = [0 for i in range(num_models)]
    
    for i, model in enumerate(models):
        for name,subject_dict in model.subjects.items():
            G_total[i] += subject_dict['least_squares_info'][0]
            N_total[i] += subject_dict['num_rows']
	
    #This is equation eq:inner_regressor in the paper!
    G_list = [G_total[i]/N_total[i] for i in range(num_models)]
    
    for individual_G in G_list: 
        math_utils.assert_pd(individual_G, "Individual G in G_list")
    
    #from https://stackoverflow.com/questions/42154606/python-numpy-how-to-construct-a-big-diagonal-arraymatrix-from-two-small-array
    def diag_block_mat_boolindex(L):
        shp = L[0].shape
        mask = np.kron(np.eye(len(L)), np.ones(shp))==1
        out = np.zeros(np.asarray(shp)*len(L),dtype=float)
        out[mask] = np.concatenate(L).ravel()
        return out
    
    G = diag_block_mat_boolindex(G_list)
        
    personalization_map, pca_info = scaled_pca(XI, G)
    
    avg_fit = pca_info['inter_subject_average_fit']
    print(f'cross personalization_map {personalization_map.shape}  avg_fit:{avg_fit.shape}')    

    #Assign each personalization map to the corresponding model
    #Create even splits
    split_personalization_map = []
    split_average_fit = []
    for i in range(len(models)):
        pmap_size = int(personalization_map.shape[0]/num_models)
        temp1 = personalization_map[i*pmap_size:(i+1)*pmap_size,:]
        temp2 = avg_fit[i*pmap_size:(i+1)*pmap_size,:]
        
        split_personalization_map.append(temp1)
        split_average_fit.append(temp2)
    
    #Todo
    
    #set the model for each part 
    for i,mod in enumerate(models):
        
        mod.cross_model_personalization_map = split_personalization_map[i]
        mod.cross_model_inter_subject_average = split_average_fit[i]
       
    #For every subject, calculate the cross model thing
    for j,subject in enumerate(mod.subjects.keys()):
        
        #For every model, add to the least square matrix
        for i,mod in enumerate(models):
    
            #Get least squares info
            RTR, RTy, yTR, yTy = mod.subjects[subject]['least_squares_info'] 
            pmap = mod.cross_model_personalization_map
            avg_fit = mod.cross_model_inter_subject_average
            
            print(f'j:{j} i:{i} RTR: {RTR.shape}  RTy: {RTy.shape} pmap: {pmap.shape}  avg_fit: {avg_fit.shape}')
        
            RTR_prime = (pmap.T) @ RTR @ pmap
            RTy_prime = (pmap.T) @ RTy - (pmap.T) @ RTR @ avg_fit
            
            if i == 0:
                RTR_prime_stack = RTR_prime
                RTy_prime_stack = RTy_prime
            else:
                RTR_prime_stack += RTR_prime
                RTy_prime_stack += RTy_prime

            
            gait_fingerprint = np.linalg.solve(RTR_prime_stack,RTy_prime_stack)
            for i,mod2 in enumerate(models):

                mod2.subjects[subject]['cross_model_gait_coefficients_unscaled'] = gait_fingerprint
    

#Save the model so that you can use them later
def model_saver(model,filename):
    with open(filename,'wb') as file:
        pickle.dump(model,file)

#Load the model from a file
def model_loader(filename):
    with open(filename,'rb') as file:
        return pickle.load(file)
    
    
    
########################################################################################################################################################################
#//////////////////////////////////////////////////////////////////////////////////##//////////////////////////////////////////////////////////////////////////////////#
########################################################################################################################################################################