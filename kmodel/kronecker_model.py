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
    def __init__(self, output_name,*funcs,subjects=None,num_gait_fingerprint=4):
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
            self.calculate_pmap(n=num_gait_fingerprint)
        


    #Add a subject without running any fitting
    def add_subject(self,subjects):
        import os
        print("CWD is: " + os.getcwd())
        for subject,filename in subjects:
            self.subjects[subject] = \
                {'filename': filename, \
                 'dataframe': pd.read_parquet(filename, columns=[self.output_name,*self.order])
             }


    def evaluate_pandas(self, dataframe):
        rows = dataframe.shape[0]
        output = np.array(1).reshape(1,1,1)
            
        for func in self.funcs:
            #Knronecker product per state
            #Get data
            var_data = dataframe[func.var_name].values
            var_data2 = var_data
            intermediary_output = func.evaluate(var_data2)
            output = (output[:,np.newaxis,:]*intermediary_output[:,:,np.newaxis]).reshape(rows,-1)
            
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
    

    def least_squares(self,dataframe,output,splits=50):
        
        #Initialize matrices to build them up with
        # low rank updates
        RTR = np.zeros((self.size,self.size))
        yTR = np.zeros((1,self.size))
        RTy = np.zeros((self.size,1))
        yTy = 0
        num_rows = len(dataframe.index)
        
        #Divide the dataset to sizes that can be fit in memory 
        # for least squares calculations
        for sub_dataframe in np.array_split(dataframe,splits):

            #Get the regressor matrix for the chunk
            R = self.evaluate_pandas(sub_dataframe)
           
           #Get the expected output
            y = sub_dataframe[output].values[:,np.newaxis]

            #Calculate the rank update
            RTR_ = R.T @ R
        
            #print("RTR rank: {} RTR shape {}".format(np.linalg.matrix_rank(RTR,hermitian=True),RTR.shape))
            
            #Add the low rank update to the accumulator matrices
            RTR += RTR_
            yTR += y.T @ R
            RTy += R.T @ y
            yTy += y.T @ y
        
        #If you have a low rank reggressor you cannot estimate 
        try:
            x = np.linalg.solve(RTR, RTy)
            residual = np.sqrt((x.T @ RTR @ x - x.T @ RTy - yTR @ x + yTy)/num_rows)

            result = (x, num_rows, residual, RTR, RTy, yTR, yTy)
        except:
            print('Singular RTR in optimal fit least squares')
            x = 0
            residual = float('inf')
            result = (0, num_rows, residual, RTR, RTy, yTR, yTy)

        return result


    def fit_subjects(self):
        #Get the optimal least squares fit for every subject
        for name,subject_dict in self.subjects.items():
            print("Doing " + name)
            print(subject_dict['filename'])
            data = subject_dict['dataframe']
            output = self.least_squares(data,self.output_name)
            subject_dict['optimal_xi'] = output[0]
            subject_dict['num_rows'] = output[1]
            subject_dict['least_squares_info'] = output[3:]



    

    def calculate_pmap(self,n):
        """
        Calculate the personalization map

        This function assumes that you already have calculated the optimal fits 
        for every subject. 

        Inputs:
        n - The amount of gait fingerprints that the personalization map will have

        Outputs:
        Personalization map 
        Personalization map vanilla pca
        Least-sqaure based gait fingerprint for every person
        """    

        ########
        # Vanilla pca implementation
        ########

        #Get the number of subjects
        num_subjects = len(self.subjects.values())

        #Get all the gait fingerprints into a matrix
        XI = np.array([subject['optimal_xi'] for subject in self.subjects.values()]).reshape(num_subjects,-1)
        
        #Get the mean gait fingerprint
        XI_mean = XI.mean(axis=0).reshape(1,-1)

        #Get the deviations from the average matrix
        XI_0 = XI - XI_mean
      

        #Initialize PCA object for vanilla pca calculation
        pca = PCA(n_components=min(num_subjects,XI_0.shape[1]))
        pca.fit(XI_0) 

        
        ########
        # Our scaled pca implementation
        ########

        #Calculate the scaled version of the pca
        scaled_pca = self.scaled_pca_single_model()


        #Select the components based on the amount of gait fingerprints
        local_vanilla_pmap = (pca.components_[:n,:]).T
        local_pmap = scaled_pca[:,:n]

        #Calculate gait fingerprints for every individual
        for subject_dict in self.subjects.values():

            #Get least squares info
            RTR, RTy, yTR, yTy = subject_dict['least_squares_info']
            
            #Get scaled personalization map gait fingerprint
            pmap = local_pmap
            avg_fit = self.inter_subject_average_fit
            
            #Reuse the least square usage of the personalization map
            RTR_prime = (pmap.T) @ RTR @ pmap
            RTy_prime = (pmap.T) @ RTy-(pmap.T) @ RTR @ avg_fit

            subject_dict['gait_coefficients'] = np.linalg.solve(RTR_prime,RTy_prime)
            
            
            
            #Get bad (naive pca) personalization map gait fingerprint 
            pmap = local_vanilla_pmap
            avg_fit = self.inter_subject_average_fit
            
            RTR_prime = (pmap.T) @ RTR @ pmap
            RTy_prime = (pmap.T) @ RTy-(pmap.T) @ RTR @ avg_fit
            
            subject_dict['gait_coefficients_vanilla_pca'] = np.linalg.solve(RTR_prime,RTy_prime)
    

        #Get the personalization map with our rmse scaling
        self.personalization_map = local_pmap

        #Get the personalization map with normal pca
        self.personalization_map_vanilla = local_vanilla_pmap
            

    def scaled_pca_single_model(self, XI=None):
        
        """
        Calculate the rmse scaled PCA from the paper
        """

        #Get the number of subjects        
        num_subjects = len(self.subjects)
        
        #Calculate the matrix of user fits
        Ξ = np.array([subject['optimal_xi'] for subject in self.subjects.values()]).reshape(num_subjects,-1)
        
        #Calculate the scaling factors
        G_total = 0
        N_total = 0
        
        for name,subject_dict in self.subjects.items():
            G_total += subject_dict['least_squares_info'][0]
            N_total += subject_dict['num_rows']
	
        #This is equation eq:inner_regressor in the paper!
       	G = G_total/N_total
           
        #Get the personalization map and the pca info
        personalization_map, pca_info = scaled_pca(Ξ,G)
                
        self.inter_subject_average_fit = pca_info['inter_subject_average_fit']
        
                
        return personalization_map



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


            pmap_scaled = self.personalization_map
            pmap_vanilla = self.personalization_map_vanilla

            output = self.least_squares(data,self.output_name)
            subject_dict['optimal_xi'] = output[0]
            subject_dict['num_rows'] = output[1]
            subject_dict['least_squares_info'] = output[3:]
            
            xi_avg = self.inter_subject_average_fit

            RTR, RTy, yTR, yTy = subject_dict['least_squares_info']
            RTR_prime = (pmap_scaled.T) @ RTR @ pmap_scaled
            RTy_prime = (pmap_scaled.T) @ RTy-(pmap_scaled.T) @ RTR @ xi_avg

            gf_scaled = np.linalg.solve(RTR_prime,RTy_prime)
        

            subject_dict['gait_coefficients'] = (gf_scaled)


            RTR, RTy, yTR, yTy = subject_dict['least_squares_info']
            RTR_prime = (pmap_vanilla.T) @ RTR @ pmap_vanilla
            RTy_prime = (pmap_vanilla.T) @ RTy-(pmap_vanilla.T) @ RTR @ xi_avg
          

            gf_unscaled = np.linalg.solve(RTR_prime,RTy_prime)
            
            subject_dict['gait_coefficients_unscaled'] = (gf_unscaled)



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


def scaled_pca(Ξ,G):
        
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
    return scaled_pca_components[:,:], pca_info

def calculate_cross_model_p_map(models):
    """
    Calculate the gait fingerprint across different models
    """

    #Get the number of models
    num_models = len(models)
    
    subjects = models[0].subjects.keys()
    

    #Concatenate all the model fits
    XI_list = [models[0].subjects[subject]['optimal_xi'] for subject in subjects]
    
    for model in models[1:]: 
        for i,subject in enumerate(subjects):    
            XI_list[i] = np.concatenate((XI_list[i], model.subjects[subject]['optimal_xi']), axis=0)
            

    #Convert the model lists into a 2D np array        
    XI = np.array(XI_list)
    XI = XI.reshape(XI.shape[:2])

    #Calculate the scalling coefficients
    G_total = [0 for i in range(num_models)]
    N_total = [0 for i in range(num_models)]
    
    for i, model in enumerate(models):
        for name,subject_dict in model.subjects.items():
            G_total[i] += subject_dict['least_squares_info'][0]
            N_total[i] += subject_dict['num_rows']
	
    #This is equation eq:inner_regressor in the paper!
    G_list = [G_total[i]/N_total[i] for i in range(num_models)]
    
    #Make sure that everything is positive definite
    for individual_G in G_list: 
        math_utils.assert_pd(individual_G, "Individual G in G_list")
    
    #from https://stackoverflow.com/questions/42154606/python-numpy-how-to-construct-a-big-diagonal-arraymatrix-from-two-small-array
    def diag_block_mat_boolindex(L):
        shp = L[0].shape
        mask = np.kron(np.eye(len(L)), np.ones(shp))==1
        out = np.zeros(np.asarray(shp)*len(L),dtype=float)
        out[mask] = np.concatenate(L).ravel()
        return out
    
    #Diagonalize the weights
    G = diag_block_mat_boolindex(G_list)
        
    personalization_map, pca_info = scaled_pca(XI, G)
    
    avg_fit = pca_info['inter_subject_average_fit']
    print(f'cross personalization_map {personalization_map.shape}  avg_fit:{avg_fit.shape}')    

    #Get the number of gait fingerprints
    num_gf = models[0].num_gait_fingerprint

    #Assign each personalization map to the corresponding model
    #Create even splits
    split_personalization_map = []
    split_average_fit = []
    for i in range(len(models)):
        pmap_size = int(personalization_map.shape[0]/num_models)
        temp1 = personalization_map[i*pmap_size:(i+1)*pmap_size,:]
        temp2 = avg_fit[i*pmap_size:(i+1)*pmap_size,:]
        
        #Make sure you have the correct number of gait fingerprints
        split_personalization_map.append(temp1[:,:num_gf])
        split_average_fit.append(temp2[:,:num_gf])
    
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