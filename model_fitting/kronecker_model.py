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

#Model Object:
# list of basis objects
# string description
# model_size
# pca_axis - list of np array of model_size length
# coefficient_list - list of coefficients for each pca_axis
#--------------------------
class Kronecker_Model:
    def __init__(self, output_name,*funcs,time_derivative=False,personalization_map=None,subjects=None,num_gait_fingerprint=None):
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
        if (self.time_derivative == True):
            if partial_derivatives is not None:
                partial_derivatives.append('time')
                partial_derivatives.append('phase')
            else:
                partial_derivatives = ['time','phase']
                
        return pandas_kronecker(dataframe,self.funcs,partial_derivatives)
    
    
    def evaluate_subject_optimal(self,subject, dataframe):
        regressor = self.evaluate_pandas(dataframe)
        print(regressor.shape)
        xi = self.subjects[subject]['optimal_xi']
        print(regressor.shape)
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
        
        xi = self.personalization_map @ gait_finterprint_vector
        
        return row_function @ xi

    def evaluate_numpy(self,states,partial_derivatives=None):
        #Copy the dict so that we dont modify the global one
        if(partial_derivatives == None):
            local_partial_derivatives = dict(self.no_derivatives)
        else:
            local_partial_derivatives = dict(partial_derivatives)
        
        #Make sure we only have the amount of states that we want
        phase_states = states[:self.num_gait_fingerprint,:]

        if self.time_derivative == True:
            local_partial_derivatives['phase']+=1
            #print(local_partial_derivatives)

        output = np.array(1).reshape(1,1,1)

        for func,state in zip(self.funcs,phase_states):
            output = (output[:,np.newaxis,:]*func.evaluate_derivative(state[:,np.newaxis],local_partial_derivatives[func.var_name])[:,:,np.newaxis]).reshape(1,-1)
            #print("I'm alive, size = " + str(output.shape))
    
        if self.time_derivative == True:
            #print("Phase dot: " + str(phase_states[1]))
            output = phase_states[1]*output

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

        if(partial_derivatives is None):
            partial_derivatives = self.no_derivatives

        else:
            derive = False
            for name in self.gait_fingerprint_names:
                if partial_derivatives[name] > 0:
                    if(derive==True):
                        gait_fingerprints[int(name[-1])-1] = 0
                        break
                    else:
                        for name_2 in self.gait_fingerprint_names:
                            gait_fingerprints[int(name_2[-1])-1] = 0
                        gait_fingerprints[int(name[-1])-1] = 1
                        derive = True


        xi = self.personalization_map @ gait_fingerprints

        row_vector = self.evaluate_numpy(phase_states,partial_derivatives)

        return row_vector @ xi


    
    #Future optimizations
    #@numba.jit(nopython=True, parallel=True)
    def least_squares(self,dataframe,output,splits=100):

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
            
            RTR += RTR_
            yTR += y.T @ R
            RTy += R.T @ y
            yTy += y.T @ y
        pass
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
       	#G = G_total/N_total
        G = G_total/1


        #Verify we are positive semidefinite
        assert(np.linalg.norm(G-G.T)<1e-7)

        #Diagonalize the matrix G as G = OVO
        eig, O = np.linalg.eigh(G)
        V = np.diagflat(eig)
        print("Gramian {}".format(G))
        #Additionally, all the eigenvalues are true
        for e in eig:
            print("Eigenvalue: {}".format(e))
            assert (e >= 0)
            assert( e>0) # pd

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
        Ψ = np.diagflat(ψs)

        #If we change the eigenvalues we also need to change the eigenvectors
        U = np.flip(Uinverted, axis=1)

        #Run tests to make sure that this is working
        assert(np.linalg.norm(Σ - U @ Ψ @ U.T)< 1e-7 * np.linalg.norm(Σ)) # passes
        for i in range(len(ψs)-1):
            assert(ψs[i] > ψs[i+1])

        #Define the amount principles axis that we want
        #η = num_gait_fingerprints
        η=self.num_gait_fingerprint
        pca_axis_array = []

        #Convert from the new basis back to the original basis vectors
        for i in range (0,η):
            pca_axis_array.append(param_from_orthonormal(U[:,i]*np.sqrt(ψs[i])))

        #Return the personalization map
        return np.array(pca_axis_array).T


    def calculate_gait_fingerprint(self,n=None,cum_var=0.95):
        num_subjects = len(self.subjects.values())
        #Get all the gait fingerprints into a matrix
        XI = np.array([subject['optimal_xi'] for subject in self.subjects.values()]).reshape(num_subjects,-1)
        print(XI.shape)
        pca = PCA(n_components=num_subjects)
        pca.fit(XI) 
        self.pca_result = pca
        
        #Get how many personalization coefficients we want
        if n == None:
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            i = 0
            for element in cumulative_variance:
                if element > cum_var:
                    i+=1
                    break
                else:
                    i+=1
            n = i
        
        self.personalization_map = (pca.components_[:n,:]).T
        self.personalization_map_scaled = self.scaled_pca()
        



    def sum_pca_axis(self,pca_coefficients):
        if(len(self.pca_axis) != len(pca_coefficients)):
            err_string = 'Wrong amount of inputs. Received:'  + str(len(pca_coefficients)) + ', expected:' + str(len(self.pca_axis))
            raise ValueError(err_string)

        return sum([axis*coeff for axis,coeff in zip(self.pca_axis,pca_coefficients)])

    def evaluate_scalar_output(self,*function_inputs,partial_derivative=None):
        states = function_inputs[:self.num_states]
        pca_coefficients = function_inputs[self.num_states:]
        return self.evaluate(*states,partial_derivative=partial_derivative) @ self.sum_pca_axis(pca_coefficients).T

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


# def optimized_least_squares(output,input,model_size,splits=10):
#     RTR = np.zeros((model_size,model_size))
#     yTR = np.zeros((1,model_size))
#     RTy = np.zeros((model_size,1))
#     yTy = 0
#     for sub_dataframe in np.array_split(input,splits):
#         R = numpy_kronecker(sub_dataframe,self.funcs)
#         #nans = sub_dataframe[output].isnull().sum()
#         #print(nans)
#         #print(sub_dataframe.shape[0])
#         y = sub_dataframe[output].values[:,np.newaxis]
#         RTR_ = R.T @ R
        
#         RTR += RTR_
#         yTR += y.T @ R
#         RTy += R.T @ y
#         yTy += y.T @ y

#     return np.linalg.solve(RTR, RTy), RTR, RTy, yTR, yTy


#Evaluate model 
def model_prediction(model,ξ,*input_list,partial_derivative=None):
    result = [model.evaluate(*function_inputs,partial_derivative=partial_derivative)@ξ for function_inputs in zip(*input_list)]
    return np.array(result)


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


def unit_test():
    pass
    #%%
    from function_bases import Fourier_Basis, Polynomial_Basis
    import copy
    
    train_models = True
    if train_models == True:
        #Determine the phase models
        phase_model = Fourier_Basis(3,'phase')
        phase_dot_model = Polynomial_Basis(3,'phase_dot')
        step_length_model = Polynomial_Basis(3,'step_length')
        ramp_model = Polynomial_Basis(3,'ramp')
    
        # #Get the subjects
        subjects = [('AB10','../local-storage/test/dataport_flattened_partial_AB10.parquet')]
        for i in range(1,10):
            subjects.append(('AB0'+str(i),'../local-storage/test/dataport_flattened_partial_AB0'+str(i)+'.parquet'))

        model_foot = Kronecker_Model('jointangles_foot_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=5)
        model_saver(model_foot,'foot_model.pickle')
        
        model_shank = Kronecker_Model('jointangles_shank_x',phase_model,phase_dot_model,step_length_model,ramp_model,subjects=subjects,num_gait_fingerprint=5)
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
        
#%%
if __name__=='__main__':
    unit_test()