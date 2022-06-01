"""
This file is meant to define the KroneckerModel object 
that takes different function bases and combines them 

"""

#Common imports
import pandas as pd
import numpy as np
from kmodel.function_bases import Basis
from typing import List

class KroneckerModel():


    def __init__(self,basis_list: List[Basis]):
        """
        Initialize the object 

        Keyword arguments:
        basis_list -- function bases that are going to be kronecker
                  producted together
        """
        #Store the basis list
        self.basis_list = basis_list

        #Store the number of basis that is provided
        self.num_basis = len(basis_list)

        #Store the names of the basis
        self.basis_names = [basis.var_name for basis in basis_list]

        #Store the shape of the output
        # the output array is the product of the size of all the functions
        self.output_size = np.product([basis.size for basis in basis_list])


    def get_basis_names(self):
        """
        Returns the names of the basis

        This is useful to remember the order if you save the basis and 
        reload it later on
        """

        return self.basis_names


    def get_num_basis(self):
        """
        Returns the amount of basis 

        This is useful to inform the size of the array when using 
        evaluate with a numpy input
        """

        return self.num_basis


    def get_output_size(self):
        """
        Returns the size of the output vector
        
        This is useful to know the size of the output of 
        the kronecker basis
        """

        return self.output_size


    def evaluate(self,dataset):
        """
        Evaluate the basis with the corresponding dataset

        Keyword arguments:
        np_dataset -- Numpy array with data to evaluate.
                      shape (num_datapoints, num_basiss)

        pd_dataset -- Data in pandas format. Expects the columns to 
                      have the each basis name. 

        Returns:
        output -- evaluation of the kronecker product
                  shape (num_datapoints, output_size)

        """
        
        #Invoke the corresponding function
        if(isinstance(dataset, np.ndarray) == True):
            return self._evaluate_numpy(dataset)
        
        elif (isinstance(dataset, pd.DataFrame) == True):
            return self._evaluate_pandas(dataset)

        else:
            raise TypeError(message="You need to input either dataset using either \
                                     np array or pandas dataframe")


    def _evaluate_numpy(self,np_dataset):
        """
        Evaluate the basis with a numpy input

        Keyword arguments:
        np_dataset -- numpy dataset with 
                      shape (num_datapoints, num_basis)

        Returns:
        output -- evaluation of the kronecker product
                  shape (num_datapoints, output_size)
        """

        #Get the number of rows that we are evaluating
        num_datapoints = np_dataset.shape[0]
        
        #Make sure we only have the amount of states that correspond to our basis
        basis_variables = np_dataset[:,:self.num_basis]

        #Initialize the output variable that will be updated in the loop
        output = np.array(1).reshape(1,1,1)

        #Calculate the kronecker basis per state
        # Transpose since numpy iterates by rows and our features are in columns
        for func,state in zip(self.basis_list,basis_variables.T):

            #Reshape the state to a columns vector
            state_t = state.reshape(-1,1)
            
            #Evaluate the function at the current state
            eval_value = func.evaluate(state_t).reshape(num_datapoints,-1,1)
            
            #Aggregate the kronecker product
            output = (output[:,np.newaxis,:]*eval_value).reshape(num_datapoints,-1)
                
        return output
    

    def _evaluate_pandas(self, pd_dataset):
        """
        Evaluate the basis with a pandas Dataframe
        
        Keyword arguments:
        pd_dataset -- Dataframe with the columns having the 
                      names of each basis
        
        Returns:
        output -- evaluation of the kronecker product. Numpy array with
                  shape (num_datapoints, output_size)
        
        """

        #Get the rows to infer size
        num_datapoints = pd_dataset.shape[0]

        #Initialize the output of the basis that will be updated in the loop
        output = np.array(1).reshape(1,1,1)
        
        #Calculate the kronecker basis per state
        for func in self.basis_list:
            
            #Get data in numpy form
            var_data = pd_dataset[func.var_name].values

            #Calculate the specific function evaluated at the datapoints
            intermediary_output = func.evaluate(var_data)

            #Convert matrices to 3D to execute the kronecker product multiplication
            output = (output[:,np.newaxis,:]*intermediary_output[:,:,np.newaxis]).reshape(num_datapoints,-1)
            
        return output