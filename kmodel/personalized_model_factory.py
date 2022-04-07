
#Common imports
from xmlrpc.client import Boolean
import numpy as np 
import pandas as pd

#Import the model fitting library
from .k_model_fitting import KModelFitter

#Import personalized k model
from .personal_k_model import PersonalKModel
from .k_model import KroneckerModel
from .personal_measurement_function import PersonalMeasurementFunction
#Import PCA library
from sklearn.decomposition import PCA

#Import special type for hints that have lists
from typing import List, Union

#Imoprt serialization module
import pickle

class PersonalizedKModelFactory:

    """
    This class is meant to instantiate the personalized 
    K_model object
    """

    def __init__(self):
        pass

    def generate_personalized_model(self,k_model: KroneckerModel, 
                                      subject_data_list: list, 
                                      output_name: Union[str, List[str]],
                                      num_pca_vectors: int = None, 
                                      keep_subject_fit: str = None,
                                      left_out_subject: Union[str, List[str]] = None,
                                      vanilla_pca: Boolean = False):
        """
        This function will calculate the personalization map 
        and return a personalized k_model object

        Keyword Arguments
        k_model -- kronecker model or list of kronecker model object to perform the fit into
        subject_data_list -- list with tuples of the form (subject_name,pandas dataframe)
        output_name -- name of the output variable that will be used to fit
        num_pca_vectors -- Int, The amount of pca vectors to use. If not, 95% will be used
        keep_subject_fit -- string, the subject name that you want to set the personalized fit model to 
        left_out_subject -- this(these) subject(s) will not be included in the personalization map calculation
        
        Returns
        p_model -- personalized kronecker model        
        fit_info -- the fit information per subject
        """
        
        #If we get a kronecker model, convert it into a list just to be consistent
        if isinstance(output_name, str) == True:
            output_name = [output_name]
        
        #Initialize the arrays to aggregate the fits, average_fits
        XI_0_list = []
        XI_average_list = []
        G_list = []

        #### Calculate the fit matrix
        #Generate the fitter object
        fitter = KModelFitter()

        #Outer loop is through the different output functions
        for output in output_name: 

            #Initialize the aggregate fit matrix
            XI_output_list = []
            total_rmse_list =  []

            #Initialize aggregator for the gram calculations for the subject
            RTR_aggregator = 0
            output_datapoints = 0

            #Calculate the subject fit for every person
            for subject_name,subject_data in subject_data_list:
                
                #Skip if the subject name is in the left out pool
                if subject_name in left_out_subject:
                    continue

                #Get the subject fit
                subject_fit, (fit_rmse, RTR, num_datapoints) = fitter.fit_data(k_model, subject_data, output)

                #Add it to the list of fits
                XI_output_list.append(subject_fit)

                #Add the information about the gramian and the number of datapoints
                RTR_aggregator += RTR
                output_datapoints += num_datapoints

                #Save the rmse to calculate the average rmse for the subject
                total_rmse_list.append(fit_rmse)
            
            #Print and report the rmse 
            print(f"Average RMSE for {output} is {np.mean(total_rmse_list)}")
            
            #Convert the list into a numpy array
            XI = np.concatenate(XI_output_list,axis=0)

            #Calculate the average fit
            XI_average = np.mean(XI, axis=0).reshape(1,-1)

            #Calculate the fits minus the average
            XI_0 = XI - XI_average

            #Add the the aggregation
            XI_0_list.append(XI_0)
            XI_average_list.append(XI_average)

            #Save the Gram matrix for this subject 
            G_list.append(RTR_aggregator/output_datapoints)

        ## Now that we have all the fits for all the output functions,
        #  aggregate them and perform the pca decomposition on them

        #Create the numpy array from the aggregate lists
        XI_0_aggregate = np.concatenate(XI_0_list, axis=1)

        #Convert to the orthonormal function base if not using vanilla pca
        if(vanilla_pca == False):
            XI_0_aggregate = self._convert_to_orthonormal(XI_0_aggregate, G_list)

        #### Calculate the Principal Components
        #Get the number of output components
        k_model_output_size = k_model.get_output_size()

        #Initialize PCA object for vanilla pca calculation
        pca = PCA()
        pca.fit(XI_0_aggregate) 

        #Print out variance explained for the fit
        print(f"Explained variance {pca.explained_variance_}")

        #Determine the number of pca components if it is not supplied
        if(num_pca_vectors is None):
            num_pca_vectors = np.count_nonzero(pca.explained_variance_ratio_ <= 0.95)

        #Select the components based on the amount of gait fingerprints
        pmap_aggregate = (pca.components_[:num_pca_vectors,:])

        #Convert back from the orthonormal function base if we are not doing vanilla pca
        if(vanilla_pca == False):
            pmap_aggregate = self._convert_from_orthonormal(pmap_aggregate, G_list)

        #Create a list for the personal k models
        k_model_list = []

        #TODO: Update the pca fit for 
        #Calculate the gait fingerprint per model
        for i, output in enumerate(output_name):
            
            #Get the pmap and average fit for this model
            pmap = pmap_aggregate[:,k_model_output_size*i:k_model_output_size*(i+1)]
            xi_avg = XI_average_list[i]

            #Calculate the personalization map
            gait_fingerprint = None

            #Get the dataset for the left out subject
            for subject_name,subject_data in subject_data_list:
                
                #If the subject name is in the list, calculate the gait fingerprint
                if subject_name == keep_subject_fit:
                    gait_fingerprint = self._calculate_gait_fingerprint(k_model, subject_data, 
                                                                        output, pmap, xi_avg)

            #### Initialize and return the P model
            personalized_k_model = PersonalKModel(k_model,pmap,xi_avg,output, 
                                                  gait_fingerprint, keep_subject_fit)
            
            #Append to the model output list
            k_model_list.append(personalized_k_model)

        #Create the personalized measurement model
        personal_model_list = PersonalMeasurementFunction(k_model_list, output_name)

        return personal_model_list


    def _calculate_gait_fingerprint(self,k_model: KroneckerModel, data: list, output_name: str,
                                    pmap: np.ndarray, average_fit: np.ndarray): 
        """
        This function will calcualte the gait fingerprint fit based on the previous models

        Keyword Arguments: 
        k_model -- kronecker model object to perform the fit into
        subject_data_list -- list with tuples of the form (subject_name,pandas dataframe)
        output_name -- name of the output variable that will be used to fit
        data -- pandas dataframe with the data to perform the fit 
        output_name -- column in data that will be used as the 'y' data

        Returns:
        personalization_map -- numpy array for the gaint fingerprints
        """

        #Create a model fitter object
        data_fitter = KModelFitter()

        #Get regression matrices
        RTR, RTy, _, _ = data_fitter.calculate_regressor(k_model, data, output_name)
        
        #Use the personalization map to calculate the components of the least squares equation
        RTR_prime = (pmap @ RTR @ pmap.T)
        RTy_prime = (pmap @ RTy) - (pmap @ RTR @ average_fit.T)

        #Calculate the gait fingerprint
        gait_fingerprint = np.linalg.solve(RTR_prime, RTy_prime).T

        return gait_fingerprint

    def _convert_to_orthonormal(self, XI_0_np: np.ndarray, G_list:List[np.ndarray]) -> np.ndarray:
        """
        This function is meant to convert vectors to the orthonormal space 
        that represents deviations from the average. 

        Keyword Arguments:
        XI_0_aggregate: list of model fits in numpy array format

        Returns
        XI_0_scaled_aggregates: numpy array with the model fits in the new basis
        """
        #Split the XI_list into multiple smaller numpy arrays corresponding to every output 
        num_outputs = XI_0_np.shape[1]/G_list[0].shape[0]

        #Create a list of numpy arrays based on splitting the input array 
        XI_0_list = np.split(XI_0_np,num_outputs,axis=1)

        #Create a list of XI that are converted into the new basis per output function
        XI_0_scaled_list = []

        for XI_0,G in zip(XI_0_list,G_list):

            #Calculate the change of basis transformation matrix
            #Diagonalize the matrix G as G = OVO
            eig, O = np.linalg.eigh(G)

            #This is based on the equation in eq:Qdef
            # Q G Q = I
            Qinv = sum([O[:,[i]] @ O[:,[i]].T * np.sqrt(eig[i]) for i in range(len(eig))])

            #Calculate the model's fit in the orthonormal space
            XI_0_scaled = XI_0 @ Qinv

            #Append to the list
            XI_0_scaled_list.append(XI_0_scaled)

        #Create a numpy array from all the new vectors
        XI_0_scaled_aggregate = np.concatenate(XI_0_scaled_list, axis=1)

        return XI_0_scaled_aggregate

    def _convert_from_orthonormal(self, XI_0_np: np.ndarray, G_list:List[np.ndarray]) -> np.ndarray:
        """
        This function is meant to convert vectors to the orthonormal space 
        that represents deviations from the average. 

        Keyword Arguments:
        XI_0_scaled_aggregate: list of model fits in numpy array format in the orthonormal basis
        
        Returns: 
        XI_0_aggregate: list of model fits in numpy array format in the original basis
        """

        #Split the XI_list into multiple smaller numpy arrays corresponding to every output 
        num_outputs = XI_0_np.shape[1]/G_list[0].shape[0]

        #Create a list of numpy arrays based on splitting the input array 
        XI_0_list = np.split(XI_0_np,num_outputs,axis=1)

        #Create a list of XI that are converted into the new basis per output function
        XI_0_original_list = []

        for XI_0,G in zip(XI_0_list,G_list):

            #Calculate the change of basis transformation matrix
            #Diagonalize the matrix G as G = OVO
            eig, O = np.linalg.eigh(G)

            #This is based on the equation in eq:Qdef
            # Q G Q = I
            Q = sum([O[:,[i]] @ O[:,[i]].T * 1/np.sqrt(eig[i]) for i in range(len(eig))])

            #Calculate the model's fit in the orthonormal space
            XI_0_original = XI_0 @ Q

            #Append to the list
            XI_0_original_list.append(XI_0_original)

        #Create a numpy array from all the new vectors
        XI_0_aggregate = np.concatenate(XI_0_original_list, axis=1)

        return XI_0_aggregate


    #Save the model so that you can use them later
    def save_model(self,model : PersonalMeasurementFunction,filename):
        """
        Saves a model to a pickle file

        Keyword Arguments:
        model: the personal measurement function that will be saved

        Returns: 
        None
        """
        with open(filename,'wb') as file:
            pickle.dump(model,file)

    #Load the model from a file
    def load_model(self,filename) -> PersonalMeasurementFunction:
        """
        Loads a model from a pickle file

        Keyword Arguments:
        filename: string that corresponds to the file that will be loaded

        Returns: 
        model: the personal measurement function corresponding to the name

        """
        with open(filename,'rb') as file:
            return pickle.load(file)


    
    def _calculate_normalized_personalization_map(Gs: List[np.ndarray], p):
        pass