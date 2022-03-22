
#Common imports
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
                                      left_out_subject: Union[str, List[str]] = None):
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

        #### Calculate the fit matrix
        #Generate the fitter object
        fitter = KModelFitter()

        for output in output_name: 

            #Initialize the aggregate fit matrix
            XI_list = []

            #Calculate the subject fit for every person
            for subject_name,subject_data in subject_data_list:
                
                #Skip if the subject name is in the left out pool
                if subject_name in left_out_subject:
                    continue

                #Get the subject fit
                subject_fit,_ = fitter.fit_data(k_model, subject_data, output)

                #Add it to the list of fits
                #Transpose to get columns vectors since 
                # that is what numpy expects
                XI_list.append(subject_fit)
            
            #Convert the list into a numpy array
            XI = np.concatenate(XI_list,axis=0)

            #Calculate the average fit
            XI_average = np.mean(XI, axis=0).reshape(1,-1)

            #Calculate the fits minus the average
            XI_0 = XI - XI_average

            #Add the the aggregation
            XI_0_list.append(XI_0)
            XI_average_list.append(XI_average)


        #Create the numpy array from the aggregate lists
        XI_0_aggregate = np.concatenate(XI_0_list, axis=1)

        #### Calculate the Principal Components
        #Get the number of output components
        k_model_output_size = k_model.get_output_size()

        #Initialize PCA object for vanilla pca calculation
        pca = PCA()
        pca.fit(XI_0_aggregate) 

        #Determine the number of pca components if it is not supplied
        if(num_pca_vectors is None):
            num_pca_vectors = np.count_nonzero(pca.explained_variance_ratio_ <= 0.95)

        #Select the components based on the amount of gait fingerprints
        pmap_aggregate = (pca.components_[:num_pca_vectors,:])

        #Create a list for the personal k models
        k_model_list = []


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
            personalized_k_model = PersonalKModel(k_model,pmap,XI_average,output_name, 
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

    #Save the model so that you can use them later
    def save_model(self,model,filename):
        with open(filename,'wb') as file:
            pickle.dump(model,file)

    #Load the model from a file
    def load_model(self,filename):
        with open(filename,'rb') as file:
            return pickle.load(file)