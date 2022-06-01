"""
This class creates a personalized K-model instance 
"""

#Common imports
from .k_model import KroneckerModel
import numpy as np
import pandas as pd

#Import kronecker model for annotation purposes

class PersonalKModel():
    """This class holds a reference to a kronecker model and 
        a personalization map"""

    def __init__(self, kronecker_model: KroneckerModel, personalization_map: np.ndarray, average_fit: np.ndarray, output_name: str,
                 gait_fingerprint: np.ndarray = None, 
                 subject_name: np.ndarray = None,
                 average_model_residual:np.ndarray = None,
                 l2_lambda:float = None):
        """
        Initialize the PersonalKModel Object

        Keyword arguments:
        kronecker_model -- KroneckerModel object
        personalization_map -- Numpy array with shape 
                              (num_gait_fingerprints, model_output_size)
        average_fit -- Numpy array with 
                       shape (1, model_output_size)
        gait_fingerprint -- an individual's optimized gait fingerprint with 
                            shape(1,model_output_size)
        subject_name -- string of the subject name

        average_model_residual -- stores the average model residual across different subjects
        """
        #Get the model and the amount of functions
        self.model = kronecker_model
        self.num_models = kronecker_model.get_num_basis()

        #Get the personalization_map
        self.pmap = personalization_map
        self.num_gait_fingerpints = personalization_map.shape[0]
    
        #If we have the personalized fit, add it as well
        self.average_fit = average_fit
        
        #Personalized fit for one person
        # Store the name so you can keep track of who's fit it is
        self.subject_gait_fingerprint = gait_fingerprint
        self.subject_name = subject_name

        #Keep the output name
        self.output_name = output_name

        #Store the output rmse
        self.avg_model_residual = average_model_residual

        #Store the l2 lambda that was used to generate this model
        self.l2_lambda = l2_lambda

    def get_subject_name(self):
        """
        Returns the subjects name, if any

        This is useful is the instance to this object is stored
        """
        return self.subject_name

    def set_pmap(self, new_pmap):
        """
        Set the personalization map of the model
        """
        self.pmap = new_pmap


    def evaluate(self,input_data,use_personalized_fit=False, use_average_fit=False):
        """
        Evaluate a Kronecker Model multiplied by 
            the (average fit plus personalization map times gait fingerprint) 
            which gives a scalar output

        In the default behavior, it will use the gait fingerprint in the input_dataset,
        which is a numpy array that has a column corresponding to each model input and 
        then the gait fingerprints
        
        The subject-average fit can also be used by setting use_average_fit to true

        If the model was initialized with a subject gait fingerprint, use_personalized_fit
        will override the gait fingerprint in the dataset for the least squares optimal gait fingerprint
        for that subject

        Keyword arguments:
        dataset -- numpy array that contains the model inputs and the gait fingeprints
                   shape (num_datapoints, num_models + num_gait_fingerprints)
                   
        use_personalized_fit -- Boolean if true just use the fit for 
                                the subject that was initialized in 
        use_average_fit -- Boolean, if true just use the average fit


        Returns
        output -- evaluated personalized kronecker model output
                  shape (num_datapoints,1)

        Notes: 
        user_personalized_fit takes precedence over use_average_fit
        
        e.g. if user_personalized_fit is true, it does not matter what
        use_average_fit is
        """

        #Calculate the output of the kronecker model
        # shape(num_datapoints, model_output_size)
        kronecker_output = self.model.evaluate(input_data)

        #Default behaviour, use the gait fingerprints from the dataset
        #The gait fingeprints will be after the model variables
        gait_fingerprints = input_data[:, self.num_models:]

        #Get the gait_fingerprints
        if(use_personalized_fit == True):

            #Return error if the object was not initialized 
            if(self.subject_gait_fingerprint is None):
                raise TypeError("Object was not initialized \
                            with a personalized_fit parameter")

            gait_fingerprints = self.subject_gait_fingerprint

        #Don't do any personalization and just do average fit
        elif(use_average_fit == True):
            #Get the number of datapoints
            num_datapoints = input_data.shape[0]
            
            #To not include personalization, just set the gait fingerprints 
            # to zero
            gait_fingerprints = np.zeros((num_datapoints,self.num_gait_fingerpints))

        #Get the personalized vector
        # shape (num_datapoints, output_model_size)
        personalization_vector = self.average_fit + gait_fingerprints @ self.pmap

        ## Reshape in order to get the output of the shape (num_datapoints, 1)

        #Shape (num_datapoints, 1, model_size)
        kronecker_output = kronecker_output[:,np.newaxis,:]
        #Shape (num_datapoints, model_size, 1)
        personalization_vector = personalization_vector[:,:,np.newaxis]

        #Output is kronecker_output multiplied with personalization vector
        # shape(num_datapoints, 1)
        output = (kronecker_output @ personalization_vector).reshape(-1,1)

        return output


