

#Common imports 
import numpy as np 
import pandas as pd

#Custom imports
from .personal_k_model import PersonalKModel

#Annotation imports
from typing import List


class PersonalMeasurementFunction:
    """
    This class holds multiple personalized kronecker models 
    in order to get the output in a simple manner

    E.g. personalized models will output (num_datapoints,1) from evaluate, 
    this one will output (num_datapoints, num_kmodels)
    """


    def __init__(self, models: List[PersonalKModel], output_names: List[str]):

        #Store reference to all the models
        self.kmodels = models
        self.num_kmodels = len(models)

        #Get the model output 

        #Keep the list of the output names in order
        self.output_names = output_names

        #Have a function to define the number of statest that are taken as the input
        self.num_states = models[0].model.get_num_basis()

        #Have a property for the number of gait fingerprint
        self.num_gait_fingerprint = models[0].num_gait_fingerpints


    def evaluate(self, input_data:np.ndarray, 
            use_personalized_fit:bool = False, 
            use_average_fit:bool=False,
            use_optimal_fit:bool=False):
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
        input_data -- numpy array that contains the model inputs and the gait fingeprints
                   shape (num_datapoints, num_models + num_gait_fingerprints)
                   
        use_personalized_fit -- Boolean if true just use the fit for 
                                the subject that was initialized in 
        use_average_fit -- Boolean, if true just use the average fit


        Returns
        output -- evaluated personalized kronecker model output
                  shape (num_datapoints,num_kmodels)

        Notes: 
        user_personalized_fit takes precedence over use_average_fit
        
        e.g. if user_personalized_fit is true, it does not matter what
        use_average_fit is
        """
    
        #Calculate the number of data points
        num_datapoints = input_data.shape[0]
        
        #Pre-allocate output buffer so that it is faster than concatenating
        output_buffer = np.zeros((num_datapoints, self.num_kmodels))

        #Since all the joint angles use the same model, precalculate the kronecker output
        kronecker_output = self.kmodels[0].get_kronecker_output(input_data)

        #Evaluate every model on the input]
        for i, kmodel in enumerate(self.kmodels):
            
            #Set the corresponding column
            #Reshape to make sure numpy broadcasting rules hold
            output_buffer[:,[i]] = kmodel.evaluate(input_data, 
                                                   use_personalized_fit = use_personalized_fit, 
                                                   use_average_fit = use_average_fit,
                                                   use_optimal_fit = use_optimal_fit,
                                                   kronecker_output=kronecker_output).reshape(num_datapoints,1)

        return output_buffer

