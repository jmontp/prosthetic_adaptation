"""
This file creates the fit model subclass with two implementations
"""
#Base imports
from typing import List
from dataclasses import dataclass
#Common imports
import numpy as np

#Custom imports
from k_model import KroneckerModel
from function_bases import Basis

class FitModel(KroneckerModel):
    """
    This class is meant to extend the kronecker model base class to have a 
    model fit that allows it to otuput a scalar from evaluate instead of a 
    vector
    """

    def __init__(self, basis_list:List[Basis],
                       output_name:str):

        #Call parent class constructor
        super().__init__(basis_list)

        #Initialize internal objects
        self.output_name = output_name


    def evaluate(self, input_data:np.ndarray, eval_cond=None) -> np.ndarray:
        """
        This is an abstract method that evaluates into a scalar
        """

        raise NotImplementedError("Please implement this method")


    def get_kronecker_output(self,input_data:np.ndarray):
        """
        This function is meant to allow access to the vector output of the 
        kronecker model

        Keyword Arguments:
        input_data: numpy array with shape (num_basis, num_datapoints)

        Returns
        output: numpy array with shape (num_datapoints, 1)
        """
        return self.evaluate(input_data)




@dataclass
class FitInformation:
    """
    This class is meant to store the fit information that can be 
    useful later on
    """
    def __init__(self,l2_lambda:float,
                 average_model_residual:float,
                 diff_from_average_matrix:np.ndarray):
        
        self.l2_lambda = l2_lambda
        self.average_model_residual = average_model_residual
        self.diff_from_average_matrix = diff_from_average_matrix



class SimpleFitModel(FitModel):
    """
    This class is meant to be a simple, concrete implementation of the FitModel
    class with the only having one model fit
    """

    def __init__(self, basis_list:List[Basis],
                        model_fit:np.ndarray,
                        output_name:str):
        
        #Call the parent constructor
        super().__init__(basis_list,  output_name)

        #Store the model fit inside the object
        self.model_fit = model_fit
        


    def evaluate(self,
                input_data:np.ndarray,
                eval_cond=None) -> np.ndarray:
        """
        Returns a scalar based on the kronecker model and the model fit

        Keyword Arguments:
        input_data: input data with shape (num_datapoints, num_basis)

        eval_cond: Has no effect, left in for compatibility with base class
        
        Returns
        output: np array with shape (num_datapoints, 1)
        """

        #Get the evaludated row vector
        kronecker_output = self.get_kronecker_output(input_data)

        #Shape (num_datapoints, 1, model_size)
        kronecker_output = kronecker_output[:,np.newaxis,:]
        #Shape (num_datapoints, model_size, 1)
        model_fit = self.model_fit[:,:,np.newaxis]

        #Calculate the output
        # Needs to have shape (num_datapoints, 1)
        output = (kronecker_output @ model_fit).reshape(-1,1)
        
        return output



class PersonalKModel(FitModel):

    """
    This class holds a reference to many different model fits that 
    can be changed in run time
    """
    #Create enumeration for conditions in evaluate
    EVAL_AVERAGE_FIT = 0
    EVAL_OPTIMAL_FIT = 1
    EVAL_GF_FIT = 2

    def __init__(self, basis_list: List[Basis],
                 output_name:str,
                 average_fit: np.ndarray,
                 personalization_map: np.ndarray,
                 gait_fingerprint: np.ndarray = None,
                 subject_name: np.ndarray = None,
                 optimal_fit:np.ndarray = None,
                 fit_information:FitInformation=None):
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
        average_model_residual -- stores the average model residual across
                                  different subjects
        diff_from_average_matrix -- stores the matrix used to generate the
                                    personalization map
        optimal_fit -- this is the optimal fit for a subject using their data
                        and least squares
        """
        
        #Call the parent constructor
        super().__init__(basis_list, output_name)

        #Get the personalization_map
        self.pmap = personalization_map
        self.num_gait_fingerpints = personalization_map.shape[0]

        #If we have the personalized fit, add it as well
        self.average_fit = average_fit

        #Store the optimal model
        self.optimal_fit = optimal_fit
        
        #Personalized fit for one person
        # Store the name so you can keep track of who's fit it is
        self.subject_gait_fingerprint = gait_fingerprint
        self.subject_name = subject_name

        #Keep the output name
        self.output_name = output_name

        #Fit information
        self.fit_information = fit_information

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

    def get_kronecker_output(self,input_data):
        return self.model.evaluate(input_data)


    def evaluate(self,input_data:np.ndarray,
                 use_personalized_fit:bool =False,
                 use_average_fit:bool=False,
                 kronecker_output:bool=None,
                 use_optimal_fit:bool=False):
        """
        Evaluate a Kronecker Model multiplied by
            the (average fit plus personalization map times gait fingerprint)
            which gives a scalar output

        In the default behavior, it will use the gait fingerprint in the
        input_dataset, which is a numpy array that has a column corresponding
        to each model input and then the gait fingerprints
        
        The subject-average fit can also be used by setting use_average_fit to
        true

        If the model was initialized with a subject gait fingerprint,
        use_personalized_fit will override the gait fingerprint in the dataset
        for the least squares optimal gait fingerprint for that subject

        Keyword arguments:
        dataset -- numpy array that contains the model inputs and the gait
                   fingeprints.
                   shape (num_datapoints, num_models + num_gait_fingerprints)
                   
        use_personalized_fit -- Boolean if true just use the fit for
                                the subject that was initialized in
        use_average_fit -- Boolean, if true just use the average fit
        kronecker_output -- kronecker model output can be feed in externally
                            to speed up calculations
        use_optimal_fit -- Booolean, if true it uses the optimal fit calculated
                           with least squares
        Returns
        output -- evaluated personalized kronecker model output
                  shape (num_datapoints,1)

        Notes:
        Order of precedence is as follows:
        user_personalized_fit > use_average_fit > use_optimal_fit
        
        e.g. if user_personalized_fit is true, it does not matter what
        use_average_fit is
        """

        #Calculate the output of the kronecker model
        # shape(num_datapoints, model_output_size)
        if kronecker_output is None:
            kronecker_output = self.model.evaluate(input_data)

        #Default behaviour, use the gait fingerprints from the dataset
        #The gait fingeprints will be after the model variables
        gait_fingerprints = input_data[:, self.num_models:]

        #Get the gait_fingerprints
        if(use_personalized_fit is True):

            #Return error if the object was not initialized 
            if(self.subject_gait_fingerprint is None):
                raise AttributeError("Object was not initialized \
                            with a personalized_fit parameter")
            #Use the gait fingerprint stored
            gait_fingerprints = self.subject_gait_fingerprint

            #Calculate the model fit vector
            model_fit_vector = self.average_fit + \
                                 gait_fingerprints @ self.pmap

        #Don't do any personalization and use the average fit
        elif(use_average_fit is True):
            
            #Use stored average fit as the model fit vector
            model_fit_vector = self.average_fit

        #Don't do any personalization and use the optimal fit
        elif(use_optimal_fit is True):
            
            #Use the optimal fit as the fit vetor
            model_fit_vector = self.optimal_fit
        
        #Use the gait fingerprint in the input data
        else:

            #Use the gait fingerprint with the personalization map
            model_fit_vector = self.average_fit + \
                                 gait_fingerprints @ self.pmap

       
        ## Reshape in order to get the output of the shape (num_datapoints, 1)
        #Shape (num_datapoints, 1, model_size)
        kronecker_output = kronecker_output[:,np.newaxis,:]
        #Shape (num_datapoints, model_size, 1)
        personalization_vector = model_fit_vector[:,:,np.newaxis]

        #Output is kronecker_output multiplied with personalization vector
        # shape(num_datapoints, 1)
        output = (kronecker_output @ personalization_vector).reshape(-1,1)

        return output


