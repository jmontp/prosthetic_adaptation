""" 
This file is meant to load in a simple model from a text description of subject 
and joint model
"""



import pickle
import pathlib
from typing import List

#My janky path solution
from .context import model_definition, model_fitting
from model_definition.k_model import KroneckerModel
from model_definition.fitted_model import SimpleFitModel, PersonalKModel
from model_definition.personal_measurement_function import PersonalMeasurementFunction
from model_fitting.k_model_fitting import KModelFitter
 
import numpy as np 
import pandas as pd

#Import PCA library
from sklearn.decomposition import PCA

#Make a definition so that the users know how to access the average fit
AVG_FIT = "AVG"


def get_subject_number(subject_name:str):
    """
    This function is meant to extract the number part of a subject
    
    E.g., it extracts "2" from "AB02"
    
    """
    #Try to parse out the number for the string
    try:
        model_fit_number = int(subject_name[2:])-1
    
    except ValueError:
        raise ValueError("Did not input the subject name correctly: "
                            "Try AB0X for subject X or 'AVG' for" 
                            "the average model fit'")
        
    return model_fit_number



def load_simple_models(model_output_name:str,
                       subject_name:str,
                       leave_subject_out:str=None) -> SimpleFitModel: 
    
    """ 
    This function loads in a simple fit model for the subject calculated 
    and joint model that are specified. Requires that 
    "calculate_optimal_fits.py" has been run already
    
    
    Keyword Arguments:
    model_output_name - the model name that you want to use. Uses the same
        naming convention as the 
        
    subject_name: subject name. E.g. "AB01" for subject 1. 
        "AVG" for average subject
        
    leave_subject_out: used for cross-model validation. Leaves out one model 
        in the average model calculation
    
    Returns
    simple fit model with that data
    """
    
    
    #Load in the saved model parameters
    current_file_path = str(pathlib.Path(__file__).parent.resolve()) + "/"
    save_location = current_file_path + '../../data/optimal_model_fits/' 
    save_file_name = save_location + model_output_name + "_optimal.p"

    #Load file
    with open(save_file_name,'rb') as data_file:
        fit_results = pickle.load(data_file)
    
    #Get the model fit list
    model_fit_list = fit_results["model fits"]
    
    #Manage the average fit case
    if (subject_name == "AVG"):
        
        if leave_subject_out is not None:
            #Get the leave subject out number 
            leave_out_number = get_subject_number(leave_subject_out)
            #Remove that element from the list before calculating the average
            model_fit_list.pop(leave_out_number)
        
        #Calculate the mean model fit
        model_fit = np.mean(model_fit_list, axis=0)
        
    #If its not the average, it should be a subject number
    else:
        
        #Get the integer number from the string name
        model_fit_number = get_subject_number(subject_name)
        
        #Get the model fit for the subject
        model_fit = model_fit_list[model_fit_number]
    
    
    #Load in the model basis list
    function_basis_list = fit_results["basis list"]
    
    #Create the simple model fit object
    simple_fit_object = SimpleFitModel(function_basis_list, 
                                       model_fit, 
                                       model_output_name)
    
    return simple_fit_object


#Fix the number of pca vectors
num_pca_vectors = 2


def load_personalized_models(model_output_name_list:List[str],
                             subject_name:str,
                             subject_data:pd.DataFrame = None,
                             normalized_pca:bool = True
                            ) -> PersonalMeasurementFunction:
   
    """
    This function is meant to load and possibly train a personalized model 

    Args
    
    model_output_name_list: This is a list of the output names
    
    subject: Subject name
    
    subject_data: This is the subject data to fit the model. It is optional 
        and if it is not supplied then the model will not be fit and it 
        will not have the personalized fit. 
    """
   
    #Get the integer number from the string name
    subject_number = get_subject_number(subject_name)    
    
    
    ##Load in all the saved model information
    fit_info_per_joint = []
    
    #Append a fit info file for each of the model output names
    for model_output_name in model_output_name_list:
    
        #Load in the saved model parameters
        current_file_path = str(pathlib.Path(__file__).parent.resolve()) + "/"
        save_location = current_file_path + '../../data/optimal_model_fits/' 
        save_file_name = save_location + model_output_name + "_optimal.p"
        
        #Load file
        with open(save_file_name,'rb') as data_file:
            fit_results = pickle.load(data_file)
        
        #Add the fit information to the list    
        fit_info_per_joint.append(fit_results)
            
    
    #Get a list of model fits
    model_fits_per_joint = [saved_info['model fits'] 
                            for saved_info 
                            in fit_info_per_joint]
    
    
    #Create list with model fits and remove fits from 
    # fits_excluding_desired_subject in one line
    subject_optimal_fits = [subject_fit_list.pop(subject_number) 
                            for subject_fit_list 
                            in model_fits_per_joint]
    
    ## Calculate the average fit
    average_fit_per_joint_model = [np.mean(model_fit_list,axis=0)
                             for model_fit_list in 
                             model_fits_per_joint]
    
    #Transpose all the fits
    average_fit_per_joint_model_T = [avg_fit.T
                                for avg_fit 
                                in average_fit_per_joint_model]
    
    #Concatenate per subject
    XI_per_joint_np = [np.concatenate(model_fit_list,axis=0).T 
                       for model_fit_list 
                       in model_fits_per_joint]
    
    #Substract the average vector per joint
    XI_0_per_joint = [(XI - XI_avg ).T
                      for XI, XI_avg 
                      in zip(XI_per_joint_np, average_fit_per_joint_model_T) ]
    
    #Create a list of pca objects that we later fit to 
    pca_fit_list = [PCA() for i in range(len(XI_0_per_joint))]
    
    
    #Calculate the personalization map based on our stuff
    if normalized_pca == True:
        
        #Calculate the gramian list by dividing the RTR over the number of
        # datapoints
        G_list_per_joint = []
        
        #Iterate through all the saved joints
        for save_info in fit_info_per_joint:
            #Set the accumulator to zero
            G_for_joint = 0 
            #Remove the cross validation subject
            save_info['RTR list'].pop(subject_number)
            save_info['num datapoints list'].pop(subject_number)
            
            #Iterate through all the RTR lists to normalize by the number 
            # of datapoints
            for RTR, num_datapoints in zip(save_info['RTR list'], 
                                           save_info['num datapoints list']):
                G_for_joint += (RTR/num_datapoints)

            #Add the gram matrix to the per joint list
            G_list_per_joint.append(G_for_joint)
            
        #Convert to the orthonormal space that corresponds to rmse error
        XI_0_per_joint = convert_to_orthonormal(XI_0_per_joint, 
                                                G_list_per_joint)
    
    #Fit all the pca objects
    for XI_0, pca_object in zip(XI_0_per_joint, pca_fit_list):
        pca_object.fit(XI_0)
    
    #Calculate the personalization map per person
    personalization_map_per_joint_model =  [pca_object.components_[:num_pca_vectors,:]
                                            for pca_object
                                            in pca_fit_list]
    
    #Once the PCA is done, revert to the normal pca
    if normalized_pca == True:
        personalization_map_per_joint_model = convert_from_orthonormal(
            personalization_map_per_joint_model, G_list_per_joint
        )
    
    
    #To do least squares, we need to calculate the regressor, to do that we 
    # need a kronecker model
    basis_list = fit_results["basis list"]
    k_model = KroneckerModel(basis_list)
    
    #Get regression matrices
    
    #Initialize accumulator matrices
    RTR_prime_total = 0
    RTy_prime_total = 0
    
    #calculate the least squares, gait-fingerprint regression matrices
    for output_name,pmap,avg_fit in zip(model_output_name_list, 
                                        personalization_map_per_joint_model, 
                                        average_fit_per_joint_model):
    
        RTR_prime, RTy_prime = calculate_gait_fingerprint_regressor(k_model, 
                                                    subject_data, 
                                                    output_name, 
                                                    pmap,
                                                    avg_fit)
        #Add tho the solution matrix
        RTR_prime_total += RTR_prime
        RTy_prime_total += RTy_prime
    
    #Calculate the gait fingerprint
    gait_fingerprint = np.linalg.solve(RTR_prime_total, RTy_prime_total).T
    print(f"Gait Fingerprint for {subject_name} -- {gait_fingerprint}")
    
    #Create the personalized model
    personalized_model_list = [PersonalKModel(basis_list, 
                                    model_output_name_list[i],
                                    average_fit_per_joint_model[i],
                                    personalization_map_per_joint_model[i],
                                    gait_fingerprint, 
                                    subject_name,
                                    subject_optimal_fits[i])
                                
                                for i
                                in range(len(model_output_name_list))]

    
    #Once all the models are calculated, we can return the model fit object
    model = PersonalMeasurementFunction(personalized_model_list,
                                        model_output_name_list, 
                                        subject_name)
    
    return model




def calculate_gait_fingerprint_regressor(k_model: KroneckerModel,
                                         data: list, 
                                         output_name: str,
                                         pmap: np.ndarray, 
                                         average_fit: np.ndarray): 
    """
    This function will calcualte the gait fingerprint fit based on
    the previous models

    Keyword Arguments: 
    k_model -- kronecker model object to perform the fit into
    subject_data_list -- list with tuples of the form 
        (subject_name,pandas dataframe)
    output_name -- name of the output variable that will be used to fit
    data -- pandas dataframe with the data to perform the fit 
    output_name -- column in data that will be used as the 'y' data

    Returns:
    RTR_prime -- gait fingerprint regressor matrix
    RTy_prime -- gait fingerprint regressor matrix
    """

    #Create a model fitter object
    data_fitter = KModelFitter()

    #Get regression matrices
    RTR, RTy, yTR, yTy = \
        data_fitter.calculate_regressor(k_model, data, output_name)
    
    #Use the personalization map to calculate the components of 
    # the least squares equation
    RTR_prime = (pmap @ RTR @ pmap.T)
    RTy_prime = (pmap @ RTy) - (pmap @ RTR @ average_fit.T)
    yTR_prime = RTy_prime.T
    yTy_prime = yTy - yTR @ average_fit.T - average_fit @ RTy\
        - average_fit @ RTR @ average_fit.T


    return RTR_prime, RTy_prime


def convert_to_orthonormal(XI_0_list:List[np.ndarray], 
                           G_list:List[np.ndarray]) -> List[np.ndarray]:
    """
    This function is meant to convert vectors to the orthonormal space 
    that represents deviations from the average. 

    Keyword Arguments:
    XI_0_aggregate: list of model fits in numpy array format

    Returns
    XI_0_scaled_aggregates: numpy array with the model fits in the new basis
    """
    #Split the XI_list into multiple smaller numpy arrays corresponding to every output 
    # num_outputs = XI_0_np.shape[1]/G_list[0].shape[0]

    # #Create a list of numpy arrays based on splitting the input array 
    # XI_0_list = np.split(XI_0_np,num_outputs,axis=1)

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
    # XI_0_scaled_aggregate = np.concatenate(XI_0_scaled_list, axis=1)

    return XI_0_scaled_list

def convert_from_orthonormal(XI_0_list:List[np.ndarray], 
                             G_list:List[np.ndarray]) -> List[np.ndarray]:
    """
    This function is meant to convert vectors to the orthonormal space 
    that represents deviations from the average. 

    Keyword Arguments:
    XI_0_scaled_aggregate: list of model fits in numpy array format in the orthonormal basis
    
    Returns: 
    XI_0_aggregate: list of model fits in numpy array format in the original basis
    """

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
    # XI_0_aggregate = np.concatenate(XI_0_original_list, axis=1)

    return XI_0_original_list
