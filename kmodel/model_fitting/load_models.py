""" 
This file is meant to load in a simple model from a text description of subject 
and joint model
"""



import pickle
import pathlib
from typing import List

#My janky path solution
from .context import model_definition
from model_definition.fitted_model import SimpleFitModel, PersonalKModel
import numpy as np 


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






def load_personalized_models(model_output_name_list:List[str],
                             subject_name:str) -> PersonalKModel:
   
    #Get the integer number from the string name
    subject_number = get_subject_number(subject_name)    
    
    
    ##Load in all the saved model information
    model_saved_info_list = []
    
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
        model_saved_info_list.append(fit_results)
            
    
    #Get a list of model fits
    model_subject_fit_list = [saved_info['model fits'] 
                              for saved_info 
                              in model_saved_info_list]
    
    
    ## Calculate the average fit
    average_fit_per_model = [np.mean(subject_fit_list,axis=0)
                             for subject_fit_list in 
                             model_subject_fit_list]
    
    #Transpose all the fits
    model_subject_fit_list = [subject_fit.T
                             for model_subject_fit in subject_optimal_fits 
                                for subject_fit in model_subject_fit_list ]
    
    #Create a copy of the fits without removing subject just yet
    # This is the fastest way to create a copy of each sublist
    fits_excluding_desired_subject = [i[:] for i in model_subject_fit_list]
    
    #Create list with model fits and remove fits from 
    # fits_excluding_desired_subject in one line
    subject_optimal_fits = [subject_fit_list.pop(subject_number) 
                            for subject_fit_list 
                            in fits_excluding_desired_subject]
    
    #Concatenate per subject
    
    
    pass