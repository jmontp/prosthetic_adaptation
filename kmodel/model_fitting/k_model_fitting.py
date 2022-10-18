
#Common imports
import pandas as pd
import numpy as np

#For docstring
from typing import List, Union
from dataclasses import dataclass

#Kronecker model imports
from .context import model_definition
from model_definition.k_model import KroneckerModel
from model_definition.function_bases import Basis
import pandas as pd

class KModelFitter():

    """
    This class defines all the methods to fit a k model to a dataset
    """

    def __init__(self):
        pass


    def fit_data(self,k_model : KroneckerModel,data : pd.DataFrame, \
                     output_name: str,  data_splits : int = 50, \
                     l2_lambda: float = 0.0,\
                     weight_col:str = None): 
        """
        This is the least squares implementation to fit data 
        to a specific model

        e.g. solve Rx = y for x
        
        Keyword Arguments:
        k_model -- KroneckerModel object, or any object with evaluate method
        data -- pandas dataframe with the data to perform the fit 
        output_name -- column in data that will be used as the 'y' data
        data_splits -- scalar that indicates how many times to sub-divide the data.
                        Small values are make the fit run faster, but use more memory.
        l2_lambda = lambda in l2 regularization
        weight_col = column that stores the weight information from weighted
            least squares
        
        Returns:
        model_fit -- best fit of the model to the data with 
                     shape(1,k_model_output_size)
        model_residual -- residual of the model with respect to the data

        Throws: 
        Exception when the R cannot be inverted
        """
        
        #Get the regressor matrix
        RTR, RTy, yTR, yTy = self.calculate_regressor(k_model, 
                                                      data, 
                                                      output_name, 
                                                      data_splits, 
                                                      weight_col)
    
        #Calculate the least squares fit
        # x = (R^T R - lambda*I)^-1 R^T y
        x = np.linalg.solve(RTR + l2_lambda * np.eye(RTR.shape[0]),RTy).T

        #Debug shapes
        #print(f"RTR: {RTR.shape} yTR {yTR.shape} RTy: {RTy.shape} yTy: {yTy.shape} x {x.shape}")

        #Calculate the number of datapoints 
        num_datapoints = len(data.index)


        #residual = sqrt((x^T R^T Rx - x^T R^T y - y^T Rx + y^T y)/num_datapoints)
        residual = np.sqrt((x @ RTR @ x.T - x @ RTy - yTR @ x.T + yTy)/num_datapoints)


        return x, (residual, RTR, num_datapoints)


    
    def calculate_regressor(self,k_model : KroneckerModel, data : pd.DataFrame\
                            ,output_name : str,  data_splits : int = 50 \
                            ,weight_col : str = None): 
        """
        Calcualte the regressor matrices to calculate least squares

        E.g. if you have y = Rx, it calculates
        R^T * R, R^T * y, y^T * R, y^T * y
        
        Keyword Arguments:
        k_model -- KroneckerModel object, or any object with evaluate method
        data -- pandas dataframe with the data to perform the fit 
        output_name -- column in data that will be used as the 'y' data
        data_splits -- scalar that indicates how many times to sub-divide the data.
                        Small values are make the fit run faster, but use more memory.
        weight_col = column that stores the weight information from weighted
            least squares
            
        Returns:
        model_fit -- best fit of the model to the data with 
                     shape(1,k_model_output_size)
        model_residual -- residual of the model with respect to the data

        Throws: 
        Exception when the R cannot be inverted
        """
        #If the weight column is not defined, initialize as list of all ones
        if weight_col is None:
            W_total = np.ones((len(data),1))
            
        #If it is defined, just get it from the dataset
        else:
            W_total = data[weight_col].values.reshape(-1,1)
            
        #Get the size of the model
        k_model_size = k_model.get_output_size()

        #Assumes that we are solving for x in Rx = y
        # R - k_model evaluated at corresponding states
        # y - output variable
        # R has too many rows to invert directly, therefore, another method is used
        # To solve least squares, we need x = (R^T R)^-1 R^T y
        # Therefore we need R^T R and R^T y
        RTR = np.zeros((k_model_size, k_model_size))
        yTR = np.zeros((1,k_model_size))

        # In addition, the residual can be calculated by 
        # sqrt((x^T R^T Rx - x^T R^T y - y^T Rx + y^T y)/num_datapoints)
        # Therefore, store RTy and y^T y and get num_datapoints
        RTy = np.zeros((k_model_size,1))
        yTy = np.zeros((1,1))

        #Divide the dataframe into multiple, smaller dataframes 
        # to calculate the desired matrices for the fit
        data_weight_list = zip(np.array_split(data, data_splits),
                               np.array_split(W_total, data_splits))
        
        #Do sub dataframe updates
        for sub_datafrmae, W in data_weight_list:
            
            #Get the regressor matrix
            # shape(sub_datapoints, k_model_output_size)
            R = k_model.evaluate(sub_datafrmae)

            #Get the expected output as 2D
            # shape(sub_datapoints, 1)
            y = sub_datafrmae[output_name].values.reshape(-1,1)

            #Calculate the rank update
            RTR += R.T @ (R * W)
            yTR += y.T @ (R * W)
            RTy += R.T @ (y * W)
            if np.isnan(RTy).sum() > 0:
                raise ValueError
            
            yTy += y.T @ (y * W)

        return RTR, RTy, yTR, yTy

    
    
    
# @dataclass(repr=True)
# class FitInfo:
#     """
#     This class is meant to store the fit information that can be 
#     useful later on
#     """
#     #List of the model fit for each subject
#     model_fit_list:List[np.array]
#     #List of the regressor transpose regressor for each person
#     RTR_list:List[np.array]
#     #List of the residuals for each person
#     residual_list:List[float]
#     #Number of datapoinst for each person
#     num_datapoints_list:List[int]
#     #Basis list for the kronecker model
#     basis_list:List[Basis]
#     #L2 regularization for the model fitting
#     l2_regularization: Union[float,List[float]]
