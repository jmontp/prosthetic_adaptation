"""
This file is meant to generate a helper function to load an ekf
based on the required parameters. This is meant to avoid having to do 
easy instanciations. 

"""
#Normal imports
import numpy as np 

#Type hint imports
from typing import List

#Custom imports
from context import kmodel
from kmodel.model_fitting.load_models import load_simple_models
from kmodel.model_definition.personal_measurement_function import PersonalMeasurementFunction

from context import ekf
from ekf.measurement_model import MeasurementModel
from ekf.dynamic_model import GaitDynamicModel
from ekf.ekf import Extended_Kalman_Filter



def ekf_loader(subject:str,
               joint_list:List[str],
               initial_state:np.ndarray, 
               initial_state_covariance:np.ndarray,
               Q:np.ndarray,
               R:np.ndarray,
               state_lower_limit:List[float],
               state_upper_limit:List[float],
               use_subject_average:bool = False,
               heteroschedastic_model = True
            ):
    
    """
    
    Args:
    subject (str): subject name
    joint_list (List[str]): list of the joints that will be used in the ekf.
        Length = j
    initial_state (np.ndarray): The initial state of the ekf. Shape(n,1)
    initial_covariance (np.ndarra): The initial covariance of the system. 
        Shape(n,n)
    Q (np.ndarray): The process model noise. Shape (n,n)
    R (np.ndarray): The measurement model noise. Shape(j,j)
    state_lower_limit (List[float]): The state lower limit for the ekf
        Length = n
    state_upper_limit (List[float]): The state upper limit for the ekf. 
        Length = n
    us_subject_average (bool): uses the one-left-out subject average model, 
        that leaves out the specified subject. 
    heteroschedastic_model (bool): Uses the measurement model heteroschedastic
    model when set to true.
    
    
    returns
        ekf_instance (Extended_Kalman_Filter)
    """
    
    
    ##Import the personalized model 
    #Load in the average model without the specified subject
    if use_subject_average is True:
        fitted_model_list = [load_simple_models(joint,"AVG",
                                                leave_subject_out=subject) 
                            for joint 
                            in joint_list]
    #Load the subject fit
    else:    
        fitted_model_list = [load_simple_models(joint,subject) 
                            for joint 
                            in joint_list]
    
    #Generate a Personal measurement function 
    model = PersonalMeasurementFunction(fitted_model_list, 
                                        joint_list, subject)
    #Initialize the measurement model
    measurement_model = MeasurementModel(model,
                                         calculate_output_derivative=True)
    # output_model = MeasurementModel(model_output, calculate_output_derivative=False)


    #Get the ground truth from the datasets
    ground_truth_labels = ['phase','phase_dot','stride_length','ramp']
    
    #Initiailze gait dynamic model
    d_model = GaitDynamicModel()

    #Initialize the EKF instance
    ekf_instance = Extended_Kalman_Filter(initial_state, 
                                          initial_state_covariance, 
                                          d_model, Q, 
                                          measurement_model, R,
                                          lower_state_limit=state_lower_limit, 
                                          upper_state_limit=state_upper_limit,
                                          # output_model=output_model,
                                          heteroschedastic_model =\
                                              heteroschedastic_model
                                        )
    
    return ekf_instance