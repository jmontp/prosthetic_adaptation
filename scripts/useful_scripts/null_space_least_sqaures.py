"""
This module is meant to calculate the nullspace projection of the
personalization map so that the dot product of the derivatives of the
joint angle with respect to each state is not in conflict with the
gait fingerprint
"""


#Common imports
import numpy as np
import pandas as pd
import itertools
from sklearn.decomposition import PCA
from sympy import product


#Custom imports
from context import kmodel
from kmodel.function_bases import FourierBasis, PolynomialBasis
from kmodel.k_model import KroneckerModel
from kmodel.personalized_model_factory import PersonalizedKModelFactory

##########
## Create the kronecker model

#Define the subject
subject_list = [f"AB{i:02}" for i in range(1,11)]

#Import the personalized model 
factory = PersonalizedKModelFactory()

#Iterate through all the subjects
for subject in subject_list:
    
    print(f"Doing subject {subject}")

    #Path to model
    model_dir = (f'../../data/kronecker_models/'
                f'left_one_out_model_{subject}.pickle')

    #Load model from disk
    model = factory.load_model(model_dir)

    #Set the number of gait fingerprints
    num_gait_fingerprints = model.num_gait_fingerprint
    num_models = len(model.output_names)

    #Update the personalization map for each of the output models
    for personal_kronecker_model_i in model.kmodels:

        #Create the module
        kronecker_model = personal_kronecker_model_i.model

        #Get the size of the model
        model_size = kronecker_model.output_size

        #Get the inter-subject average fit
        average_fit = personal_kronecker_model_i.average_fit

        #Get the difference from the average matrix from the model
        djff_from_avg = personal_kronecker_model_i.diff_from_average_matrix

        #Get the number of gait fingerprints
        NUM_GF = personal_kronecker_model_i.num_gait_fingerpints

        ##########
        ## Create constraints for all the states

        #Set the constraints for ramp for the nullspace projection
        ramp_constraints = np.array([-10,-7.5,-5,0,5,7.5,10])

        #Get equidistant points in stride length
        stride_length_constraints = np.array([0.8,1.0,1.2,1.4])

        #Set the constraints for ramp
        # phase_dot_constraints = np.array([1.0,1.4])

        #Create list with all the constraints, make sure they are in the same
        # order that the kronecker product expects the arguments
        constraint_list = [#phase_dot_constraints,
                           stride_length_constraints,
                           ramp_constraints
                           ]       

        #Get all the states we will take partials for
        NUM_STATES_PARTIAL = len(constraint_list) + 1

        #Get the number of constraints
        #product of number of each constrait times the number of states
        COMB_NO_PARTIAL = np.prod([c.shape[0] for c in constraint_list])

        num_constraints = COMB_NO_PARTIAL * (NUM_STATES_PARTIAL)


        print(f"    Doing model {personal_kronecker_model_i.output_name}")

        #Make sure the user is aware of the relative size of constraints to
        # model fit size. It should always be smaller
        print((f"Total constraints {num_constraints}"
            f" size of vector {model_size}"))

        assert num_constraints < model_size

        #Num constraint variables
        NUM_CONSTRAINT_VAR = len(constraint_list)

        #Create an array with all the combinations of the constraints
        constraint_combinations = np.stack(np.meshgrid(*constraint_list),-1)\
                                    .reshape(-1,NUM_CONSTRAINT_VAR)
        
        #Have to add a phase dot just so it evaluates properly
        #Since we don't have phase in yet, phase dot is the first element
        #This model actually does not use phase dot, therefore just set it
        #to a random value
        PHASE_DOT_INDEX = 0
        constraint_combinations = np.insert(constraint_combinations\
                                            ,PHASE_DOT_INDEX,0.5,axis=1)

        #Set the constraints for phase
        NUM_PHASE_POINTS = 100
        phase_constraints_integration = np.linspace(0,1,NUM_PHASE_POINTS)\
                                          .reshape(-1,1)

        #Create a storage for the integral aggregator
        M = np.zeros((num_constraints,model_size))

        #Iterate through all the combinations
        for comb_row_i in range(constraint_combinations.shape[0]):

            #Repeat the combination vector for each instance of phase
            comb_row_repeat = np.repeat(constraint_combinations[[comb_row_i],:]
                                          , NUM_PHASE_POINTS, axis=0)

            #Create the constrait vector with all the phases required for 
            # numerical integration. 
            constraint_combinations_phase =  \
                np.concatenate([phase_constraints_integration,comb_row_repeat],axis=1)

            #Iterate through all the constraints with phase added in
            for phase_comb_i in range(NUM_PHASE_POINTS):    
                
                #Get the current constraint
                curr_const = constraint_combinations_phase[[phase_comb_i],:]

                #Calculate the model output
                A = kronecker_model.evaluate(curr_const)

                #Calculate the model output with a derivative
                A_diff = kronecker_model.evaluate_derivative(curr_const)

                #For every combination row there are several partials that are
                #intervewaved in
                M_i_offset = comb_row_i*NUM_STATES_PARTIAL

                #We get the partial derivatives in A, have to index it out
                for state_i in range(NUM_STATES_PARTIAL):    
                
                    #Partial derivative of Ith state
                    A_diff_state_i = A_diff[state_i,:]

                    #Add to the integral aggregator
                    M[M_i_offset + state_i,:] += \
                        average_fit @ A_diff_state_i.T @ A / NUM_PHASE_POINTS

            #Print rank just to make sure it goes up to full rank to invert
            print(f"M rank {np.linalg.matrix_rank(M)}")

        #Get the change of basis into the new shape
        M_null = np.eye(M.shape[1]) - M.T @ np.linalg.inv(M @ M.T) @ M

        #Calculate the difference from the average
        djff_from_avg_new_basis = (djff_from_avg @ M_null)

        #Initialize pca fitter
        pca_fitter = PCA(n_components=NUM_GF)

        #Calculate the personalization map in the nullspace
        pca_fitter.fit(djff_from_avg_new_basis)

        #Convert to numpy array
        pmap_null = pca_fitter.components_[:num_gait_fingerprints,:]
        #Set the pmap to the model
        personal_kronecker_model_i.set_pmap(pmap_null)

        #Validate that the personalization map components are in the nullspace
        # of M
        pmap_projection_norm = np.linalg.norm(pmap_null @ M)

        assert pmap_projection_norm < 1e-7

    #Create the save location with slightly different name
    model_save_dir = (f'../../data/kronecker_models/'
                    f'left_one_out_model_{subject}_null.pickle')

    #Save the model
    factory.save_model(model,model_save_dir)
    print(f"     saved {subject} in {model_save_dir}")