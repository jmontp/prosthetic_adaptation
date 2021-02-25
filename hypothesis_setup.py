
from data_generators import get_subject_names
from model_framework import Fourier_Basis, Polynomial_Basis, Bernstein_Basis, Kronecker_Model
from model_fitting import generate_regression_matrices, least_squares_r


#Get the joint that we want to generate regressors for
joint = 'hip'
#Get the names of the subjects
subjects = get_subject_names()


#Initialize the model that we are going to base the regressor on
phase_model = Fourier_Basis(4,'phase')
phase_dot_model = Polynomial_Basis(3, 'phase_dot')
ramp_model = Polynomial_Basis(3, 'ramp')
step_length_model = Polynomial_Basis(3,'step_length')

model_hip = Kronecker_Model(phase_dot_model, ramp_model, step_length_model,phase_model)


R = generate_regression_matrices(model_hip, subjects, joint)

