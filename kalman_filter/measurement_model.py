import numpy as np

##LOOK HERE 
##There is a big mess with how the measurement model is storing the gait fingerprint coefficients
##They should really just be part of the state vector, the AXIS should be stored internally since that is 
## fixed


#Measurement Model = []



class Measurement_Model():
	def __init__(self,*models):
		self.models = models

	def evaluate_h_func(self,*states):
		#get the output
		result = [model.evaluate_scalar_output(*states) for model in self.models]
		return np.array(result)



	def evaluate_dh_func(self,*states):
		result = []
		for model in self.models:
			state_derivatives = [model.evaluate_scalar_output(*states,partial_derivative=func.var_name) for func in model.funcs]
			gait_fingerprint_derivatives = [model.evaluate(*states)@axis for axis in model.pca_axis]
			total_derivatives = state_derivatives + gait_fingerprint_derivatives
			result.append(total_derivatives)

		return np.array(result)

