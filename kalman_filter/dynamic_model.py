
#Idea: intialize state with dictionary, but then how to do covariance? double dictionary? That actually makes sense. 

#However that sounds like a pain in the ass to update. However, that can be done internally.

#I guess that the measurement function should also output expected measurements in a dictionary and have it compare to 
# dictionary inputs




#I think that everything should have the key-value pairs assigned to them

#I also think that I should take some time to design this right


#Large features to implement:
## Make everything dictionary based
## This makes sure that there is no fat fingering of inputs where they dont belong.
### Should be easy to implement with functions that convert 2d and 2d dicts and np arrays back and forth between them



import numpy as np
import matplotlib.pyplot as plt


class Dynamic_Model:

	#Initial States is a dictionary that will contain a variable name
	# for every state and the initial value for that state
	def __init__(self):
		pass

	def get_predicted_state(self,current_state,time_step,control_input_u):
		pass



class Gait_Dynamic_Model(Dynamic_Model):

	def __init__(self):
		pass

	#The gait dynamics are an integrator for phase that take into consideration 
	# the phase_dot and the timestep. It is assumed that the phase_dot is the 
	# second element in the state vector
	def f_jacobean(self, current_states, time_step):
		amount_of_states = current_states.shape[0]


		#All the states will stay the same except for the first one
		jacobean = np.eye(amount_of_states)


		#Add the integrator
		jacobean[0][1] = time_step

		return jacobean

	#This is a linar function based on the jacobean its pretty straighforward to calculate
	def f_function(self, current_states, time_step):
		#Essentially we want to calculate dot x = Ax


		jacobean = self.f_jacobean(current_states,time_step)

		return jacobean @ current_states





#This model test does not work anymore

def dynamic_model_unit_test():

	#Set the integration timestep
	time_step = 0.1

	#Set an initial state
	initial_states = np.array([1,1,0,0,0])

	#Get the amount of states
	num_states = initial_states.shape[0]

	#Set the covariance to zero since we are very sure 
	#that the state is where we say that it is
	initial_covariance = np.zeros((num_states,num_states))

	gait_model = Gait_Dynamic_Model()


	state_list = []
	covariance_list = []

	state_list.append(initial_states)
	current_state = initial_states

	covariance_list.append(initial_covariance)
	current_covariance = initial_covariance

	#Iterate to see how the state evolves over time
	for i in range(100):
		current_state, current_covariance = gait_model.get_predicted_state(current_state, current_covariance,time_step)
		state_list.append(current_state)
		covariance_list.append(current_covariance)

	time = np.linspace(0,time_step*100,101)

	state_array = np.array(state_list)
	print(state_array[0,:])
	plt.plot(state_array)
	plt.show()


if __name__ == '__main__':
	dynamic_model_unit_test()