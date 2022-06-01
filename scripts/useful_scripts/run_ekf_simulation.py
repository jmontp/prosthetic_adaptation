#Common imports 
from matplotlib.pyplot import plot
import numpy as np

#Import custom library
from simulate_ekf import simulate_ekf
from context import kmodel
from context import ekf
from context import utils
from kmodel.personalized_model_factory import PersonalizedKModelFactory

#Import loaders
import pickle


#Define the subject
subject_list = [f"AB{i:02}" for i in range(1,11)]

#Import the personalized model 
factory = PersonalizedKModelFactory()

#Path to model
model_dir = f'../../data/kronecker_models/left_one_out_model_{subject_list[0]}.pickle'

#Load model from disk
model = factory.load_model(model_dir)

#Set the number of gait fingerprints
num_gait_fingerprints = model.num_gait_fingerprint
num_models = len(model.output_names)

#Define states
num_states = 3



#Measurement covarience, Innovation
position_noise_scale = 1
velocity_noise_scale = 1 #noise added for derivative terms
residual_power = 2
#Get the joint degree tracking error and then joint velocity measurement error
r_diag = [position_noise_scale*(kmodel.avg_model_residual**residual_power) for kmodel in model.kmodels] + \
         [velocity_noise_scale*(kmodel.avg_model_residual**residual_power) for kmodel in model.kmodels]

R = np.diag(r_diag)
R_scaled = R*0.9

frec_component = position_noise_scale * velocity_noise_scale/6


#Generate the initial covariance as being very low
# try:
#     initial_state_covariance = np.load('pred_cov.npy')
#     if initial_state_covariance.shape[0] != num_states + num_gait_fingerprints:
#         raise Exception

# except Exception:

cov_diag = 5e-3
gf_cov_init = 5e-5
initial_state_diag = [cov_diag]*num_states + [gf_cov_init*1e+6] + [gf_cov_init]*(num_gait_fingerprints-1)
initial_state_covariance = np.diag(initial_state_diag)

#Set state limits
upper_limits = np.array([ np.inf, np.inf, 2,  
                                            #20
                                            ] + [np.inf]*num_gait_fingerprints).reshape(-1,1)
lower_limits = np.array([-np.inf,      0.8, 0, 
                                            #-20
                                            ] + [-np.inf]*num_gait_fingerprints).reshape(-1,1)


#Test queue
#-> Make gait fingerprint faster - works pretty well
#-> Turn heteroschedastic model off totally, just Q_h, just R_h
#Make ramp faster
#Updade model structure to have higher order models

#Process noise
gf_var =                4e-10# * frec_component/30
phase_var =             0   
phase_dot_var =         9e-8# * frec_component/4
stride_length_var =     6e-8# * frec_component/4#32447 out of 108000 rmse [0.012 0.031 0.067]  with 1e-7
# ramp_var =              1e-4



# q_diag = [phase_var,phase_dot_var,stride_length_var,
#                                                     #ramp_var
#                                                     ] + [gf_var*(10**i) for i in range (num_gait_fingerprints)]
q_diag = [phase_var,
          phase_dot_var,
          stride_length_var,
          #ramp_var
          ] + [5e-6] + [gf_var]*(num_gait_fingerprints-1)

# q_diag = [phase_var,phase_dot_var,stride_length_var,
#                                                    #ramp_var
#                                                    ] + [5e-5,1e-6,1e-6]
Q_gf = np.diag(q_diag)



##########################################3
# try: 
#     initial_state_covariance_avg = np.load('pred_covar_avg.npy')
# except Exception:
#     print("Using default covar")
cov_diag = 5e-6
initial_state_diag_avg = [cov_diag]*num_states
initial_state_covariance_avg = np.diag(initial_state_diag_avg)


#Process noise
#Phase, Phase, Dot, Stride_length, ramp, gait fingerprints
phase_var_avg =         0     
phase_dot_var_avg =     2e-8 #2e-8
stride_length_var_avg = 7e-8  #7e-8
# ramp_var_avg =          1e-5  


Q_avg = [phase_var_avg,
         phase_dot_var_avg,
         stride_length_var_avg,
         #ramp_var_avg
         ]

Q_avg_diag = np.diag(Q_avg)

average_subject_upper_limit = np.array([ np.inf, np.inf, 2, 
                                                         #20
                                                         ]).reshape(-1,1)
average_subject_lower_limit = np.array([-np.inf,      0.5, 0.8, 
                                                        #-20
                                                        ]).reshape(-1,1)



#Generate random initial conditions 
# n_initial_conditions = 1
# lower_state_random_limit = [0.0, -0.3,   0, -20] + [0]*num_gait_fingerprints
# upper_state_random_limit = [1.0,  1.4, 2.0, 20] + [0]*num_gait_fingerprints

# #Unsure how to feed in the upper and lower bounds for each state as a vector, so just generate each condition 
# #with a list comprehension and then append all of them together as a np array
# initial_conditions = np.array([np.random.uniform(l,h,n_initial_conditions) for l,h in zip(lower_state_random_limit, 
#                                                                                           upper_state_random_limit)])

#Static initial condition near the expected values
initial_conditions = np.array([0,
                               0.8,
                               1.2,
                               #0
                               ] 
                               + [0]*num_gait_fingerprints).reshape(-1,1)
n_initial_conditions = 1

#Initialize a dictionary for all the rmse with the subjects
subject_to_rmse_gf_dict = {subject: [] for subject in subject_list}
subject_to_rmse_avg_dict = {subject: [] for subject in subject_list}

#Setup Flags                              phase,  phase dot,  stride_length
DO_GF = True   #10878 out of 135000 rmse [0.023   0.029       0.065]
DO_AVG = True  #14314 out of 191700 rmse [0.009   0.017       0.088] 
DO_LS_GF = True

RT_PLOT = True

ls_gf_list=[]

#Calculate the RMSE for all the subjects
for subject in subject_list[1:2]:

    print(f"Doing Subject {subject}")

    #Iterate through all the initial conditions
    for i in range(n_initial_conditions): 

        #Get the random initial state
        initial_state = initial_conditions[:,[i]]

        print(f"Doing initial condition {initial_state}")
        
        if(DO_GF):   
            
            #Wait until the other plot is done to start
            if(RT_PLOT==True):
                input("Press enter when plot is done updating")
            
            #Get the rmse for that initial state
            rmse_testr, ls_gf = simulate_ekf(subject, initial_state, initial_state_covariance, Q_gf, R_scaled, 
                                     lower_limits, upper_limits, plot_local=RT_PLOT)

            print(f"\n\r  {i:02} {subject} gf system rmse {rmse_testr}")
            
            #Add to the rmse dict
            subject_to_rmse_gf_dict[subject].append(rmse_testr)

        if(DO_AVG):
            
            #Wait until the other plot is done to start
            if(RT_PLOT==True):
                input("Press enter when plot is done updating")

            #Crop the gait fingerprint from the initial state
            initial_state_no_gf = initial_state[:num_states,:]


            #Run the simulation again, with the average fit
            #Get the rmse for that initial state
            rmse_testr, ls_gf = simulate_ekf(subject, initial_state_no_gf, initial_state_covariance_avg, Q_avg_diag, R_scaled, 
                                      average_subject_lower_limit, average_subject_upper_limit, plot_local=RT_PLOT, 
                                      use_subject_average=True)

            print(f"\n\r  {i:02} {subject} avg system rmse {rmse_testr} initial state")
            
            #Add to the rmse dict
            subject_to_rmse_avg_dict[subject].append(rmse_testr)
        
        if(DO_LS_GF):
            
            #Wait until the other plot is done to start
            if(RT_PLOT==True):
                input("Press enter when plot is done updating")

            #Crop the gait fingerprint from the initial state
            initial_state_no_gf = initial_state[:num_states,:]


            #Run the simulation again, with the average fit
            #Get the rmse for that initial state
            rmse_testr, ls_gf = simulate_ekf(subject, initial_state_no_gf, initial_state_covariance_avg, Q_avg_diag, R_scaled, 
                                      average_subject_lower_limit, average_subject_upper_limit, plot_local=RT_PLOT, 
                                      use_ls_gf=True)

            print(f"\n\r  {i:02} {subject} ls gf system rmse {rmse_testr} initial state")
            print(f"Norm of ls-gf  {ls_gf}")
            #Add to the rmse dict
            subject_to_rmse_avg_dict[subject].append(rmse_testr)

        #Add the least squares solutoin
        ls_gf_list.append(ls_gf)

print("Done")
print(subject_to_rmse_gf_dict)
print(subject_to_rmse_avg_dict)

save_content = [model,
                initial_conditions,
                subject_to_rmse_gf_dict,
                subject_to_rmse_avg_dict,
                ls_gf_list,
                [Q_gf,R_scaled],
                [Q_avg,R_scaled]]

with open('sim_rmse_results.pickle', 'wb') as handle:
    pickle.dump(save_content, handle, protocol=pickle.HIGHEST_PROTOCOL)





#Calibrations that I like 
# gf_var =                1e-6 #for two gf, for three use 1e-7, for give use 5e-5
# phase_var =             0   
# phase_dot_var =         4e-7
# stride_length_var =     6e-8 #for two gf, for three/five use 7e-8
# ramp_var =              3e-5


# #Measurement covarience, Innovation
# r_diag = [12,36,12] + [15,38,15]

# phase_basis = FourierBasis(6,'phase')
# phase_dot_basis = HermiteBasis(2,'phase_dot')
# ramp_basis = HermiteBasis(2,'ramp')
# stride_basis = HermiteBasis(2,'stride_length')
# l2_lambda = [0.2,0.01,0,0]

#With heterschedastic model on
#Two gait figerprints



####################################################


# #Process noise
# gf_var =                1e-6
# phase_var =             0   
# phase_dot_var =         4e-7
# stride_length_var =     7e-8
# ramp_var =              3e-5


# #Measurement covarience, Innovation
# r_diag = [12,36,12] + [15,38,15]
# R = np.diag(r_diag)*3


# phase_basis = FourierBasis(9,'phase')
# phase_dot_basis = HermiteBasis(2,'phase_dot')
# ramp_basis = HermiteBasis(2,'ramp')
# stride_basis = HermiteBasis(2,'stride_length')
# l2_lambda = [0,0,0,0]


#With heterschedastic model on
#One gait figerprints



#Doing higher powers than 2 in the models not worth it for phase