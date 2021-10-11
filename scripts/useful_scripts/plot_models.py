#Common imports
import numpy as np 
import matplotlib.pyplot as plt


#Relative Imports
from context import kmodel
from kmodel.kronecker_model import model_loader



model_dir = '../../data/kronecker_models/'

#%%
def plot_model():
    #%%
    #%matplotlib qt



    model_foot = model_loader('foot_model.pickle')
    model_shank = model_loader('shank_model.pickle')
    model_dict = {'jointangles_shank_x': model_shank,
                  'jointangles_foot_x': model_foot}
    
    #Variables
    ##########################################################################
    subject = 'AB10'
    joint_angle = 'jointangles_foot_x'
    model = model_dict[joint_angle]
    trials = ['s0x8i0','s0x8i10','s1x2i0']
    #trials = ['s0x8i0']
    #trials = list(df.trial.unique())[:]
    mean_plots = False
    ##########################################################################
    
    joint_str = joint_angle.split('_')[1].capitalize()
    try:
        subject_dict = model.subjects[subject]
    except KeyError:
        print("Not in subject dict, checking left out dict")
        subject_dict = model.one_left_out_subjects[subject]
        print("In left out dict")

    
    #Get constants
    inter_subject_average_fit = model.inter_subject_average_fit
    personalization_map_scaled = model.personalization_map_scaled
    bad_personalization_map_scaled = model.personalization_map
    gait_fingerprints = subject_dict['gait_coefficients']
    bad_gait_fingerprints = subject_dict['gait_coefficients_unscaled']
    optimal_fit = subject_dict['optimal_xi']
    
    df = pd.read_parquet(subject_dict['filename'])
    
    points_per_stride = 150
    x = np.linspace(0,1+1/points_per_stride,points_per_stride)
    
    fig, ax = plt.subplots(1,len(trials), sharex='all',sharey ='all')
    
    if (len(trials)==1):
        ax = [ax]
    
    for i,trial in enumerate(trials):
        #Level ground walking
        #Get measured data
        trial_df = df[df['trial'] == trial]
        
        #Uncomment to get rmse for all trials
        #trial_df = df
        measured_angles = trial_df[joint_angle]
        foot_mean, foot_std_dev= get_mean_std_dev(measured_angles)
        
        #Get regressor rows
        if(mean_plots == True):
            foot_angle_evaluated = model.evaluate_pandas(trial_df)
        else:
            measured_angles_total = measured_angles
            measured_angles = measured_angles[:150]
            foot_angle_evaluated = model.evaluate_pandas(trial_df)[:150]
        
        
        print(trial)
        #Optimal fit
        optimal_estimate = foot_angle_evaluated @ optimal_fit
        optimal_mean, optimal_std_dev = get_mean_std_dev(optimal_estimate)
        optimal_rmse = get_rmse(optimal_estimate,measured_angles)
        print("Optimal rmse {}".format(optimal_rmse))

        #Intersubject fit
        inter_subject_estimate = foot_angle_evaluated @ inter_subject_average_fit
        inter_subject_mean,  inter_subject_std_dev = get_mean_std_dev(inter_subject_estimate)
        inter_subject_rmse = get_rmse(inter_subject_estimate,measured_angles)
        print("Inter subject average rmse {}".format(inter_subject_rmse))

        
        #Gait fingerprint fit
        gait_fingerprint_estimate = foot_angle_evaluated @ (inter_subject_average_fit + personalization_map_scaled @ gait_fingerprints)
        gait_fingerprint_mean, gait_fingerprint_std_dev = get_mean_std_dev(gait_fingerprint_estimate)
        gait_fingerprint_rmse = get_rmse(gait_fingerprint_estimate,measured_angles)
        print("Gait fingerprint rmse {}".format(gait_fingerprint_rmse))

        
        #Bad gait fingerprint fit
        bad_gait_fingerprint_estimate = foot_angle_evaluated @ (inter_subject_average_fit + bad_personalization_map_scaled @ bad_gait_fingerprints)
        bad_gait_fingerprint_mean, bad_gait_fingerprint_std_dev = get_mean_std_dev(bad_gait_fingerprint_estimate)
        bad_gait_fingerprint_rmse = get_rmse(bad_gait_fingerprint_estimate,measured_angles)
        print("Bad gait fingerprint rmse {}".format(bad_gait_fingerprint_rmse))
        
        
        clrs = cm.get_cmap('tab20').colors
            
        if(mean_plots == True):
            #Measured
            #Mean plots with width 
            ax[i].plot(x, foot_mean,label='Measured Foot Angle', c=clrs[0], linestyle = 'solid')
            ax[i].fill_between(x, foot_mean-foot_std_dev, foot_mean+foot_std_dev ,alpha=0.3, facecolor=clrs[0])
            #Optimal
            ax[i].plot(x, optimal_mean,label='Optimal Fit RMSE:{:.2f}'.format(optimal_rmse), c=clrs[1])
            ax[i].fill_between(x, optimal_mean-optimal_std_dev, optimal_mean+optimal_std_dev ,alpha=0.3, facecolor=clrs[1])
            #Inter subject average
            ax[i].plot(x, inter_subject_mean,label='Inter-Subject Averate Fit RMSE:{:.2f}'.format(inter_subject_rmse), c=clrs[2])
            ax[i].fill_between(x, inter_subject_mean-inter_subject_std_dev, inter_subject_mean+inter_subject_std_dev ,alpha=0.3, facecolor=clrs[2])
            #Gait fingerprint
            ax[i].plot(x, gait_fingerprint_mean,label='Gait Fingerprint Fit RMSE:{:.2f}'.format(gait_fingerprint_rmse), c=clrs[3])
            ax[i].fill_between(x, gait_fingerprint_mean-gait_fingerprint_std_dev, gait_fingerprint_mean+gait_fingerprint_std_dev ,alpha=0.3, facecolor=clrs[3])
            #Bad Gait fingerprint
            ax[i].plot(x, bad_gait_fingerprint_mean,label='Bad Gait Fingerprint Fit RMSE:{:.2f}'.format(bad_gait_fingerprint_rmse), c=clrs[4])
            ax[i].fill_between(x, bad_gait_fingerprint_mean-bad_gait_fingerprint_std_dev, bad_gait_fingerprint_mean+bad_gait_fingerprint_std_dev ,alpha=0.3, facecolor=clrs[4])
        
        else:
            
            line_width = 6
            # Individual line plots
            stride_data = measured_angles_total.values.reshape(-1,150)
            for k in range (0,stride_data.shape[0],3):
                if (150 - np.count_nonzero(stride_data[k,:]) > 20):
                    continue
                if k == 0:
                    ax[i].plot(x, stride_data[k,:],label='Measured Foot Angle', linestyle = 'solid', alpha=0.2, linewidth=5, c='darkgrey')
                else:
                    ax[i].plot(x, stride_data[k,:], linestyle = 'solid', alpha=0.3, linewidth=5, c='darkgrey')
            
            #Inter subject average
            ax[i].plot(x, inter_subject_estimate,label='Inter-Subject Averate Fit RMSE:{:.2f}'.format(inter_subject_rmse),
                       linewidth=line_width, c=clrs[6])#, linestyle=(0, (1, 1)), alpha=0.8) #Densely dotted
           
            #Bad Gait fingerprint
            # ax[i].plot(x, bad_gait_fingerprint_estimate,label='Bad Gait Fingerprint Fit RMSE:{:.2f}'.format(bad_gait_fingerprint_rmse),
            #            linewidth=line_width,c=clrs[4],linestyle=(0,(6,1,1,1)), alpha=0.8)
            
            #Optimal fit
            ax[i].plot(x, optimal_estimate,label='Optimal Fit RMSE:{:.2f}'.format(optimal_rmse),
                       linewidth=line_width, c=clrs[2])#, linestyle=(0, (6, 1)), alpha=0.8) #Densely dash dot dotted
            
            #Gait fingerprint
            ax[i].plot(x, gait_fingerprint_estimate,label='Gait Fingerprint Fit RMSE:{:.2f}'.format(gait_fingerprint_rmse),
                       linewidth=line_width, c=clrs[0], linestyle='solid', alpha=0.8) 
            
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)
            ax[i].title.set_text(trial_to_string(trial,joint_str))
            ax[i].legend()
            
#%%
def plot_cumulative_variance():
    pass
#%%
    model_foot = model_loader('foot_model.pickle')
    model_shank = model_loader('shank_model.pickle')
    model_foot_dot = model_loader('foot_dot_model.pickle')
    model_shank_dot = model_loader('shank_dot_model.pickle')
    
    
    clrs = cm.get_cmap('tab20').colors

    model = model_foot
    
    pca_values = model.scaled_pca_eigenvalues
    
    pca_values_sum= np.sum(pca_values)
    marker_on = [5]
    pca_cum_sum = np.cumsum(pca_values)/pca_values_sum
    
    plt.plot(pca_cum_sum[:11], '-o', markevery=marker_on, linewidth=7, markersize=15, mfc = 'r', mec='r',c=clrs[0])
    plt.xticks(np.arange(0, 11, 1.0))
    plt.show()
    