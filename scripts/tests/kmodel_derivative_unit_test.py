
#%%
def validate_velocity_derivative():
    pass
#%% 
    
    trial = 's1x2i7x5'
    subject = 'AB01'
    joint = 'jointangles_shank_x'
    filename = '../local-storage/test/dataport_flattened_partial_{}.parquet'.format(subject)
    
    df = pd.read_parquet(filename)
    trial_df = df[df['trial'] == trial]
    
    model_foot_dot = model_loader('foot_dot_model.pickle')
    model_foot = model_loader('foot_model.pickle')    
    
    states = ['phase', 'phase_dot', 'stride_length', 'ramp']
    states_data = trial_df[states].values.T
    
    
    foot_angle_evaluated = model_foot.evaluate_numpy(states_data)
    foot_angle_dot_evaluated = model_foot_dot.evaluate_numpy(states_data)

    
    #Calculate the derivative of foot dot manually
    foot_anles_cutoff = trial_df[joint].values[:-1]    
    foot_angles_future = trial_df[joint].values[1:]
    phase_rate = trial_df['phase_dot'].values[:-1]
    
    measured_foot_derivative = (foot_angles_future-foot_anles_cutoff)*(phase_rate)*150
    calculated_foot_derivative = foot_angle_dot_evaluated @ model_foot_dot.subjects[subject]['optimal_xi']
    
    measured_foot_angle = trial_df[joint]
    calculated_foot_angles = foot_angle_evaluated @ model_foot.subjects[subject]['optimal_xi']
    
    points_per_stride = 150
    start_stride = 40
    num_strides = 3 + start_stride
    x = np.linspace(0,1+1/(num_strides-start_stride)*points_per_stride,(num_strides-start_stride)*points_per_stride)
    fig, axs = plt.subplots(2,1)
    axs[0].plot(x,measured_foot_derivative[start_stride*points_per_stride:num_strides*points_per_stride])
    axs[0].plot(x,calculated_foot_derivative[start_stride*points_per_stride:num_strides*points_per_stride])
    axs[0].legend(['measured','calculated'])
    axs[0].grid(True)

    
    axs[1].plot(x, measured_foot_angle[start_stride*points_per_stride:num_strides*points_per_stride])
    axs[1].plot(x, calculated_foot_angles[start_stride*points_per_stride:num_strides*points_per_stride])
    axs[1].legend([ 'measured foot angle', 'calculated foot angle'])
    axs[1].grid(True)
    plt.show()

