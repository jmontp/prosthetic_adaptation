#Common imports
from re import S
from matplotlib.pyplot import fill_between, title
import numpy as np
from numpy import average
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


#Custom imports
from context import kmodel
from kmodel.personalized_model_factory import PersonalizedKModelFactory





####################################################
####################################################
PLOT_EFFECT_OF_GF = False
PLOT_GF_VS_AVERAGE_VS_DATA = True
####################################################
####################################################
















##Import the personalized model 
factory = PersonalizedKModelFactory()

subject_model = "AB09"

model_dir = f'../../data/kronecker_models/left_one_out_model_{subject_model}.pickle'

model = factory.load_model(model_dir)

#Get the number of gait fingerprints
num_gait_fingerprints = model.num_gait_fingerprint
num_states = model.num_states


names = model.kmodels[0].model.get_basis_names()
num_basis = [str(m.n) for m in model.kmodels[0].model.basis_list]

plot_title = subject_model+ ' '  + ' '.join([name + ' ' + n for name, n in zip(names,num_basis)])
plot_title += ' ' + f' num gf {num_gait_fingerprints}'
print(plot_title)



#Get the subject data
file_location = "../../data/flattened_dataport/dataport_flattened_partial_{}.parquet"
#Get the file for the corresponding subject
filename = file_location.format(subject_model)
#Read in the parquet dataframe
total_data = pd.read_parquet(filename)

#Common values found in the dataset
phase_dot = 1
stride_length = 1.0
ramp = 0

# mask = (total_data['ramp'] == ramp) & (total_data['speed'] == 1.2)
mask = total_data['ramp'] == ramp 
total_data = total_data[mask]

#Get the least amount of steps per condition to make sure one condition doesn't overshadow mean and std_dev
speed_list = [0.8,1.0,1.2]
find_min_stride = lambda dataset: min([(dataset.speed == speed).sum() for speed in speed_list])
remove_steps_per_condition = lambda dataset,min_stride: pd.concat([dataset[dataset.speed == speed].iloc[:min_stride] for speed in speed_list])
min_strides = find_min_stride(total_data)
total_data = remove_steps_per_condition(total_data,min_strides)


##Generate data
mean_list = [total_data[output_name].values.reshape(-1,150).mean(axis=0) for output_name in model.output_names]
std_list =  [total_data[output_name].values.reshape(-1,150).std(axis=0) for output_name in model.output_names]


#Calculate the predicted joint angles 
states_names = ['phase','phase_dot','stride_length']
states_np = total_data[states_names].values
average_fit_output = model.evaluate(states_np,use_average_fit=True)
personalized_fit_output = model.evaluate(states_np,use_personalized_fit=True)

get_mean_strides = lambda data: data.reshape(-1,150).mean(axis=0)
get_std_strides = lambda data: data.reshape(-1,150).std(axis=0)

#Get the mean and std dev for the average fit
average_fit_thigh_mean = get_mean_strides(average_fit_output)
average_fit_thigh_std = get_std_strides(average_fit_output)

#Get the mean and std dev for the average fit
personalized_fit_thigh_mean = get_mean_strides(personalized_fit_output)
personalized_fit_thigh_std = get_std_strides(personalized_fit_output)

#Create the dataset

#Define a arrays
phase_np = np.linspace(0,1,150)
gf_np = [-10,0,10]
colors = ['red', 'green','blue']

#Define input list without one gait fingeprint and the others set as zero
input_list = [phase_np, [phase_dot], [stride_length], 
                                                    #[ramp]
                                                    ] + [[0]]*(num_gait_fingerprints-1)

#Create all the permutations of the inputs based on the number of gaint fingerprints
gf_list = range(num_gait_fingerprints)

#Create a list of inputs with the gait fingerprint varying in different locations
input_data_list = [input_list[:num_states+i] + [gf_np] + input_list[num_states+i:] for i in gf_list]
data_vary_list = [np.stack(np.meshgrid(*data), -1).reshape(-1,num_states+num_gait_fingerprints) for data in input_data_list]

#Run the model 
model_output_vary_gf_list = [model.evaluate(data_vary) for data_vary in data_vary_list]

#Calculate the gait fingerprint otuput for the subject
model_output_subject_gf = model.evaluate(data_vary_list[0].reshape(150,3,num_states+num_gait_fingerprints)[:,0,:],
                                         use_personalized_fit=True)



# Create subplots for every joint and every gait fingerprint
if PLOT_GF_VS_AVERAGE_VS_DATA:
    model_output_vary_gf_list = [model_output_vary_gf_list[0]]
    rows = 1
else:
    rows = len(model_output_vary_gf_list)

columns = len(model.output_names)

fig = make_subplots(
    rows=rows, cols=columns,
    specs=[[{'type':'scatter'}]*columns]*rows,
    subplot_titles=[i+f" gf {gf_list[0]}" for i in model.output_names],
    )

fig.update_layout(title_text = plot_title)

#Iterate through the rows as the model output for one gait fingerprint
for gf_index,model_output_data in enumerate(model_output_vary_gf_list):
    
    #The columns represent the different outputs
    for joint_index in range(columns):
        
        #Get the data forthe joint that we are plottin
        joint_output_data = model_output_data[:,joint_index]

        #Reshape based on the number of gait fingerprints that will be plotted
        joint_output_data_gf = joint_output_data.reshape(150,len(gf_np))

        
        if PLOT_GF_VS_AVERAGE_VS_DATA:
            #Define number of standard deviations
            num_std_dev = 2
            
            #Add one standard deviation in the positive direction
            fig.add_trace(
                go.Scatter(x=phase_np, y=mean_list[joint_index] + num_std_dev*std_list[joint_index], 
                            line=dict(color='pink'),
                            name = f'{model.output_names[joint_index]} mean + {num_std_dev} std dev',
                            #fill = 'tonexty'
                        ),
                row = gf_index+1, col = joint_index+1,
            )

            #Add the mean of the data
            mean_trace = fig.add_trace(
                go.Scatter(x=phase_np, 
                            y=mean_list[joint_index], 
                            line=dict(color='pink'),
                            name = f'{model.output_names[joint_index]} mean' if not gf_index and not joint_index else '',
                            fill='tonexty'
                        ),
                row = gf_index+1, col = joint_index+1,
            ) 

            fig.add_trace(
                go.Scatter(x=phase_np, y=mean_list[joint_index] - num_std_dev*std_list[joint_index], 
                            line=dict(color='pink'),
                            name = f'{model.output_names[joint_index]} mean - {num_std_dev} std dev',
                            fill = 'tonexty'
                        ),
                row = gf_index+1, col = joint_index+1,
            )

             #Plot the gait fingerprint model 
            # fig.add_trace(
            #     go.Scatter(x=phase_np, y=model_output_subject_gf[:,joint_index], 
            #                 line=dict(color="orange"),
            #                 name = f'gf fit'),
            #     row = gf_index+1, col = joint_index+1,
            # )

        
        if PLOT_EFFECT_OF_GF:

            #Plot each joint and gf variation
            #Add a trace per joint output at a specific gf
            for gf_variation_number in range(len(gf_np)):

                fig.add_trace(
                    go.Scatter(x=phase_np, y=joint_output_data_gf[:,gf_variation_number], 
                                line=dict(color=colors[gf_variation_number]),
                                name = f'gf={gf_np[gf_variation_number]}' if not gf_index and not joint_index else ''),
                    row = gf_index+1, col = joint_index+1,  
                )

        #Add thigh predictions
        if PLOT_GF_VS_AVERAGE_VS_DATA:

            opacity = 1.0

            fig.add_trace(  
                go.Scatter(x=phase_np, y=average_fit_thigh_mean - 2*average_fit_thigh_std, 
                            line=dict(color=f'rgba(255, 0, 0, {opacity})'),
                            name = 'aveage_fit mean - 2 std dev'),
                row = 1, col = joint_index+1,  
            )
            fig.add_trace(  
                go.Scatter(x=phase_np, y=average_fit_thigh_mean + 2*average_fit_thigh_std, 
                            line=dict(color=f'rgba(255, 0, 0, {opacity})'),
                            name = 'aveage_fit mean - 2 std dev',
                            fill='tonexty'
                ),
                row = 1, col = joint_index+1,  
            )


            fig.add_trace(  
                go.Scatter(x=phase_np, y=personalized_fit_thigh_mean - 2*personalized_fit_thigh_std, 
                            line=dict(color=f'rgba(0, 0, 255, {opacity})'),
                            name = 'personalized_fit mean - 2 std dev'),
                row = 1, col = joint_index+1,  
            )
            fig.add_trace(  
                go.Scatter(x=phase_np, y=personalized_fit_thigh_mean + 2*personalized_fit_thigh_std, 
                            line=dict(color=f'rgba(0, 0, 255, {opacity})'),
                            name = 'personalized_fit mean - 2 std dev',
                            fill='tonexty'
                ),
                row = 1, col = joint_index+1,  
            )

        #Update axes
        fig.update_xaxes(title_text="Phase (Stride Completion %)", row=gf_index+1, col=joint_index+1)
        fig.update_yaxes(title_text="Global Thigh Angle (Deg)", row=gf_index+1, col=joint_index+1)


fig.show()
fig.write_image("Figures/gait_fingerprint_vs_data.svg")