import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

storage_location = "~/workspace/rtplot/rtplot/saved_plots/"


data_online_gf = pd.read_parquet(storage_location + "online_gf_AB02.parquet")
data_ls_gf =  pd.read_parquet(storage_location + "ls_gf_AB02.parquet")
data_avg_model = pd.read_parquet(storage_location + "average_model_AB02.parquet")

#Create list of datasets
datasets = [data_online_gf, data_ls_gf, data_avg_model]

#Create color list for the plots
color_list = ['green', #online gf
              'blue',  #least squares gf
              'red',   #average
              'pink'   #data
              ]*2
names = ['Online GF', 'LS-GF','AVG', 'Data']

names = [name + '  phase' for name in names] +  [name + '  stride' for name in names]

#Get the thigh angles for each model
# thigh_angles = [dataset['Predicted Thigh Angle'].values[:-1].reshape(-1,150) for dataset in datasets]
thigh_angles = [dataset['phase'].values[:-1].reshape(-1,150) for dataset in datasets]

#Add ground truth
thigh_angles.insert(0,data_online_gf['phase_a'].values[:-1].reshape(-1,150))

#Get the thigh velocity for each model
thigh_angular_velocity = [dataset['stride_length'].values[:-1].reshape(-1,150) for dataset in datasets]

#Add ground truth
thigh_angular_velocity.insert(0,data_online_gf['stride_length_a'].values[:-1].reshape(-1,150))


#Get average velocity and standard deviation
thigh_angles_mean = [data.mean(axis=0) for data in thigh_angles]
thigh_angles_std = [data.std(axis=0) for data in thigh_angles]


thigh_angular_velocity_mean =  [data.mean(axis=0) for data in thigh_angular_velocity]
thigh_angular_velocity_std =  [data.std(axis=0) for data in thigh_angular_velocity]

rows = 2
columns = 1

means = thigh_angles_mean + thigh_angular_velocity_mean
stds = thigh_angles_std + thigh_angular_velocity_std

#Create the subplots
fig = make_subplots(
    rows=rows, cols=columns,
    specs=[[{'type':'scatter'}]*columns]*rows,
    )

#Create an axis for phase
phase_np = np.linspace(0,1,150)

#Define the number of standard deviation we want to cover
num_std_dev = 2

#Create the plots for the joint anlges
for i in range(len(means)):

    #Create the plot
    fig.add_trace(
                go.Scatter(x=phase_np, y=means[i] + num_std_dev*stds[i], 
                            line=dict(color=color_list[i]),
                            name = names[i],
                            #fill = 'tonexty'
                        ),
                row = 1 if i < 4 else 2, col = 1,
            )
    
    #Create the plot
    fig.add_trace(
                go.Scatter(x=phase_np, y=means[i] - num_std_dev*stds[i], 
                            line=dict(color=color_list[i]),
                            name = names[i],
                            fill = 'tonexty'
                        ),
                row = 1 if i < 4 else 2, col = 1,
            )

   

fig.show()