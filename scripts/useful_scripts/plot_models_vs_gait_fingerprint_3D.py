#Common imports
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import column

#Custom imports
from context import kmodel
from kmodel.personalized_model_factory import PersonalizedKModelFactory


##Import the personalized model 
factory = PersonalizedKModelFactory()

subject_model = "AB01"

model_dir = f'../../data/kronecker_models/left_one_out_model_{subject_model}.pickle'

model = factory.load_model(model_dir)

#Get the number of gait fingerprints
num_gait_fingerprints = model.num_gait_fingerprint


##Generate data

#Common values found in the dataset
phase_dot = 1
stride_length = 1.2
ramp = 0

#Create the dataset

#Define a arrays
phase_np = np.linspace(0,1,150)
gf_np = np.linspace(-10,10,150)

#Define input list without one gait fingeprint and the others set as zero
input_list = [phase_np, [phase_dot], [stride_length], [ramp]] + [[0]]*(num_gait_fingerprints-1)

#Create all the permutations of the inputs based on the number of gaint fingerprints
gf_list = range(num_gait_fingerprints)

input_data_list = [input_list[:4+i] + [gf_np] + input_list[4+i:] for i in gf_list]
data_vary_list = [np.stack(np.meshgrid(*data), -1).reshape(-1,4+num_gait_fingerprints) for data in input_data_list]

#Run the model 
model_output_vary_gf_list = [model.evaluate(data_vary) for data_vary in data_vary_list]

#Generate meshgrid for plotting
xx, yy = np.meshgrid(phase_np, gf_np)

# Create subplots for every joint and every gait fingerprint
rows = len(gf_list)
columns = len(model.output_names)

fig = make_subplots(
    rows=rows, cols=columns,
    specs=[[{'type':'surface'}]*columns]*rows,
    subplot_titles=[i+f"gf {gf_list[0]}" for i in model.output_names])


for gf_index,model_output_data in enumerate(model_output_vary_gf_list):
    for joint_index in range(columns):

        #Create the layout for this subplot
        layout = go.Layout(
            scene = dict(
                xaxis = dict(
                    title = 'Phase',
                ),
                # yaxis = dict(
                #     title = f'Gait Fingerprint {gf_index}',
                # ),

                # zaxis = dict(
                #     title = f'Joint Angles (Degree)',
                # ),
                # font = dict(
                #     family='Courier New, monospace',
                #     size=50,
                #     color='#7f7f7f'                
                # )           
            )
        )



        fig.add_trace(
            go.Surface(x=xx, y=yy, z=model_output_data[:,joint_index].reshape(150,150).T,
            colorscale='Viridis',showscale=False, 
            ),
            row = gf_index+1, col = joint_index+1,
        )

        fig.update_xaxes(title_text="phase", row=gf_index+1, col=joint_index+1)
        fig.update_yaxes(title_text="gait fingerprint", row=gf_index+1, col=joint_index+1)


fig.show()
