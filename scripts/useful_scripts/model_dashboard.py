#%%Dash dependencies
import enum
import dash 
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

#Bread and butter imports
import pandas as pd
import numpy as np

#Using the bootstrap library to help with the layout
# https://dash-bootstrap-components.opensource.faculty.ai/
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

#Custom imports
from context import kmodel
from kmodel.personalized_model_factory import PersonalizedKModelFactory


#Import the personalized model 
factory = PersonalizedKModelFactory()

subject_model = "AB10"

model_dir = f'../../data/kronecker_models/left_one_out_model_{subject_model}.pickle'

model = factory.load_model(model_dir)

#Define the joints that you want to import 
joint_names = model.output_names

#Get the number of gait fingerprints from the model
num_gf = model.kmodels[0].num_gait_fingerpints


#Phase, Phase Dot, Ramp, Step Length, 5 gait fingerprints
state_names = ['phase', 'phase_dot', 'stride_length', 'ramp'] + [f"gf{i}" for i in range(num_gf)]



#%%

#Create my sliders

#Slider for all the gait fingerprints
def custom_gf_slider(title):
    return dcc.Slider(
        id=f"{title}-slider",
        min=-10.0,
        max=10.0,
        step=0.5,
        value=0,
        tooltip={"placement": "bottomLeft", "always_visible": True},
    )
    
def get_slider_names(num_sliders=1):
    return [f"GF{i+1}" for i in range(num_sliders)]

def create_gf_slider_array(num_sliders=1):
    slider_names = get_slider_names(num_sliders)
    sliders = [custom_gf_slider(slider_names[i]) for i in range(num_sliders)]
    titles = [html.Label(slider_names[i]) for i in range(num_sliders)]

    #Interleave the titles with the sliders
    children = [val for pair in zip(titles,sliders) for val in pair]
    return html.Div(
            id='gf-slider-box',
            children=children
        )

def state_slider(name,min,max,step,initial_value):
    id_name = '-'.join(name.lower().split(' '))
    title = html.Label(name)
    slider = dcc.Slider(
        id=f'{id_name}-slider',
        min=min,
        max=max,
        step=step,
        value=initial_value,
        tooltip={"placement": "bottomLeft", "always_visible": True},
    )
    return html.Div(
        id=f'{id_name}-length-box',
        children=[title,slider]
    )

def create_phase_dot_slider():
    return state_slider("Phase Dot", 0.0,2.0,0.1,0.5)
  
def create_stride_length_slider():
   return state_slider("Stride Length",0.0,2.0,0.1,1.4)

def create_ramp_slider():
    return state_slider("Ramp", -200.0,200.0,2.0,0.0)
   
#Define the input and output for the plotter callback
callback_input = [Input(f"{i}-slider","value") for i in get_slider_names(num_gf)]
callback_input.append(Input("phase-dot-slider","value"))
callback_input.append(Input("ramp-slider","value"))
callback_input.append(Input("stride-length-slider","value"))

callback_outputs = [Output("model-plot-div","children")]


#This is the function that will calculate the plot
@app.callback(
inputs = callback_input,
output = callback_outputs
)
def plotter_callback(*input):

    global num_gf

    gf = input[:num_gf]
    
    phase_dot = input[num_gf]
    
    ramp = input[num_gf+1]
    
    stride_length = input[num_gf+2]

    #Define the number of points to plot per step
    points_per_step = 150

    #Create a list for each model
    model_figs = []

    #Create the titles for every plot
    model_titles = [' '.join(joint_name.split('_')[1:]).title() for joint_name in joint_names]

    #Set the phase as one step
    phase = np.linspace(0,1,points_per_step).reshape(-1,1)

    #Create an array for the  samples
    states = np.array([phase_dot,
                       stride_length,
                       #ramp,
                       *gf]).reshape(1,-1)

    #Concatenate the phase to all the variables (e.g. everything but phase will be fixed)
    total_states = np.concatenate((phase,np.tile(states,(points_per_step,1))), axis=1)

    joint_angles = model.evaluate(total_states)

    phase_plot = phase.reshape(-1)

    for i,plot_name in enumerate(model_titles):

        fig = go.Figure(data=go.Scatter(x=phase_plot,y=joint_angles[:,i]))
        
        fig.update_layout(
            xaxis_title="Phase",
            yaxis_title= plot_name + " Angle (Deg)"
        )

        layout = dbc.Col([
                html.Div(plot_name + " Angle (Deg)"),
                dcc.Graph(
                    id=plot_name + '-angle-graph',
                    figure = fig, 
                    style={'margin.t':0}
                )
            ])
        model_figs.append(layout) 

    return [model_figs]




#This is what defines the layout of the dashboard
app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(
            html.Div([
                html.H1(children='Gait Fingerprints'),


                html.Div(children=f'''
                    Plot gait paths based on the gait fingerprints that you have right now 
                    ls gf {model.kmodels[0].subject_gait_fingerprint}
                '''),
                html.Br(),
                create_phase_dot_slider(),
                create_ramp_slider(),
                create_stride_length_slider(),
                create_gf_slider_array(num_gf), 
            ])
        ),


        dbc.Col(
           html.Div(
               id='model-plot-div'
           )
        )
    ]),

])



if __name__ == '__main__':
    app.run_server(debug=True)
# %%
