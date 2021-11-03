#%%Dash dependencies
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
from kmodel.kronecker_model import model_loader

#Initialize the measurement model
#Define the joints that you want to import 
joint_names = ['jointangles_hip_dot_x','jointangles_hip_x',
                'jointangles_knee_dot_x','jointangles_knee_x',
                'jointangles_thigh_dot_x','jointangles_thigh_x']

model_dir = '../../data/kronecker_models/model_{}.pickle'

models = [model_loader(model_dir.format(joint)) for joint in joint_names]

subject = 'AB01'

#Phase, Phase Dot, Ramp, Step Length, 5 gait fingerprints
state_names = ['phase', 'phase_dot', 'stride_length', 'ramp',
                    'gf1', 'gf2','gf3', 'gf4', 'gf5']

#%%

#Create my sliders

def custom_gf_slider(title):
    return dcc.Slider(
        id=f"{title}-slider",
        min=-3.0,
        max=3.0,
        step=0.1,
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
   return state_slider("Stride Length",0.0,2.0,0.2,0.5)

def create_ramp_slider():
    return state_slider("Ramp", -20.0,20.0,2.0,0.0)
   

callback_input = [Input(f"{i}-slider","value") for i in get_slider_names(5)]
callback_input.append(Input("phase-dot-slider","value"))
callback_input.append(Input("ramp-slider","value"))
callback_input.append(Input("stride-length-slider","value"))

callback_outputs = [Output("hip-angle-graph","figure")]

@app.callback(
inputs = callback_input,
output = callback_outputs
)
def plotter_callback(gf1,gf2,gf3,gf4,gf5,phase_dot,ramp,stride_length):

    points_per_step = 150

    phase = np.linspace(0,1,points_per_step).reshape(-1,1)

    states = np.array([phase_dot,stride_length,ramp,gf1,gf2,gf3,gf4,gf5]).reshape(1,-1)

    total_states = np.concatenate((phase,np.tile(states,(points_per_step,1))), axis=1).T

    joint_angles = [models[0].evaluate_gait_fingerprint_cross_model_numpy(total_states[:,[i]]) for i in range(points_per_step)]


    joint_angles_np = np.array(joint_angles).reshape(-1)

    phase_plot = phase.reshape(-1)

    fig = go.Figure(data=go.Scatter(x=phase_plot,y=joint_angles_np))
    
    fig.update_layout(
        xaxis_title="Phase",
        yaxis_title="Hip Angle (Deg)"
    )


    return [fig]




#This is what defines the layout of the dashboard
app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(
            html.Div([
                html.H1(children='Gait Fingerprints'),


                html.Div(children='''
                    Plot gait paths based on the gait fingerprints that you have right now
                '''),
                html.Br(),
                create_phase_dot_slider(),
                create_ramp_slider(),
                create_stride_length_slider(),
                create_gf_slider_array(5), 
            ])
        ),


        dbc.Col(
            html.Div([
                html.H1("Hip Angle (Deg)"),
                dcc.Graph(
                    id='hip-angle-graph',
                )
            ]) 
        )
    ]),

])



if __name__ == '__main__':
    app.run_server(debug=True)
# %%
