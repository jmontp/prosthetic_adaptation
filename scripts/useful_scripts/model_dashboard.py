"""
This module is meant to load kronecker models and plot their joint kinematics
over phase for a given phase and gait fingerprint
"""

#Dash dependencies
import dash 
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

#Bread and butter imports
import pandas as pd
import numpy as np

#Custom imports
from context import kmodel
from kmodel.personalized_model_factory import PersonalizedKModelFactory
###############################################################################
###############################################################################
## Constants definition

#Create the subject for the different plot configurations
GF_CONFIG = 'Gait Fingerprint'
AVG_CONFIG = 'Average Subject'
OPT_CONFIG = 'Optimal Fit'
LSGF_CONFIG = 'Least Squares GF'
NULL_SPACE_CONFIG = 'Null-Space'


###############################################################################
###############################################################################
## Load the model

def load_subject_model(subject,cond):
    """Loads the model and certain parameters for a subject"""

    #Import the personalized model 
    factory = PersonalizedKModelFactory()

    #Get the file for the model
    if(cond == NULL_SPACE_CONFIG):
         model_dir = (f'../../data/kronecker_models/'
                f'left_one_out_model_{subject}_null.pickle')
    else:
        model_dir = (f'../../data/kronecker_models/'
                    f'left_one_out_model_{subject}.pickle')

    model = factory.load_model(model_dir)

    #Define the joints that you want to import
    joint_names = model.output_names

    #Get the number of gait fingerprints from the model
    num_gf = model.kmodels[0].num_gait_fingerpints

    #Phase, Phase Dot, Ramp, Step Length, 5 gait fingerprints
    state_names = model.kmodels[0].model.basis_names\
        + [f"gf{i}" for i in range(num_gf)]

    return model, joint_names, num_gf, state_names

#Define an initial subject
initial_subject = "AB01"

#load the subject
model, joint_names, num_gf, state_names = load_subject_model(initial_subject,
                                                             GF_CONFIG)

#Describe the model 
def describe_model(model):
    """Take in a personal measurement function and extract the internal
    model structure"""
    #Get the names of the parameters of the basis functions
    basis_names = model.kmodels[0].model.basis_names
    #Get the size of each basis list
    basis_sizes = [basis.n for basis in model.kmodels[0].model.basis_list]
    #Get the basis type (e.g. Fouriter, Polynomial, etc)
    basis_type = [basis.name for basis in model.kmodels[0].model.basis_list]
    #Get the list of l2 regularization
    l2_status = [(kmodel.output_name,kmodel.l2_lambda) for kmodel in model.kmodels]


    #Get a list of every model name
    model_description = \
        [html.Div(children = f"{name.capitalize()}, {type} - {size}")
         for name,size,type
         in zip(basis_names,basis_sizes,basis_type)] + \
        [html.Div(children=f"{output_name} L2 Reg - {l2_lambda}")
         for output_name, l2_lambda
         in l2_status]



    return model_description

model_desc = describe_model(model)

###############################################################################
###############################################################################
## Configure the dash elements

#Using the bootstrap library to help with the layout
# https://dash-bootstrap-components.opensource.faculty.ai/
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

#Create my sliders

def custom_gf_slider(title):
    """Create a slider for gait fingerprints"""
    return dcc.Slider(
        id=f"{title}-slider",
        min=-10.0,
        max=10.0,
        #step=0.5,
        value=0,
        tooltip={"placement": "bottomLeft", "always_visible": True},
    )


def get_slider_names(num_sliders=1):
    """Create the names for the gait fingerprint sliders"""
    return [f"GF{i+1}" for i in range(num_sliders)]


def create_gf_slider_array(num_sliders=1):
    """Generates the dash container with the sliders"""
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
    """Creates the slider for a particular state with the nim max values"""
    #Make the ID for the slider
    id_name = '-'.join(name.lower().split(' '))
    #Create the title for the slider
    title = html.Label(name)
    #Create the slider object itself
    slider = dcc.Slider(
        id=f'{id_name}-slider',
        min=min,
        max=max,
        #step=step,
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
    return state_slider("Ramp", -90.0,90.0,2.0,0.0)

def create_dropdown(name, values):
    """This will create a dropdown that has the id of name and it will be
    filled with the content in values 
    
    Keyword arguments:
    name -- string that will set the identifies of the dropdown
    values -- content of the dropdown. It will be initialized with the first 
        value in the list
    """
    return html.Div(
        id=f'{name}-div',
        children=[dcc.Dropdown(options=values,
                               value=values[0],
                               id=name)]
    )

#Create the content for the subject drop down box
subject_names = [f"AB{i:02}" for i in range(1,11)]
subject_dropdown_name = "subject_dropdown"
create_subject_dropdown = lambda: create_dropdown(subject_dropdown_name,\
                                                  subject_names)

#Create the dropdown for the slider configuration
conditions = [GF_CONFIG, AVG_CONFIG, OPT_CONFIG, LSGF_CONFIG, NULL_SPACE_CONFIG]
conditions_dropdown_name = 'joint_model_configuration_dropdown'
create_conditions_dropdown = lambda:create_dropdown(conditions_dropdown_name,\
                                                     conditions)
###############################################################################
###############################################################################
## Configure the plotter callback

#Define the input and output for the plotter callback
callback_input = [Input(f"{i}-slider","value") 
                  for i 
                  in get_slider_names(num_gf)]
callback_input.append(Input("phase-dot-slider","value"))
callback_input.append(Input("ramp-slider","value"))
callback_input.append(Input("stride-length-slider","value"))
callback_input.append(Input(subject_dropdown_name,"value"))
callback_input.append(Input(conditions_dropdown_name,"value"))

callback_outputs = [Output("model-plot-div","children")]

#This is the function that will calculate the plot
@app.callback(
inputs = callback_input,
output = callback_outputs
)
def plotter_callback(*input):
    """
    This is the function that will run whenever any of the sliders are updated

    It is meant to take in the new data and output the new joint kinematics
    """

    global model
    global joint_names
    global num_gf
    global state_names

    #The first elements in the input list are the gait fingerprints
    gf = input[:num_gf]
    
    #After the gait fingerprints, the states are defined in the order that they
    # are appended to callback_input
    phase_dot = input[num_gf]
    ramp = input[num_gf+1]
    stride_length = input[num_gf+2]

    #Get the subject and condition
    subject = input[num_gf+3]
    condition = input[num_gf+4]

    #Verify that we have the correct model
    if not (model.kmodels[0].subject_name == subject):
        print(f"udpated model to {subject} from {model.kmodels[0].subject_name}")
        model, joint_names, num_gf, state_names = load_subject_model(subject,condition)

    
    #Define the number of points to plot per step
    points_per_step = 150
    
    #Format each joint name to have spaces instead of underscores and title
    # capitalization
    format_name = lambda joint_name:' '.join(joint_name.split('_')[1:]).title()
    
    #Create the titles for every plot
    model_titles = [format_name(joint_name)
                    for joint_name
                    in joint_names]

    #Set the phase as one step
    phase = np.linspace(0,1,points_per_step).reshape(-1,1)

    #Create an array for the  samples
    states = np.array([phase_dot,
                       stride_length,
                       ramp,
                       *gf]).reshape(1,-1)

    #Concatenate the phase to all the variables
    # (e.g. everything but phase will be fixed)
    total_states = np.concatenate(
                        (phase, np.tile(states,(points_per_step,1)))
                    , axis=1)

    ##Evaluate the joint angles
    #Evaluate just with gait fingerprints
    if(condition == GF_CONFIG or condition == NULL_SPACE_CONFIG):
        joint_angles = model.evaluate(total_states)
    #Use the average configuration
    elif (condition == AVG_CONFIG):
        joint_angles = model.evaluate(total_states,use_average_fit=True)
    elif (condition == OPT_CONFIG):
        joint_angles = model.evaluate(total_states,use_optimal_fit=True)
    elif(condition == LSGF_CONFIG):
        joint_angles = model.evaluate(total_states,use_personalized_fit=True)

    #Reshape to plot
    phase_plot = phase.reshape(-1)

    #Create a list for each model's dash object
    model_figs = []

    #Plot all the joint angles
    for i,plot_name in enumerate(model_titles):

        #Create the figure for the plot
        fig = go.Figure(data=go.Scatter(x=phase_plot,y=joint_angles[:,i]))
        
        #Add axis information
        fig.update_layout(
            xaxis_title="Phase",
            yaxis_title= plot_name + " Angle (Deg)"
        )

        #Create a css columns
        layout = dbc.Col([
                #Header for the title
                html.Div(plot_name + " Angle (Deg)"),
                #Graph content
                dcc.Graph(
                    id=plot_name + '-angle-graph',
                    figure = fig, 
                    style={'margin.t':0}
                )
            ])
        
        #Add the configuration dash component list
        model_figs.append(layout)

    return [model_figs]




#This is what defines the layout of the dashboard
app.layout = dbc.Container([

    dbc.Row([
        #First column will be configuration of the plots
        dbc.Col(
            html.Div([

                #Header
                html.H1(children='Joint Model Dashboard'),

                #Short introduction
                html.Div(
                    [html.Div(children='Plot angles based on the model parameters'),\
                        *model_desc]
                ),
                html.Br(),
                #Add the sliders
                create_phase_dot_slider(),
                create_ramp_slider(),
                create_stride_length_slider(),
                create_gf_slider_array(num_gf),
                #Create the dropdowns
                html.Div(children='Select the subject'),
                create_subject_dropdown(),
                html.Div(children='Select the model configuration'),
                create_conditions_dropdown(),
            ])
        ),

        #Second column will be the plots themselves
        dbc.Col(
            #This is the container for the plots
            html.Div(
               id='model-plot-div'
           )
        )
    ]),

])


###############################################################################
###############################################################################
## Main function

if __name__ == '__main__':
    app.run_server(debug=True)
