#Fourier Coefficient PCA vizualizer using Dash
# Relevant references
#   Dash tutorial: https://www.youtube.com/watch?v=hSPmj7mK6ng&ab_channel=CharmingData
#   Did similar layout to: https://github.com/plotly/dash-svm/blob/master/app.py


import dash
import dash_core_components as dcc
import dash_html_components as html
import utils.dash_reusable_components as drc
from dash.dependencies import Input,Output,State

import plotly.express as px

import numpy as np

from sklearn.decomposition import PCA



from fourier_calculations import get_fourier_prediction



#Initialize the app
app = dash.Dash(__name__)
server = app.server

#Here comes the logic

#Load the numpy array from memory that contains the fourier coefficients
np_parameters = np.load('fourier coefficient matrix.npy')


pca=PCA(n_components=3)

pca.fit(np_parameters)

fourier_paramaters_pca = pca.transform(np_parameters)


def update_variance_graph():
 	return px.line(np.cumsum(pca.explained_variance_ratio_))



#Going to copy the container layout from https://github.com/plotly/dash-svm

app.layout = html.Div(children=[

	#Create a banner
	html.Div(className="banner", children=[
		# Change app name here?
		html.Div(className='container scalable', children=[
			# Change app name here??
			html.H2(html.A(
				'Fourier Coefficient PCA Vizualizer',
				href='https://github.com/jmontp/prosthetic-adaptation',
				style={
				'text-decoration':'none',
				'color':'inherit'
				}
			)),

			html.A(
				html.Img(src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"),
                href='https://plot.ly/products/dash/'
            )

		])
	]),

	html.Div(id='body', className='container scalable', children=[
		html.Div(
			className='three cloumns',
			style={
				'min-width': '24.5%',
            	'max-height': 'calc(100vh - 85px)',
                'overflow-y': 'auto',
                'overflow-x': 'hidden',
			},
			children=[
				html.Div(className='row', children=[
              		dcc.Graph(
							id='graph-fourier-pca',
						),
             	 	dcc.Graph(
					   		id='explained-variance',
							figure=update_variance_graph()
              		)]		
				),
            

				html.Div(className='row', children=[
                    drc.Card([
                        dcc.Input(
                            placeholder='Enter the amount of PCA axis...',
                            type='number',
                            value=3)
                    ]),
                    
                    drc.Card([     
						drc.NamedSlider(
							name='PCA axis 1',
							id='pca1',
							min=-50,
							max=50,
							step=1,
							marks={i: str(i) for i in range(-50,60,10)},
							value = 0
						),

						drc.NamedSlider(
								name='PCA axis 2',
								id='pca2',
								min=-50,
								max=50,
								step=1,
								marks={i: str(i) for i in range(-50,60,10)},
								value = 0
						),

						drc.NamedSlider(
							name='PCA axis 3',
							id='pca3',
							min=-50,
							max=50,
							step=1,
							marks={i: str(i) for i in range(-50,60,10)},
							value = 0
						),


						drc.NamedSlider(
							name='Step length (meters)',
							id='step-length',
							min=0,
							max=2,
							step=0.01,
							marks={i: str(i/10) for i in range(21)},
							value = 1
						)
					])
				])
			])
		])
	])



#This is the callback that updates the graphs
@app.callback(Output('graph-fourier-pca', 'figure'),
			 [Input('pca1','value'),
			  Input('pca2', 'value'), 
		      Input('pca3','value'), 
		      Input('step-length', 'value')])
def update_pca_graph(pca1_slider, pca2_slider, pca3_slider, step_length):

    
    #Recreate the phi and step length inputs
    phi=np.linspace(0,1,150)#.reshape(1,150)
    step_length_array=np.full((150,),step_length)
    num_params=12
       
    #Get the axis for the first three pca vectors
    pca1_axis=pca.components_[0]
    pca2_axis=pca.components_[1]
    pca3_axis=pca.components_[2]
    #Get the predicted y from the model
    y_pred=get_fourier_prediction(np_parameters[0]+pca1_slider*pca1_axis+pca2_slider*pca2_axis+ pca3_slider*pca3_axis,
                                  phi, 
                                  step_length_array,
                                  num_params)
    # print('Phi is ' + str(phi))
    # print('Step length is ' + str(step_length_array))
    # print('Params is '+str(np_parameters[0]))
    # print('Predicted y is ' + str(y_pred))
    
    return px.line(x=phi, y=y_pred)



#This is the callback for the explained variance
#@app.callback(Output('explained-variance', 'figure'))
# def update_variance_graph():
# 	return px.line(np.cumsum(pca.explained_variance_ratio_))






#Run the server
if __name__=='__main__':
	app.run_server(debug=True)