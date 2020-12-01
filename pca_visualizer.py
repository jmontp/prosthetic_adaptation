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

import math

from fourier_calculations import get_fourier_prediction



#Initialize the app
app = dash.Dash(__name__)
server = app.server

#Here comes the logic

#Load the numpy array from memory that contains the fourier coefficients
Ξ = np.load('fourier coefficient matrix.npy')
G = np.load('lambda Gram matrix.npy')	

# print("G", G)
eig, O = np.linalg.eigh(G)
V = np.diagflat(eig)

# print(eig.shape)
# print(O.shape)
# print((O.T @ V @ O).shape)
assert(np.linalg.norm(G-G.T)<1e-7)
for e in eig:
	assert (e>=0)
	assert( e>0) # pd


# print(np.linalg.norm(G - O @ V @ O.T)) # good
# print(np.linalg.norm(G - O.T @ V @ O)) # false
# print(np.linalg.norm(G - sum([O[:,[i]]@ O[:,[i]].T * eig[i] for i in range(len(eig))]))) # good
# print(np.linalg.norm(G)) # false



#This is based on the equation in eq:Qdef
Q = sum([O[:,[i]] @ O[:,[i]].T * 1/np.sqrt(eig[i]) for i in range(len(eig))])
Qinv = sum([O[:,[i]] @ O[:,[i]].T * np.sqrt(eig[i]) for i in range(len(eig))])
# Q G Q = I

# G = sum ( R_p.T @ R_p)
# prime G = I = Q G Q = sum (Q @ R_p.T @ R_p @ Q)
# Lambda_hat(x) = Lambda(x) @ Q
# theta = Lambda(x) * ξ  = Lambda_hat(x) [Q^{-1} ξ]

# assert that G = O (eig) O.T
assert(np.linalg.norm(G - O @ V @ O.T)< 1e-7 * np.linalg.norm(G)) # passes

# what is this matrix?
	# the matrix is literally \Xi
# print(Ξ.shape) # (10, 45)

#print("ξₐᵥ=", np.mean(Ξ, axis=0))
ξ_avg = np.mean(Ξ, axis=0)
Ξ0 = Ξ - ξ_avg
#print("Ξ0", Ξ0)


Ξ0prime = Ξ0 @ Q.T # basically done!


Σ = Ξ0prime.T @ Ξ0prime / (Ξ0prime.shape[0]-1)

ψs, U = np.linalg.eigh(Σ)
Ψ = np.diagflat(ψs)

assert(np.linalg.norm(Σ - U @ Ψ @ U.T)< 1e-7 * np.linalg.norm(Σ)) # passes
for i in range(len(ψs)-1):
	assert(ψs[i] < ψs[i+1])

#Define the amount principles axis that we want
η = 6
ss = []

for i in range (1,η+1):
	ss.insert(0, Qinv @ (U[:,-i]*np.sqrt(ψs[-i])))
	# print("At i = {}".format(i))
	# print("Shape of the Si is")
	# print(ss[i-1].shape)
	# print("The shape of Ui is")
	# print(U[:,-i].shape)

"""
"""
# print("ξ average")
# print(ξ_avg.shape)
# print("Test sum")
test = ss[0] + ss[1] + ss[2] + ss[3]
# print(test)
# print(test.shape)

# print("shape of Q")
# print(Q.shape)
# print("Shape of Qinv")
# print(Qinv.shape)

pca_input_list=[Input('pca'+str(i),'value') for i in range(η)]


def update_variance_graph():
 	return px.line(y=np.cumsum([0]+list(np.flip(ψs[-η:])/sum(ψs))), x=range(η+1), title='Cumulative Sum of variance', labels={'x': 'pca axis', 'y': 'percentage of variance covered'})


pca_sliders=[]

marker_step = 20
marker_min = 0
marker_range = 2
slider_marks1 = {int(i/marker_step):str(int(i/marker_step)) for i in range(-marker_range*marker_step, (marker_range+1)*marker_step,marker_step)}
print(slider_marks1)


pca_sliders=[drc.NamedSlider(
							name='Step length (meters)',
							id='step-length',
							min=.8,
							max=1.2,
							step=.4/marker_step,
							marks=slider_marks1,
							value = 1
						)]


marker_step = 20
marker_range = 3
slider_marks2 = {int(i/marker_step):str(int(i/marker_step)) for i in range(-marker_range*marker_step, (marker_range+1)*marker_step,marker_step)}
print(slider_marks2)


for i in range(η):
	pca_sliders+= [drc.NamedSlider(
							name='PCA axis '+str(i),
							id='pca'+str(i),
							min=-marker_range,
							max=marker_range,
							step=2*marker_range/marker_step,
							marks=slider_marks2,
							value = 0
						)]



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
			))

		# 	html.A(
		# 		html.Img(src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"),
  #               href='https://plot.ly/products/dash/'
  #           )

		# ])
	]),

	html.Div(id='body', className='container scalable', children=[
		html.Div(
			
			children=[
				html.Div(
					className='three cloumns', 
					style={'width': '49%', 'display': 'inline-block'},
					children=[
              		dcc.Graph(
							id='graph-fourier-pca',
						),
             	 	dcc.Graph(
					   		id='explained-variance',
							figure=update_variance_graph()
              		)]		
				),
            

				html.Div(
					className='three cloumns', 
					style={'width': '49%', 'display': 'inline-block'},
					children=[
                    drc.Card(pca_sliders)
				])
			])
		])
	])
])


#This is the callback that updates the graphs
@app.callback(Output('graph-fourier-pca', 'figure'),
			 [Input('step-length', 'value'),
			  *pca_input_list])
def update_pca_graph(step_length, *pca_argv):

    
    #Recreate the phi and step length inputs
    phi=np.linspace(0,1,150)#.reshape(1,150)
    step_length_array=np.full((150,),step_length)
    
    num_params=12
       
    #Get the axis for the first three pca vectors
    parameter_tuples = zip(ss,pca_argv)
    print("PCA argv")
    print(pca_argv)
    print("Axis")
    print([i.shape for i in ss])


    parameter=ξ_avg + sum([pca_axis*pca_slider_value for pca_axis,pca_slider_value in parameter_tuples])

    #Get the predicted y from the model
    y_pred=get_fourier_prediction(parameter,
                                  phi, 
                                  step_length_array,
                                  num_params)
    # print('Phi is ' + str(phi))
    # print('Step length is ' + str(step_length_array))
    # print('Params is '+str(Ξ[0]))
    # print('Predicted y is ' + str(y_pred))
    
    return px.line(x=phi, y=y_pred,title='Fourier Model Prediction', labels={'x': 'Phi', 'y': 'Thigh Angle'})




#Run the server
if __name__=='__main__':
	app.run_server(debug=True)
