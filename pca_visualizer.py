#Fourier Coefficient PCA vizualizer using Dash
# Relevant references
#   Dash tutorial: https://www.youtube.com/watch?v=hSPmj7mK6ng&ab_channel=CharmingData
#   Did similar layout to: https://github.com/plotly/dash-svm/blob/master/app.py


#Plotting
import dash
import dash_core_components as dcc
import dash_html_components as html
import utils.dash_reusable_components as drc
from dash.dependencies import Input,Output,State
import plotly.express as px
import pandas as pd


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

#Verify that the G matrix is at least positive semidefinite
#To be psd or pd, G=G^T
assert(np.linalg.norm(G-G.T)<1e-7)



#Diagonalize the matrix G as G = OVO
eig, O = np.linalg.eigh(G)
V = np.diagflat(eig)


#Additionally, all the eigenvalues are true
for e in eig:
	assert (e>=0)
	assert( e>0) # pd

# Verify that it diagonalized correctly G = O (eig) O.T
assert(np.linalg.norm(G - O @ V @ O.T)< 1e-7 * np.linalg.norm(G)) # passes
# print(np.linalg.norm(G - O @ V @ O.T)) # good
# print(np.linalg.norm(G - O.T @ V @ O)) # false
# print(np.linalg.norm(G - sum([O[:,[i]]@ O[:,[i]].T * eig[i] for i in range(len(eig))]))) # good
# print(np.linalg.norm(G)) # false



#This is based on the equation in eq:Qdef
# Q G Q = I
Q       = sum([O[:,[i]] @ O[:,[i]].T * 1/np.sqrt(eig[i]) for i in range(len(eig))])
Qinv    = sum([O[:,[i]] @ O[:,[i]].T * np.sqrt(eig[i]) for i in range(len(eig))])


# IF we had to calculate G by hand:
# G = sum ( R_p.T @ R_p)
# prime G = I = Q G Q = sum (Q @ R_p.T @ R_p @ Q)
# Lambda_hat(x) = Lambda(x) @ Q
# theta = Lambda(x) * ξ  = Lambda_hat(x) [Q^{-1} ξ]



#print("ξₐᵥ=", np.mean(Ξ, axis=0))
ξ_avg = np.mean(Ξ, axis=0)

#Substract the average row
Ξ0 = Ξ - ξ_avg

#Calculate the coefficients in the orthonormal space

Ξ0prime = Ξ0 @ Qinv


Σ = Ξ0prime.T @ Ξ0prime / (Ξ0prime.shape[0]-1)

ψinverted, Uinverted = np.linalg.eigh(Σ)
#Eigenvalues are received from smalles to bigger, make it bigger 
#to smaller
ψs = np.flip(ψinverted)
Ψ = np.diagflat(ψs)
#If we change the eigenvalues we also need to change the eigenvectors
U = np.flip(Uinverted, axis=1)

#Run tests to make sure that this is working
assert(np.linalg.norm(Σ - U @ Ψ @ U.T)< 1e-7 * np.linalg.norm(Σ)) # passes
for i in range(len(ψs)-1):
	assert(ψs[i] > ψs[i+1])

#Define the amount principles axis that we want
η = 6
ss = []


#Convert from the new basis back to the original basis vectors
for i in range (0,η):
	ss.append(Qinv @ (U[:,i]*np.sqrt(ψs[i])))
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
#test = ss[0] + ss[1] + ss[2] + ss[3]
# print(test)
# print(test.shape)

# print("shape of Q")
# print(Q.shape)
# print("Shape of Qinv")
# print(Qinv.shape)

#Make sure that our new basis meets the criteria that we had
#To be more specific, make sure that the rms error is close to one for
#different unit vectors
num_params=ξ_avg.shape[0]
#Recreate the phi and step length inputs
phi=np.linspace(0,1,150)#.reshape(1,150)
step_length_array=np.full((150,),1)
for i in range(num_params):
    ξtest = np.zeros(num_params)
    ξtest[i] = 1
    test_deviation_function = get_fourier_prediction(Qinv @ ξtest,
    											phi,
    											step_length_array,
    											12)
    test_rmse = np.sqrt(np.mean(np.square(test_deviation_function)))
    print(ξtest)
    print(test_rmse)
    #assert(1-1e-3 < abs(test_rmse) < 1+1e-3)



pca_input_list=[Input('pca'+str(i),'value') for i in range(η)]


def update_variance_graph():
 	return px.line(y=np.cumsum([0]+list((ψs[0:η])/sum(ψs))), x=range(η+1), title='Cumulative Sum of variance', labels={'x': 'pca axis', 'y': 'percentage of variance covered'})


pca_sliders=[]

marker_step = 6
marker_min = 0.8
marker_range = 1.2

slider_marks1 = {i: "{0:.2f}".format(i) for i in np.append([0],np.linspace(marker_min, marker_range, num=marker_step))}
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


marker_step = 6
marker_range = 3
slider_marks2 = {i: str(i) for i in range(-marker_range, marker_range)}
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
    # print("PCA argv")
    # print(pca_argv)
    # print("Axis")
    # print([i.shape for i in ss])

    #Calculate the deviation from the average basis vector
    deviation_from_average = sum([pca_axis*pca_slider_value for pca_axis,pca_slider_value in parameter_tuples])

    average_function = get_fourier_prediction(ξ_avg,
    										  phi,
    										  step_length_array,
    										  num_params)

    deviation_function = get_fourier_prediction(deviation_from_average,
    											phi,
    											step_length_array,
    											num_params)

    rmse = np.sqrt(np.mean(np.square(deviation_function)))

    #print("The rmse for this function from the average is: " + str(rmse))

    parameter=ξ_avg + deviation_from_average

    #Get the predicted y from the model
    # y_pred=get_fourier_prediction(parameter,
    #                               phi, 
    #                               step_length_array,
    #                               num_params)
    #print("Average function + deviation function - y_pred" + str(np.cumsum(average_function + deviation_function - y_pred)))

    y_pred = average_function + deviation_function

    # print('Phi is ' + str(phi))
    # print('Step length is ' + str(step_length_array))
    # print('Params is '+str(Ξ[0]))
    # print('Predicted y is ' + str(y_pred))
    


    #Construct a dataframe to ease plotting
    df = pd.DataFrame({"x": phi,
    	               "Individualized Function": y_pred, 
    	               "Average Function": average_function,
    	               "Deviation Function": deviation_function})

    df_melt = df.melt(id_vars="x", value_vars=["Individualized Function", "Average Function", "Deviation Function"])

    fig = px.line(df_melt, 
    			  x="x",
    			  y="value", 
    			  color='variable',
    			  title='Fourier Model Prediction', 
    			  labels={'x': 'Phi', 'value': 'Thigh Angle'})

    fig.update_yaxes(
    range=[-60,80],  # sets the range of xaxis
    constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
	)

    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
	))
    
    fig.update_layout(legend_title_text="Deviation RMSE: {0:.2f}    ".format(rmse))

    return fig




#Run the server
if __name__=='__main__':
	app.run_server(debug=True)
