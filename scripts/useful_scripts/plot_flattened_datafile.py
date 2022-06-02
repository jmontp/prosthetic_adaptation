from operator import sub
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

# Configuration of plot 
################################################################################################


subject = 'AB02'
joint = 'jointangles_thigh_x'
filename = '../../data/flattened_dataport/dataport_flattened_partial_{}.parquet'

#Conditions that we want to filter by 
ramp = 0
speed = 1.0


# Create Boolean flags to determine what to plot
ONE_SUBJECT = True      #False plots all the subjects
FILTER_SPEED = False
FILTER_RAMP = False    
AVERAGE_STEP = False   #True calculates the average step across people instead of the individual steps
ADD_STD_DEV = False    #Adds plus and minus two std deviation 
NUMBER_OF_LINES_TO_PLOT = 1000
LOW_PASS_FILTER = False

plot_x_label = "phase"
plot_y_label = joint

################################################################################################



#Get data for one subject
if (ONE_SUBJECT):
    #Get data
    data = pd.read_parquet(filename.format(subject))
    #Filter data
    if(FILTER_RAMP):
        mask = (data['ramp'] == ramp)
        data = data[mask]
    if(FILTER_SPEED):
        mask = (data['speed'] == speed)
        data = data[mask]
    #Convert pandas to numpy array
    if(AVERAGE_STEP):
        step_data = data[joint].values.reshape(-1,150).mean(axis=0).reshape(1,150)
    else:
        step_data = data[joint].values.reshape(-1,150)



#Get data for all the subjects
else:
    #Get data
    subjects = [f'AB{i:02}' for i in range(1,11)]
    datas = [pd.read_parquet(filename.format(subb)) for subb in subjects]
    #Filter Data
    if(FILTER_RAMP):
        mask_list = [(d['ramp'] == ramp) for d in datas]
        datas = [data[mask] for data,mask in zip(datas,)]
    if(FILTER_SPEED):
        mask_list = [(d['speed'] == speed) for d in datas]
        datas = [data[mask] for data,mask in zip(datas,)]
    if(AVERAGE_STEP):
        step_data = np.concatenate([data[joint].values.reshape(-1,150).mean(axis=0).reshape(1,150) for data in datas], axis=0)
    else:
        step_data = np.concatenate([data[joint].values.reshape(-1,150) for data in datas], axis=0)


    #Define the x axis
    phase = np.linspace(0,100,150)

#Plot the average step
if(AVERAGE_STEP):
    
    #Calculate the mean and stendard deviation at each point in phase
    mean = step_data.mean(axis=0)
    std = step_data.std(axis=0)
    # mean_list = [data[mask][output_name].values.reshape(-1,150).mean(axis=0) for output_name in [joint]]
    # std_list = [data[mask][output_name].values.reshape(-1,150).std(axis=0) for output_name in [joint]]
    # mean = mean_list[0]
    # std = std_list[0]
    #Plot mean line
    plt.plot(phase,mean)
    #Add standard deviation line
    plt.plot(phase,mean + 2*std)
    #Add standard deviation line
    plt.plot(phase,mean - 2*std)

#Display all the steps
else:

    #Make sure we are not plotting more than we have
    if (NUMBER_OF_LINES_TO_PLOT > 0):
        lines_to_plot = min(NUMBER_OF_LINES_TO_PLOT, step_data.shape[0])
    else:
        lines_to_plot = step_data.shape[0]

    #Plot all the steps individually
    for i in range(lines_to_plot):

        #Get current step
        curr_data = step_data[i,:]

        #Add low pass filtering
        if(LOW_PASS_FILTER):
            fsig = np.fft.fft(curr_data)
            fsig[20:] = 0
            curr_data = np.fft.ifft(fsig)

        #Plot step
        plt.plot(phase,curr_data)

   
#Add plot info
plt.xlabel(plot_x_label)
plt.ylabel(plot_y_label)
plt.show()