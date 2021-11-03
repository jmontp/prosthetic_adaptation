#Import communication
import zmq

#Import plotting
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

#Common imports 
import numpy as np 
import json


#Create connection layer
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


### START QtApp #####
# you MUST do this once (initialize things)
app = QtGui.QApplication([])            
# ####################

# width of the window displaying the curve
WINDOW_WIDTH = 200                      

#Create the plot from the json file that is passed in
def initialize_plot(json_config):
    
    #Set background white
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    win = pg.GraphicsWindow(title="Random Number Generator") # creates a window
   
    
    #Array of number per plot and array of pointer to plots
    subplot_per_plot = []
    subplots = []

    num_plots = 0

    for plot_description in json_config.values():
        
        #Add a plot
        num_plots += 1

        trace_names = plot_description['names']

        #Count how many traces we want
        num_traces = len(trace_names)
        
        #Add the indices in the numpy array
        subplot_per_plot.append(num_traces)

        #Initialize the new plot
        new_plot = win.addPlot()
        win.nextRow()

        #Add the names of the plots to the legend
        new_plot.addLegend()

        #Add the axis info
        if 'xlabel' in plot_description:
            new_plot.setLabel('bottom', plot_description['xlabel'])

        if 'ylabel' in plot_description:
            new_plot.setLabel('left', plot_description['ylabel'])

        #Add title
        if 'title' in plot_description:
            new_plot.setTitle(plot_description['title'])


        #Define default Style
        colors = ['r','g','b','c','m','y']
        if 'colors' in plot_description:
            colors = plot_description['colors']

        line_style = [QtCore.Qt.SolidLine] * 10
        if 'line_style' in plot_description:
            line_style = [QtCore.Qt.DashLine if desc == '-' else QtCore.Qt.SolidLine for desc in plot_description['line_style']]

        for i in range(num_traces):
            #Add the plot object
            pen = pg.mkPen(color = colors[i], style=line_style[i])
            subplots.append(new_plot.plot(name=trace_names[i], pen=pen))

    print("Initialized Plot!")
    return subplot_per_plot, subplots, num_plots, win


#Receive a numpy array
def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])




try:
    initialized_plot = False
    while True:
        #Do not continue unless you have initialized the plot
        if(not initialized_plot):
            flags = 0 
            rec_json = socket.recv_json(flags=flags)
            plot_configuration = json.loads(rec_json)
            subplot_per_plot, subplots, num_plots, win = initialize_plot(plot_configuration)
            Xm = np.zeros((sum(subplot_per_plot),WINDOW_WIDTH))    

            #Everything is initialized
            initialized_plot = True


        #Read some data and plot it
        else:
            print("Waiting for data...")
            #Read in numpy array
            receive_np_array = recv_array(socket)
            #Get how many new values are in it
            num_values = receive_np_array.shape[1]    

            #Remember how much you need to offset per plot
            subplot_offset = 0

            #Plot for every subplot
            for plot_index in range(num_plots):

                for subplot_index in range(subplot_per_plot[plot_index]):
                    i = subplot_offset + subplot_index
                    Xm[i,:-num_values] = Xm[i,num_values:]    # shift data in the temporal mean 1 sample left
                    Xm[i,-num_values:] = receive_np_array[i,:]              # vector containing the instantaneous values      
                    subplots[i].setData(Xm[i,:])
                
                #Update before the next loop
                subplot_offset += subplot_per_plot[plot_index]

            #Indicate you MUST process the plot now
            QtGui.QApplication.processEvents()    


            #Print received message shape
            print(f"Received request: {receive_np_array.shape}")



        #Whenever you get a message, reply something to make tcp happy
        socket.send(b"World")


except KeyboardInterrupt:
    print("You can move around the plot now")
    pass


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



#References
#ZMQ Example code
#https://zeromq.org/languages/python/

#How to send/receive numpy arrays
#https://pyzmq.readthedocs.io/en/latest/serialization.html

#How to real time plot with pyqtgraph
#https://stackoverflow.com/questions/45046239/python-realtime-plot-using-pyqtgraph