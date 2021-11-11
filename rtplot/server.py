#Import communication
import zmq

#Import plotting
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

#Common imports 
import numpy as np 
import json
from time import perf_counter

#Create connection layer
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")


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

    win = pg.GraphicsLayoutWidget(title="Random Number Generator", show=True) # creates a window
   
    
    #Array of number per plot and array of pointer to plots
    subplot_per_plot = []
    subplots = []

    num_plots = 0
    top_plot = None
    top_plot_title = ""
    for plot_num, plot_description in enumerate(json_config.values()):
        
        #Add a plot
        num_plots += 1

        trace_names = plot_description['names']

        #Count how many traces we want
        num_traces = len(trace_names)
        
        #Add the indices in the numpy array
        subplot_per_plot.append(num_traces)

        #Initialize the new plot
        new_plot = win.addPlot()
        
        #Potential performance boost
        new_plot.setYRange(-10,10)
        new_plot.setXRange(0,WINDOW_WIDTH)

        if top_plot == None:
            top_plot = new_plot

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
            
            if plot_num == 0:
                top_plot_title = plot_description['title']


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
            new_curve = pg.PlotCurveItem(name=trace_names[i], pen=pen)
            new_plot.addItem(new_curve)
            subplots.append(new_curve)

    print("Initialized Plot!")
    return subplot_per_plot, subplots, num_plots, win, top_plot, top_plot_title


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
            subplot_per_plot, subplots, num_plots, win, top_plot, top_plot_title = initialize_plot(plot_configuration)
            Xm = np.zeros((sum(subplot_per_plot),WINDOW_WIDTH))    

            #Everything is initialized
            initialized_plot = True

            #Set counter index for debugging
            counter = 0

            #Define fps variable
            fps = None

            #Get last time to estimate fps
            lastTime = perf_counter()

           
        #Read some data and plot it
        else:

            #Read in numpy array
            receive_np_array = recv_array(socket)
            #Get how many new values are in it
            num_values = receive_np_array.shape[1]    

            #Remember how much you need to offset per plot
            subplot_offset = 0

            #Estimate fps
            now = perf_counter()
            dt = now - lastTime
            lastTime = now

            #Calculate the fps
            if fps is None:
                fps = 1.0/dt
            else:
                s = np.clip(dt*3., 0, 1)
                fps = fps * (1-s) + (1.0/dt) * s

            #Plot for every subplot
            for plot_index in range(num_plots):
                
                for subplot_index in range(subplot_per_plot[plot_index]):
                    i = subplot_offset + subplot_index
                    Xm[i,:-num_values] = Xm[i,num_values:]    # shift data in the temporal mean 1 sample left
                    Xm[i,-num_values:] = receive_np_array[i,:]              # vector containing the instantaneous values  
                    subplots[i].setData(Xm[i,:])
                
                #Update before the next loop
                subplot_offset += subplot_per_plot[plot_index]

            #Update fps in title
            top_plot.setTitle(top_plot_title + f" - FPS:{fps:.0f}")

            #Indicate you MUST process the plot now
            QtGui.QApplication.processEvents()    

          

except KeyboardInterrupt:
    print("You can move around the plot now")
    #Need to run this for 
    QtGui.QApplication.instance().exec_()



#References
#ZMQ Example code
#https://zeromq.org/languages/python/

#How to send/receive numpy arrays
#https://pyzmq.readthedocs.io/en/latest/serialization.html

#How to real time plot with pyqtgraph
#https://stackoverflow.com/questions/45046239/python-realtime-plot-using-pyqtgraph