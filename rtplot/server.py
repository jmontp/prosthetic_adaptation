# Import communication
import zmq

# Import plotting
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

# Common imports 
import numpy as np 

# Get timer to calculate fps
from time import perf_counter

# Import argparse to handle different configurations
# of the plotter
import argparse


############################
# Command Line Arguments #
###########################

# Create command line arguments
parser = argparse.ArgumentParser()

#Add argument to enable bigger fonts
parser.add_argument("--bigscreen", help="Increase fonts to print in the big screen",
                    action="store_true")
args = parser.parse_args()


# If big screen mode is on, set font sizes big
if args.bigscreen:
    axis_label_style = {'font-size':'20pt'}
    title_style = {'size':'25pt'}
    #Accepts parameters into LegendItem constructor
    legend_style = {'labelTextSize':'14pt'}
    tick_size = 25
    

# Else set to normal size
else:
    axis_label_style = {'font-size':'10pt'}
    title_style = {'size':'14pt'}
    legend_style = {'labelTextSize':'8pt'}
    tick_size = 12



###################
# ZMQ Networking #
##################

# Create connection layer
context = zmq.Context()

# Using the pub - sub paradigm
socket = context.socket(zmq.SUB)
socket.bind("tcp://*:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

############################
# PyQTgraph Configuration #
###########################


### START QtApp #####
# You MUST do this once (initialize things)
app = QtGui.QApplication([])            

# Width of the window displaying the curve
WINDOW_WIDTH = 200                      

# Window title
WINDOW_TITLE = "Real Time Plotter"

# Define if a new subplot is placed in a 
# new row or columns
NEW_SUBPLOT_IN_ROW = True

# Set background white
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Define the window object for the plot
win = pg.GraphicsLayoutWidget(title=WINDOW_TITLE, show=True)


# Create the plot from the json file that is passed in
def initialize_plot(json_config, subplots_to_remove=None):
    """Initializes the plots and returns many handles to plot items

    Inputs
    ------

    json_config: Python dictionary with relevant plot configuration
    subplots_to_remove: Previous plot subplot items that will be removed from the window

    Returns
    -------
    traces_per_plot: num of traces per each subplot
    subplots_traces: Object that is used to update the traces
    subplots: Handle to subplots used to delete the subplots uppon re-initialization
    num_plots: Number of subplots
    top_plot: Reference to top plot object to update title of
    top_plot_title: Reference to top plot title string to add on FPS
    """
    

    # If there are old subplots, remove them
    if subplots_to_remove is not None:
        for subplot in subplots_to_remove:
            win.removeItem(subplot)

    # Initialize arrays of number per plot and array of pointer to plots and traces
    traces_per_plot = []
    subplots_traces = []
    subplots = []

    # Initialize the top plot to None so that we can grab it
    top_plot = None

    # Initialize top plot title in case the user does not provide a title
    top_plot_title = ""

    # Generate each subplot
    for plot_num, plot_description in enumerate(json_config.values()):
        
        # Get the trace names for this plot
        trace_names = plot_description['names']

        # Count how many traces we want
        num_traces = len(trace_names)
        
        # Add the indices in the numpy array
        traces_per_plot.append(num_traces)

        # Initialize the new plot
        new_plot = win.addPlot()
        
        # Move to the next row
        if NEW_SUBPLOT_IN_ROW == True:
            win.nextRow()
        else:
            win.nextCol()

        # Capture the first plot
        if top_plot == None:
            top_plot = new_plot

        # Add the names of the plots to the legend
        new_plot.addLegend(**legend_style)

        # Add the axis info
        if 'xlabel' in plot_description:
            new_plot.setLabel('bottom', plot_description['xlabel'],**axis_label_style)

        if 'ylabel' in plot_description:
            new_plot.setLabel('left', plot_description['ylabel'],**axis_label_style)

        # Potential performance boost
        new_plot.setXRange(0,WINDOW_WIDTH)

        # Get the y range
        if 'yrange' in plot_description:
            new_plot.setYRange(*plot_description['yrange'])
        
        # Set axis tick mark size
        font=QtGui.QFont()
        font.setPixelSize(tick_size)
        # New_plot.getAxis("left").tickFont = font
        new_plot.getAxis("bottom").setStyle(tickFont = font)

        font=QtGui.QFont()
        font.setPixelSize(tick_size)
        # New_plot.getAxis("bottom").tickFont = font
        new_plot.getAxis("bottom").setStyle(tickFont = font)


        # Add title
        if 'title' in plot_description:
            new_plot.setTitle(plot_description['title'],**title_style)
            
            if plot_num == 0:
                top_plot_title = plot_description['title']
        
        # If zeroth-plot does not have tittle, add something in blank
        # so fps counter gets style
        elif plot_num == 0:
            new_plot.setTitle("",**title_style)


        # Define default Style
        colors = ['r','g','b','c','m','y']
        if 'colors' in plot_description:
            colors = plot_description['colors']

        line_style = [QtCore.Qt.SolidLine] * num_traces
        if 'line_style' in plot_description:
            line_style = [QtCore.Qt.DashLine if desc == '-' else QtCore.Qt.SolidLine for desc in plot_description['line_style']]

        line_width = [1] * num_traces
        if 'line_width' in plot_description:
            line_width = plot_description['line_width']

        # Generate all the trace objects
        for i in range(num_traces):
            # Create the pen object that defines the trace style
            pen = pg.mkPen(color = colors[i], style=line_style[i], width=line_width[i])
            # Add new curve
            new_curve = pg.PlotCurveItem(name=trace_names[i], pen=pen)
            new_plot.addItem(new_curve)
            # Store pointer to update later
            subplots_traces.append(new_curve)

        # Add the new subplot
        subplots.append(new_plot)

    print("Initialized Plot!")
    return traces_per_plot, subplots_traces, subplots, top_plot, top_plot_title


# Receive a numpy array
def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


# Create definitions to define when you receive data or new plots
RECEIVED_PLOT_UPDATE = 0
RECEIVED_DATA = 1

# Define function to detect category
def rec_type():
    # Sometimes we get miss-aligned data
    # In this case just ignore the data and wait until you have a valid type
    while True:
        received = socket.recv_string()
        try:
            return int(received)
        except ValueError:
            print(f"Had a value error. Expected int, received: {received}")
            pass


#####################
# Main code section #
#####################

# Run until you get a keyboard interrupt
try:
    # Initialize variables

    # Initialize plots expected the old plots to delete them
    # since we have no plots, initialize to None
    subplots = None
    

    # Make sure that you don't try to plot data without having a plot
    initialized_plot = False

    # Main code loop
    while True:
        # Receive the type of information
        category = rec_type()

        # Do not continue unless you have initialized the plot
        if(category == RECEIVED_PLOT_UPDATE):
            
            # Receive plot configuration
            flags = 0 
            plot_configuration = socket.recv_json(flags=flags)

            # Initialize plot
            traces_per_plot, subplots_traces, subplots,\
            top_plot, top_plot_title \
                    = initialize_plot(plot_configuration, subplots)
            
            # Get the number of plots
            num_plots = len(subplots)

            # Initialize data buffer
            Xm = np.zeros((sum(traces_per_plot),WINDOW_WIDTH))    

            # You can now plot data
            initialized_plot = True

            # Define fps variable
            fps = None

            # Get last time to estimate fps
            lastTime = perf_counter()

           
        # Read some data and plot it
        elif (category == RECEIVED_DATA) and (initialized_plot == True):

            # Read in numpy array
            receive_np_array = recv_array(socket)
            # Get how many new values are in it
            num_values = receive_np_array.shape[1]    

            # Remember how much you need to offset per plot
            subplot_offset = 0

            # Estimate fps
            now = perf_counter()
            dt = now - lastTime
            lastTime = now

            if fps is None:
                fps = 1.0/dt
            else:
                s = np.clip(dt*3., 0, 1)
                fps = fps * (1-s) + (1.0/dt) * s

            #Update every subplot
            for plot_index in range(num_plots):
                
                # Update every trace
                for subplot_index in range(traces_per_plot[plot_index]):
                    # Get index to plot
                    i = subplot_offset + subplot_index
                    # Update the rolling buffer with new values
                    Xm[i,:-num_values] = Xm[i,num_values:]    
                    Xm[i,-num_values:] = receive_np_array[i,:]
                    # Set the data in the trace              
                    subplots_traces[i].setData(Xm[i,:])
                
                # Update offset to account for the past loop's traces
                subplot_offset += traces_per_plot[plot_index]

            # Update fps in title
            top_plot.setTitle(top_plot_title + f" - FPS:{fps:.0f}")
            # Indicate you MUST process the plot now
            QtGui.QApplication.processEvents()    

          

except KeyboardInterrupt:

    try: 
        win
        print("You can move around the plot now")
        QtGui.QApplication.instance().exec_()
    except:
       print("\nNo plot - killing server")


#References
#ZMQ Example code
#https://zeromq.org/languages/python/

#How to send/receive numpy arrays
#https://pyzmq.readthedocs.io/en/latest/serialization.html

#How to real time plot with pyqtgraph
#https://stackoverflow.com/questions/45046239/python-realtime-plot-using-pyqtgraph
