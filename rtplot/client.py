#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import numpy as np
import json 
import time


context = zmq.Context()

#Socket to talk to server
socket = context.socket(zmq.PUB)

#Neurobionics Pi Address
#socket.connect("udp://10.0.0.24:5555")

#Local testing address
socket.bind("tcp://127.0.0.1:5555")
#Sleep so that subscribers can join
time.sleep(0.2)


def send_array(A, flags=0, copy=True, track=False):
    """send a numpy array with metadata
    Inputs
    ------
    A: np array to transmit
    """

    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def initialize_plots(plot_descriptions):
    """Send a json description of desired plot
    Inputs
    ------
    plot_description: list of names or list of plot descriptions 
    """

    #Process list of names
    if type(plot_descriptions[0]) == list:
        plot_desc_dict = {f"plot{i}":{"names":desc} for i,desc in enumerate(plot_descriptions)}
        plot_desc_json = json.dumps(plot_desc_dict)
    
    #Process list of dics
    elif type(plot_descriptions[0]) == dict:
        plot_desc_dict = {f"plot{i}":desc for i,desc in enumerate(plot_descriptions)}
        plot_desc_json = json.dumps(plot_desc_dict)
    
    #Throw error
    else:
        raise TypeError("Only List of List and Dict are supported")

    socket.send_json(plot_desc_json)



def main():

    #  Do 10 requests, waiting each time for a response
    #Configure the plot
    plot1_names = ['phase', 'phase_dot', 'stride_length']
    plot2_names = [f"gf{i+1}" for i in range(5)]



    plot_1_config = {'names': ['phase', 'phase_dot', 'stride_length'],
                    'title': "Phase, Phase Dot, Stride Length",
                    'ylabel': "reading (unitless)",
                    'xlabel': 'test 1'}

    plot_2_config = {'names': [f"gf{i+1}" for i in range(1)],
                    'colors' : ['b' for i in range(1)],
                    'line_style' : ['-','','-','','-']*4,
                    'title': "Phase, Phase Dot, Stride Length",
                    'ylabel': "reading (unitless)",
                    'xlabel': 'test 2'}

    total_plots = len(plot_1_config['names']) + len(plot_2_config['names'])
    # total_plots = 1

    initialize_plots([plot_1_config,plot_2_config])
    # initialize_plots([['phase']])
    print("Sent Plot format")
    time.sleep(1)

    for request in range(1000):

        print("Sending request %s" % request)
        arr = np.linspace(request,request+1,1)
        arr2 = np.repeat(arr,total_plots).reshape(total_plots,1)
        send_array(np.sin(arr2))



if __name__ == '__main__':
    main()