#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import numpy as np
import json 


context = zmq.Context()

#Socket to talk to server
socket = context.socket(zmq.REQ)

#Neurobionics Pi Address
#socket.connect("udp://10.0.0.24:5555")

#Local testing address
socket.connect("tcp://127.0.0.1:5555")


def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)





def initialize_plots(plot_descriptions):

    if type(plot_descriptions[0]) == list:
        plot_desc_dict = {f"plot{i}":{"names":desc} for i,desc in enumerate(plot_descriptions)}
        plot_desc_json = json.dumps(plot_desc_dict)
    elif type(plot_descriptions[0]) == dict:
        plot_desc_dict = {f"plot{i}":desc for i,desc in enumerate(plot_descriptions)}
        plot_desc_json = json.dumps(plot_desc_dict)
    else:
        raise TypeError("Only List of List and Dict are supported")

    socket.send_json(plot_desc_json)



def main():

    #  Do 10 requests, waiting each time for a response
    #Configure the plot
    plot1_names = ['phase', 'phase_dot', 'stride_length']
    plot2_names = [f"gf{i+1}" for i in range(5)]

    total_plots = len(plot1_names) + len(plot2_names)


    plot_1_config = {'names': ['phase', 'phase_dot', 'stride_length'],
                    'title': "Phase, Phase Dot, Stride Length",
                    'ylabel': "reading (unitless)",
                    'xlabel': 'test 1'}

    plot_2_config = {'names': [f"gf{i+1}" for i in range(5)],
                    'colors' : ['w' for i in range(5)],
                    'line_style' : ['-','','-','','-'],
                    'title': "Phase, Phase Dot, Stride Length",
                    'ylabel': "reading (unitless)",
                    'xlabel': 'test 2'}

    initialize_plots([plot_1_config,plot_2_config])

    #  Get the reply.
    message = socket.recv()
    print("Received reply %s [ %s ]" % (0, message))

    for request in range(10):

        print("Sending request %s" % request)
        send_array(socket, np.random.randn(total_plots,10))

        #  Get the reply.
        message = socket.recv()
        print("Received reply %s [ %s ]" % (request, message))


if __name__ == '__main__':
    main()