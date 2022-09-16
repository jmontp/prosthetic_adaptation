from cProfile import label
import pandas as pd
import numpy as np 
from manim import *

joint = 'jointangles_thigh_x'
filename = '../../data/flattened_dataport/dataport_flattened_partial_{}.parquet'

#Get data for one subject
subject = 'AB02'

data = pd.read_parquet(filename.format(subject))

ramp = 0

mask = (data['ramp'] == ramp) #& (data['speed'] == speed)
step_data = data[mask][joint].values.reshape(-1,150)
all_step_data = None

#Get data for all the subject


subjects = [f'AB{i:02}' for i in range(1,11)]
datas = [pd.read_parquet(filename.format(subb)) for subb in subjects]
#All the data, all subjects
# step_data = np.concatenate([d[joint].values.reshape(-1,150) for d in datas], axis=0)

#Filtered by ramp, all subjects
# mask_list = [(d['ramp'] == ramp) & (d['speed'] == speed) for d in datas]
mask_list = [(d['ramp'] == ramp) for d in datas]
all_step_data = np.concatenate([d[mask_i][joint].values.reshape(-1,150)[::5] for d,mask_i in zip(datas,mask_list)], axis=0)
num_all_steps = all_step_data.shape[0]

y_min = all_step_data.min()
y_max = all_step_data.max()

phase = np.linspace(0,100,150)

mean_step = step_data.mean(axis=0)

class ManimPlotter(Scene):

    def construct(self):

        #Define the axes
        axes = Axes(
            
            y_range = [y_min, y_max + 1, 30],
            x_range = [0,100,50],
            tips = False,
            x_axis_config={
                # "numbers_to_include": [0,50,100],
                # "include_ticks": True,
                "include_numbers":False,
            },
            y_axis_config={
                # "stroke_width": 0,
                "numbers_to_include": [0,90],
                "include_numbers":False,
                "include_ticks": False,

            },

        )
        
        self.add(axes)
        self.wait(2)
        graph = axes.plot_line_graph(x_values=phase, y_values=mean_step, add_vertex_dots=False, line_color='#b3e766')
        self.play(Write(graph))        
        self.wait(2)
        num_graphs = step_data.shape[0]
        graphs = VGroup(*[axes.plot_line_graph(x_values=phase, y_values=step_data[i,:],add_vertex_dots=False,stroke_opacity=0.2,line_color='#ffab40')["line_graph"] for i in range(1,num_graphs)])


        self.play(Create(graphs,lag_ratio=0.1,run_time=2.5))
        self.wait(2)

        print(f"all save data {num_all_steps}")
        graphs_all_data = VGroup(*[axes.plot_line_graph(x_values=phase, y_values=all_step_data[i,:],add_vertex_dots=False,stroke_opacity=0.2,line_color='#ffab40')["line_graph"] for i in range(1,num_all_steps)])
        self.play(Create(graphs_all_data,lag_ratio=0.1,run_time=2.5))
        self.wait(2)

        graph = axes.plot_line_graph(x_values=phase, y_values=mean_step, add_vertex_dots=False, line_color='#b3e766')
        self.play(Write(graph))        
        self.wait(3)

