"""
This file is meant to plot a log-log heatmap of each gait state error against
each combination of phase error     
    
"""

#%%
import altair as alt
import pandas as pd
import numpy as np 
from decimal import Decimal
from first_paper_utils import filter_to_good_range_case, phase_rate_noise, \
    stride_length_noise, ramp_noise

#Allow plotting large datasets
alt.data_transformers.disable_max_rows()

#Read in data
data_location = '../ekf_sim/first_paper_NLS_vs_ISA.csv'
df = pd.read_csv(data_location)

#Name the gait state rmse entries in the dataset
phase_rmse = 'phase'
stride_length_rmse = 'stride_length'
ramp_rmse = 'ramp'

#Create list of gait state rmse
gait_state_rmser = [phase_rmse, stride_length_rmse, ramp_rmse]

#Filter the data for log bining
df = filter_to_good_range_case(df,filter=False,inplace=True)

#%%
#Convert to new naming scheme for easy plot legibility
#Set the new names
new_phase_rate_noise = 'Phase Rate Noise'
new_stride_length_noise = 'Stride Length Noise'
new_ramp_noise = 'Ramp Noise'
#Create a list for old and new
new_name_list = [new_phase_rate_noise,new_stride_length_noise,new_ramp_noise]
old_name_list = [phase_rate_noise,stride_length_noise,ramp_noise]
#Create dictionary for conversion
convert_names_dict = {old:new for old,new in zip(old_name_list,new_name_list)}
#Change names
df.rename(columns=convert_names_dict,inplace=True)

#Change the variables
phase_rate_noise = new_phase_rate_noise
stride_length_noise = new_stride_length_noise
ramp_noise = new_ramp_noise


#Create a list of conditions to plot over in (x,y)
conditions = [(phase_rate_noise, stride_length_noise),
              (phase_rate_noise, ramp_noise),
              (stride_length_noise, ramp_noise)]

#Axis layout
axis=alt.Axis(
    #tickCount=10, 
    format=".1e"
)
bin=alt.Bin(
    #maxbins=20
)
scale=alt.Scale(
    #type='log'
)

def create_heatmap(x_label,y_label,z_label):
    
    #Decide Scheme
    if z_label == phase_rmse:
        scheme = 'purplegreen'
    elif z_label == stride_length_rmse:
        scheme = 'pinkyellowgreen'
    else: # z_label == ramp_rmse:
        scheme = 'redyellowgreen'
    
    #Override all of them
    scheme = 'redyellowgreen'

    
    chart = alt.Chart(df).mark_rect().encode(
            x=alt.X(f"{x_label}:O", axis=axis),#, scale=scale),
            y=alt.Y(f"{y_label}:O", axis=axis, scale=alt.Scale(reverse=True)),
            color=alt.Color(f'median({z_label}):Q', 
                            scale=alt.Scale(scheme=scheme,
                                            reverse=True)),
            tooltip=f'median({z_label}):Q'
        ).properties(
            width=200,
            height=200
        ).interactive()
    
    return chart

#Holy one liner
total_chart = alt.vconcat(
    *[alt.hconcat(*[create_heatmap(x,y,gait_state) 
                  for x,y 
                  in conditions])#.resolve_scale(color='independent')
    for gait_state
    in gait_state_rmser]
).configure_axis(
    labelFontSize=20,
    titleFontSize=20,
).configure_legend(
    titleFontSize=18,
    labelFontSize=15
).resolve_scale(color='independent')

total_chart.show()
 # %%
