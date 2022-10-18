"""
This file is meant to plot the summary statistics (using all the data) 
for the first test in the offline ekf paper
"""


import altair as alt
import pandas as pd 
from first_paper_utils import filter_to_good_range_case, filter_to_optimal_case
from first_paper_utils import phase_rate_noise, stride_length_noise, ramp_noise

#Allow plotting large datasets
alt.data_transformers.disable_max_rows()

#Read the data in 
data_location = '../ekf_sim/first_paper_NLS_vs_ISA.csv'
df, title = pd.read_csv(data_location), 'All Data'



#Filter data to optimal case
# df, title = filter_to_good_range_case(df), '"Good" Range'
df, title = filter_to_optimal_case(df,test_type='all'), 'Optimal Case'

#Create the title
title=f"Histogram of RMSE in {title}"



#Gait state RMSE that we want
#Another idea for optimal is that we use the best case rmse for every state
# irrelevant of tunning
gait_states= ['phase', 'stride_length', 'ramp']

#Rename to get better plots labels
new_gait_states = ['Phase RMSE', 'Stride Length RMSE', 'Ramp RMSE']
change_dict = {old:new for old,new in zip(gait_states,new_gait_states)}
df.rename(columns=change_dict,inplace=True)

#Rename gait states to new scheme
gait_states = new_gait_states
bin_step_list = [0.005,0.005, 0.1]

histogram = alt.hconcat(*[alt.Chart(df).mark_bar().encode(
                            x=alt.X(f'{state}',bin=alt.Bin(step=bin_step)),
                            y='count()'
                            )
                          for state, bin_step
                          in zip(gait_states, bin_step_list)]).properties(
                              title=title
                          )


histogram.show()