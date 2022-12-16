"""
This file is meant to plot the summary statistics (using all the data) 
for the first test in the offline ekf paper
"""


import altair as alt
import pandas as pd 
from decimal import Decimal
import numpy as np 
from first_paper_utils import filter_to_good_range_case, gait_states

#Filter data to the good noise tuning
FILTER_DATA_GOOD_TUNNING=False

#Allow plotting large datasets
alt.data_transformers.disable_max_rows()

#Read the data in 
# data_location = '../ekf_sim/first_paper_NLS_vs_ISA.csv'
data_location = '../ekf_sim/online_ls_test.csv'
df, title = pd.read_csv(data_location), 'All data'



#Decide to filter for good tunning values
# if (FILTER_DATA_GOOD_TUNNING == True):
#    df, title = filter_to_good_range_case(df), '"Good" Range'

#Filter for the tests that we want
# df = df[(df['Test'] == 'PCA_GF') | (df['Test'] == 'ISA')]

#Create the chart
box_plot = alt.hconcat(*[alt.Chart(df).mark_boxplot(outliers=False).encode(
                           #  x = 'Test:N',
                            y = f'{state}:Q',
                           #  column='Subject:N'
                           column='Test:N',
                           row='num conditions:Q'
                        )
                         for state 
                         in gait_states
                         ]
                       )

box_plot.title=title


box_plot.show()