"""
This file is meant to plot the pairwise box plot of the good region 
for the first test in the offline ekf paper
"""

import altair as alt
import pandas as pd 
from first_paper_utils import filter_to_good_range_case, filter_to_optimal_case
from first_paper_utils import gait_state_noise_list

#T-test calculation
from scipy.stats import ttest_rel

#Allow plotting large datasets
alt.data_transformers.disable_max_rows()

#Read the data in 
data_location = '../ekf_sim/first_paper_NLS_vs_ISA.csv'

df, title = pd.read_csv(data_location), 'All Data'


#Configuration for the first part of the paper NLS vs ISA
baseline_test = 'ISA'
comparison_test = 'NSL'

#Configuration for the second part of the paper PCA_GF vs ISA
# baseline_test = 'ISA'
# comparison_test = 'PCA_GF'

#Filter for the test that we will compare on
df = df[(df['Test']==baseline_test) | (df['Test']==comparison_test)]

#Filter data to optimal case
# df = filter_to_good_range_case(df) #Comment line out for all the "good" range
test_type = baseline_test
df_optimal = filter_to_optimal_case(df,test_type)


#Define 
title = '"Good Range" Pairwise difference in performance'

#Gait state RMSE that we want
#Another idea for optimal is that we use the best case rmse for every state
# irrelevant of tunning
gait_states= ['phase', 'stride_length', 'ramp']


#Create names for the dataframe pairwise difference
gait_state_diff = [f'{state}_diff' for state in gait_states]
gait_state_diff_norm = [f'{state}_diff_norm' for state in gait_states]
#Rename to get better plots labels once data processing is done
# (having spaces might make certain operations more difficult)
new_gait_states = ['Phase RMSE Diff',
                   'Stride Length RMSE Diff', 
                   'Ramp RMSE Diff']
#Create a dictionary that maps old names to new names
change_dict = {old:new for old,new in zip(gait_state_diff,new_gait_states)}

# We want to create two plots - one with the units in stride length
# and one with the units in percent difference. Therefore, calculate the 
# median value that we will use to scale the axis by grouping across all subjects
gait_state_ISA_median = {state:df[df['Test'] == baseline_test].groupby(gait_state_noise_list).mean()[state].median()
                         for state
                         in gait_states}

#Create a list to store the modified dataframes
df_output_list = []

#Create list to store the optimal t-tests and optimal 
t_test_results_list = []

for df_temp in [df,df_optimal]:
    
    #Create temp dataframe
    df2=pd.DataFrame()

    #Get filters for the tests that are ran
    ISA = df_temp[df_temp['Test'] == baseline_test].reset_index(drop=True)
    NSL = df_temp[df_temp['Test'] == comparison_test].reset_index(drop=True)

    #Aggregate t-test results per state
    df_t_test_result_list = []
    
    #Create the pairwise differences for each state
    for state,df_name,df_name_norm in zip(gait_states,gait_state_diff,
                                          gait_state_diff_norm):
        
        #Get the values for this state in each test case
        ISA_state = ISA[state]
        NSL_state = NSL[state]
        
        #Calculate the pairwise difference 
        df2[df_name] = ISA_state - NSL_state
        #Calculate the pairwise difference based on scaled by the difference
        df2[df_name_norm] = (ISA_state - NSL_state) / gait_state_ISA_median[state]
        
        #Calculate and store p-value
        t_test_result = ttest_rel(ISA_state, NSL_state)
        df_t_test_result_list.append(t_test_result)
        
        

    #Add the subject indicator
    df2['Subject'] = NSL['Subject']
    df2[gait_state_noise_list] = NSL[gait_state_noise_list]

    #Average out the subjects
    df2 = df2.groupby(gait_state_noise_list, as_index=False).mean()

    #Rename diff labels
    df2.rename(columns=change_dict,inplace=True)
    
    #Rename to new dataframe    
    df_output_list.append(df2)
    
    #Store the t-test result list
    t_test_results_list.append(df_t_test_result_list)

#Assign the processed datasets and calculate the number of comparisons   
df, df_optimal = df_output_list
num_datapoints = len(df)

#Update the title to have the number of datapoints
title = title + f" | n = {num_datapoints}"

#Rename gait states to new scheme
gait_state_diff = new_gait_states

#Create the base plot
base = alt.Chart(df,width=100).mark_boxplot(outliers=False).encode(
    # x=alt.X('index:O',title=None,axis=None)
)

#Group the optimal points
optimal_point = alt.Chart(df_optimal).mark_tick(
    color='red', height=5, width=30
).encode(
    # x=alt.X('index:O',title=None,axis=None)
)

def create_subplot(diff_label, diff_label_norm,state_index):
    '''This function will create each subplot'''
    
    #Get the p-value for the 'good results' dataframe
    p_value = t_test_results_list[0][state_index].pvalue
    
    # Debug - I used this to verify that the scaling was consistent in the 
    # dataset. Turns out, scale isn't absolute and will add a 'nice' white
    # space buffer. Can disable by passing in nice = False
    median_scale_fix = gait_state_ISA_median[gait_states[state_index]] 
    
    
    min_scale_norm = -0.30 #-5 percent
    # min_scale_norm = -0.05 #-5 percent
    max_scale_norm = 0.45 #55 percent
    
    
    #Create the scale so that they match
    # min_scale = float(df[diff_label].min())
    # #Make sure 0 is included
    # min_scale = min_scale if min_scale < 0 else 0 
    # max_scale = float(df[diff_label].max())
    min_scale = min_scale_norm * median_scale_fix
    max_scale = max_scale_norm * median_scale_fix
    
    space_buffer = (max_scale - min_scale) / 2
    space_buffer = 0 
    
    scale=alt.Scale(domain=[min_scale - space_buffer,
                            max_scale + space_buffer],nice=False)
    
    
    #Make box plot for the pairwise difference
    box_plot = base.encode(
        y=alt.Y(f'{diff_label}:Q',scale=scale),
        tooltip=f'{diff_label}:Q',
    )
    
    #Create variables to offset text so it does not sit on top of the mark
    dx = -10
    dy = 10
    
    #Create the text anotation
    # text_annotation = alt.Chart(df).mark_text(
    text_annotation = alt.Chart(df_optimal).mark_text(
        align='right',
        dx=dx, #Nudges for the box plot  
        dy=dy  #Only do this when applying the df_optimal as the data
    ).encode(
        y=alt.Y(f'median({diff_label}):Q',title=None),
        text=alt.Text(f'median({diff_label}):Q', format='.0e')
    )

    #State Pairwise Normalized difference scale 
    # min_scale_norm = float(df[diff_label_norm].min())
    # #Make sure 0 is included
    # min_scale_norm = min_scale_norm if min_scale_norm < 0 else 0 
    # max_scale_norm = float(df[diff_label_norm].max()) 
    # space_buffer = (max_scale_norm - min_scale_norm) /  2
    
    #Create the scale for the normalized values
    scale_norm=alt.Scale(domain=[min_scale_norm - space_buffer
                                 ,max_scale_norm + space_buffer],nice=False)
    
    #Make box plot for the pairwise normalized difference
    box_plot_norm = base.encode(
        y=alt.Y(f'{diff_label_norm}:Q',scale=scale_norm,
                axis=alt.Axis(format='.0%')),
        tooltip=f'{diff_label_norm}:Q',
    )
    
    #Create the text anotation
    # text_annotation_norm = alt.Chart(df).mark_text(
    text_annotation_norm = alt.Chart(df_optimal).mark_text(

        align='left',
        dx=-dx, #Nudges for the box plot  
        dy=dy  #Only do this when applying the df_optimal as the data
    ).encode(
        y=alt.Y(f'median({diff_label_norm}):Q',title=None),
        text=alt.Text(f'median({diff_label_norm}):Q', format='.1%')
    )
    
    #calculate the point where the optimal plot is 
    optimal_plot = optimal_point.encode(
        y=alt.Y(f'{diff_label}:Q',scale=scale,axis=None),
        tooltip=f'{diff_label}:Q'
    )

    #Add the plots
    subplot = alt.layer(
        box_plot + text_annotation,
        box_plot_norm + text_annotation_norm,
        optimal_plot
    ).encode(
        x=alt.X('index:N',title=f'p={p_value:0.2}',axis=alt.Axis(ticks=False,labels=False))
    ).resolve_scale(
        y='independent'
    ).properties(
        title=f'{diff_label}',
    )
    
    return subplot



# #Make svg for better scaling
# bar_charts.display(renderer='svg')


phase_rmse_plot = create_subplot(gait_state_diff[0],gait_state_diff_norm[0],0)
strid_rmse_plot = create_subplot(gait_state_diff[1],gait_state_diff_norm[1],1)
ramp_rmse_plot = create_subplot(gait_state_diff[2],gait_state_diff_norm[2],2)


final_plot = alt.hconcat(phase_rmse_plot,
                         strid_rmse_plot,
                         ramp_rmse_plot).resolve_scale(
    y='shared'
).properties(
    title=title
)


final_plot.show()