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

#T-test calculation
from scipy.stats import ttest_rel



#Filter data to optimal case
# df, title = filter_to_good_range_case(df), '"Good" Range'
test_type = 'ISA'
df, title = filter_to_optimal_case(df,test_type), f'Optimal Case - Filtered by {test_type}'

#Gait state RMSE that we want
#Another idea for optimal is that we use the best case rmse for every state
# irrelevant of tunning
gait_states= ['phase', 'stride_length', 'ramp']

#Calculate the test scores
NSL= (df['Test'] == 'NSL')
ISA= (df['Test'] == 'ISA')




#Rename to get better plots labels
new_gait_states = ['Phase RMSE', 'Stride Length RMSE', 'Ramp RMSE']
change_dict = {old:new for old,new in zip(gait_states,new_gait_states)}
df.rename(columns=change_dict,inplace=True)

#Rename gait states to new scheme
gait_states = new_gait_states

#Calculate p-values
p_value_dict = {state:ttest_rel(df[NSL][state],df[ISA][state]).pvalue
                for state
                in gait_states}


#Convert to long format
df = pd.melt(df, 
             id_vars=['Test'],
             value_vars=gait_states, 
             var_name='states', 
             value_name='rmse')

#Add the pvalue to the state column
df['p_value'] = 0
for state in gait_states:
    df['p_value'].mask(df['states'] == state, 
                       other=p_value_dict[state],
                       inplace=True)

#Initialize new column
df['percent_diff'] = 0

#Recreate the test filters since we made the dataset into long format
NSL= (df['Test'] == 'NSL')
ISA= (df['Test'] == 'ISA') 

#Calculate percentage difference
for state in gait_states:
    
    #Create another boolean array filter for the state
    state_fil = df['states']==state
    
    
    #Get the mean for the ISA and NSL cases
    NSL_mean = df['rmse'].loc[NSL & state_fil].mean()
    ISA_mean = df['rmse'].loc[ISA & state_fil].mean()
    
    #Calculate the percentage difference
    percent_diff = (NSL_mean - ISA_mean)/ISA_mean
        
    #Set the percentage difference to be a value for the NSL case 
    df['percent_diff'].mask(NSL & state_fil, other=percent_diff, inplace=True)
        
        
#Create the bar charts
BAR_PLOT=True
if BAR_PLOT:
    
    bar_charts = alt.Chart(df).mark_bar().encode(
        x=alt.X('Test:N',title=None),
        y=alt.Y('average(rmse):Q',title=None),
        tooltip=['average(rmse):Q','p_value:Q']
    ).properties(
        width=150
    )

    #Create the text anotation
    text_annotation = bar_charts.mark_text(
        align='center',
        dy=-5  # Nudges text to right so it doesn't appear on top of the bar  
    ).encode(
        text=alt.Text('average(percent_diff):Q', format='.2%')
    )
    
    #Create the text anotation
    text_annotation2 = bar_charts.mark_text(
        align='center',
        dy=-20  # Nudges text to right so it doesn't appear on top of the bar  
    ).encode(
        text=alt.Text('p_value:Q', format='.2')
    )


else: #Make box plot
    bar_charts = alt.Chart(df).mark_boxplot(outliers=False).encode(
        x=alt.X('Test:N',title=None),
        # y=alt.Y('average(rmse):Q',title=None),
        # tooltip='average(rmse):Q',
        y=alt.Y('rmse:Q',title=None),
        tooltip='rmse:Q',
    ).properties(
        width=150
    )

    #Create the text anotation
    text_annotation = alt.Chart(df).mark_text(
        align='left',
        dx=10 #Nudges for the box plot  
    ).encode(
        x=alt.X('Test:N',title=None),
        y=alt.Y('median(rmse)'),
        text=alt.Text('median(percent_diff):Q', format='.2%')
    )

# #Make svg for better scaling
# bar_charts.display(renderer='svg')

#Show in browser
combined = (bar_charts + text_annotation + text_annotation2).facet(
    column=alt.Column('states:N',title=None)
).resolve_scale(
    y='independent'
).properties(
    title=title
)


combined.show()