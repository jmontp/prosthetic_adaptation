#Common imports for plotting and serializatoin
from collections import OrderedDict
import altair as alt
import pickle
import pandas as pd

#Import the file with all the conditions
with open('../ekf_sim/all_test_conditions.pickle','rb') as file:
    all_task_conditions = pickle.load(file)

#Load the dataframe
df = pd.read_csv('../ekf_sim/online_ls_test.csv')

#Initialize an empy dataframe
df_output = pd.DataFrame()

#Define the columns we care about
df_name_update=OrderedDict()
df_name_update['phase'] = 'phase_rmse_delta'
df_name_update['phase_dot']='phase_dot_rmse_delta'
df_name_update['stride_length']='stride_length_rmse_delta'
df_name_update['ramp']='ramp_rmse_delta'

#Update the names
df = df.rename(columns=df_name_update)

#Get lists with columns that we think are important
gait_states = list(df_name_update.values())
task_tests =['conditions','Test','num_train_steps']

#Iterate through all the tasks
for task in all_task_conditions:
    
    #Create the filter
    task_filter = (df['conditions'] == repr(task)) \
                  & (df['num_train_steps'] < 100) \
                #   & (df['num conditions'] == 27)
    
    #Get data for a task
    task_df  = df[task_filter][gait_states+task_tests]
    
    #Get the results of the tests (Ordered by subject)
    ISA = task_df[task_df['Test'] == 'ISA'][gait_states].reset_index(drop=True)
    NLS = task_df[task_df['Test'] == 'NLS'][gait_states].reset_index(drop=True)
    PCA = task_df[task_df['Test'] == 'PCA_GF'][gait_states].reset_index(drop=True)
        
    #Create temporary dataframes to store conditions
    df_temp1 = NLS - ISA
    df_temp1['test_delta'] = "NLS_m_ISA"
    
    df_temp2 = PCA - ISA
    df_temp2['test_delta'] = "PCA_m_ISA"
    
    df_temp3 = PCA - NLS
    df_temp3['test_delta'] = "PCA_m_NLS"

    for df_temp in [df_temp1,df_temp2,df_temp3]:
        df_temp['num_conditions'] = len(task)
        df_temp['condition'] = repr(task)
    
    df_temp['num_train_steps'] = task_df['num_train_steps'].reset_index(drop=True)

    #Concatenate to the output dataframe
    df_output = pd.concat([df_output,df_temp1,df_temp2,df_temp3],
                          ignore_index=True)
    
    
#Should have the same length
# assert len(df_output) == len(df), f"{len(df_output)}= != {len(df)=}"

line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='orange').encode(y='y')


#Create the chart
box_plot = alt.hconcat(
    *[alt.Chart(df_output).mark_boxplot(
        outliers=False
        ).encode( 
            y = f'{state}:Q',
            column=alt.Column('test_delta:N',
                title="",
                header=alt.Header(labelAngle=90,labelAlign='left')
            ),
            x ='num_conditions:O'
        )

        for state 
        in gait_states
        ]
)

box_plot = box_plot

box_plot.display(renderer='svg')

box_plot.show()