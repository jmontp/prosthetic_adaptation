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

#Greatest difference
df_optimal_diff = df.iloc[[214]]  #Opitmal index found manually
df_optimal = df_optimal_diff

# Filter for each method
baseline_results_df = df_optimal[(df_optimal['Test']==baseline_test)]
my_method_results_df = df_optimal[(df_optimal['Test']==comparison_test)]

states = ['phase','phase_dot','stride_length','ramp']

print(f"Mean results for Subject-Indepent \n\r{baseline_results_df[states].mean()}")
print(f"Mean results for Personalized \n\r{my_method_results_df[states].mean()}")


# print(f"std dev results for Subject-Indepent \n\r{baseline_results_df[states].std()}")
# print(f"std dev results for Personalized \n\r{my_method_results_df[states].std()}")

