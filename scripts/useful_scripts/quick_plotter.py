import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

# Configuration of plot 
################################################################################################


for subject in [f"AB{i:02d}" for i in range(1,11)]:
    filename = '../../data/flattened_dataport/dataport_flattened_partial_{}.parquet'


    data = pd.read_parquet(filename.format(subject))
    

    plt.scatter(data['stride_length'], data['phase_dot'])

plt.xlabel('stride length')
plt.ylabel('phase rate')
plt.ylim((0,2))
plt.show()