import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
import numpy as np 



dataset_name = '../../local-storage/test/dataport_flattened_partial_{}.parquet'

subject_names = ['AB01','AB02','AB03']


joint = 'jointangles_foot_x'

x_axis = np.linspace(0,1,150)

ax = plt.subplot(111)


for name in subject_names:
    file_name = dataset_name.format(name)
    
    file_data = pd.read_parquet(file_name, columns = [joint])
    
    file_numpy = file_data[joint].values

    default_cycler = (cycler(color=['r', 'g', 'b', 'y']) + \
                      cycler(alpha=[1.0,0.4,0.4,0.4]))

    plt.rc('lines', linewidth=4)
    plt.rc('axes', prop_cycle=default_cycler)

    plt.plot(x_axis,file_numpy[:150])


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
    
plt.axvspan(0.35,0.65, alpha=0.5, color='yellow')

plt.savefig('slide2.png', transparent=True)


plt.show()