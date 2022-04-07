import matplotlib.pyplot as plt
import pandas as pd

trial = 's1x2i7x5'
subject = 'AB07'
joint = 'jointangles_thigh_x'
filename = '../../data/flattened_dataport/dataport_flattened_partial_{}.parquet'.format(subject)


data = pd.read_parquet(filename)

step_data = data[joint][:450]
trial = data['trial'][0]


plt.plot(step_data)
plt.title(f"{trial} - {joint}")
plt.show()