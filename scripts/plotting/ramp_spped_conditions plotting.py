import matplotlib.pyplot as plt
import numpy as np 
import pickle
import altair as alt
import pandas as pd
# conditions =        [ (0.0, 0.8),
#                       (0.0, 1.0),
#                       (0.0, 1.2),
#                       (-2.5,1.2),
#                       (-5,1.2),
#                       (-7.5,1.2),
#                       (-10,0.8),
#                       (-7.5,0.8),
#                       (-5,1.2),
#                       (-2.5,0.8),
#                       (0.0, 0.8),
#                       (2.5,1.2),
#                       (5,1.2),
#                       (7.5,1.2),
#                       (10,0.8),
#                       (7.5,0.8),
#                       (5,1.2),
#                       (2.5,0.8),
#                       (0.0, 1.2),
#                       (-7.5,0.8),
#                       (10,0.8)]

with open('../ekf_sim/random_ramp_speed_condition.pickle','rb') as file:
    conditions = pickle.load(file)

ramp_list, speed_list = list(zip(*conditions))



# Matplotlib implementation
# fig,axs = plt.subplots(2)

# num_conditions = len(conditions)


# x = np.arange(num_conditions)

# axs[0].plot(x, ramp_list,'-o')
# axs[0].set_ylabel('Ground Inclination (degrees)')
# axs[1].plot(x, speed_list, '-o', c='orange')
# axs[1].set_ylabel('Treadmill Speed (m/s)')
# plt.show()


#Altair implementation
data = {'ramp':ramp_list, 'speed':speed_list}
df = pd.DataFrame.from_records(data)
#Create the index as a column
df.reset_index(inplace=True)
print(df)

#Define axis with no gridlines and scale with no zero point
axis = alt.Axis(grid=False) 


ramp_line = alt.Chart(df).mark_line(color='blue').encode(
    x=alt.X('index:O',title=None, axis=axis),
    y=alt.Y('ramp:O',title='Ground Inclination (degrees)',axis=axis,sort='-y')
)
ramp_scatter = ramp_line.mark_circle(size=200,color='blue')
ramp_combined = (ramp_line + ramp_scatter).properties(
    width = 400,
    height = 200
)



speed_line = ramp_line.mark_line(color='orange').encode(
        y=alt.Y('speed:O',title='Treadmill Speed (m/s)',axis=axis,sort='-y'),
)
speed_scatter = speed_line.mark_circle(size=200,color='orange')
speed_combined = (speed_line + speed_scatter).properties(
    width = 400,
    height = 200
)


#Vertical concatenation
total_plot = ramp_combined & speed_combined

total_plot.show()