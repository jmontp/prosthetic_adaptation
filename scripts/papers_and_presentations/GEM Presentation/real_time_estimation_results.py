

import numpy as np 
import matplotlib.pyplot as plt 
import math
from cycler import cycler

zero_foot_angle_correction_factor = 75

measured_foot_angle = np.load('actual_foot_angles_no_gf.npy') + zero_foot_angle_correction_factor
measured_foot_angle_learned = np.load('actual_foot_angles_gf.npy')[:,0] + zero_foot_angle_correction_factor

predicted_foot_angles = np.load('predicted_foot_angles_no_gf.npy') + zero_foot_angle_correction_factor
predicted_foot_angles_learned = np.load('predicted_foot_angles_gf.npy')[:,0] + zero_foot_angle_correction_factor


x_axis = np.linspace(0,1,150)

ax = plt.subplot(111)




halfway_point = int(math.floor(measured_foot_angle.shape[0]/150/2)*150)




default_cycler = (cycler(color=['r', 'm', 'b', 'c','m','y']) + \
                      cycler(alpha=[1.0, 0.7, 0.7, 0.7, 0.4, 0.4]))

plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=default_cycler)
    


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.axes.yaxis.set_visible(False)
ax.axes.xaxis.set_visible(True)


t = 150
initial_point = 40*150 

# ax.plot(x_axis[:t], measured_foot_angle[initial_point:initial_point+t,0])
# ax.plot(x_axis[:t], predicted_foot_angles[initial_point:initial_point+t,0])
ax.plot(x_axis[:t], measured_foot_angle_learned[initial_point:initial_point+t])
ax.plot(x_axis[:t], predicted_foot_angles_learned[initial_point:initial_point+t])
legend = ['Measured Foot Angle', 'Predicted Foot Angle', 'learned measured', 'learned predicted']
plt.legend(legend, prop={'size': 14})
plt.savefig('error_plot_after_learn.png', transparent=True)
# plt.savefig('error_plot.png', transparent=True)



# ax.plot(x_axis, measured_foot_angle[:150, 0])
# ax.plot(x_axis, predicted_foot_angles[:150, 0])
# legend = ['Measured Foot Angle', 'Predicted Foot Angle']
# plt.legend(legend, prop={'size': 14})
# plt.savefig('before training.png', transparent=True)

# ax.plot(x_axis, measured_foot_angle[halfway_point:halfway_point+150, 0])
# ax.plot(x_axis, predicted_foot_angles[halfway_point:halfway_point+150, 0])
# legend = ['Measured Foot Angle', 'Predicted Foot Angle']
# plt.legend(legend, prop={'size': 14})
# plt.savefig('after training.png', transparent=True)



plt.show()



