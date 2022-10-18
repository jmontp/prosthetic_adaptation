import matplotlib.pyplot as plt
import numpy as np 

conditions =        [ (0.0, 0.8),
                      (0.0, 1.0),
                      (0.0, 1.2),
                      (-2.5,1.2),
                      (-5,1.2),
                      (-7.5,1.2),
                      (-10,0.8),
                      (-7.5,0.8),
                      (-5,1.2),
                      (-2.5,0.8),
                      (0.0, 0.8),
                      (2.5,1.2),
                      (5,1.2),
                      (7.5,1.2),
                      (10,0.8),
                      (7.5,0.8),
                      (5,1.2),
                      (2.5,0.8),
                      (0.0, 1.2),
                      (-7.5,0.8),
                      (10,0.8)]


ramp_list, speed_list = list(zip(*conditions))



fig,axs = plt.subplots(2)

num_conditions = len(conditions)


x = np.arange(num_conditions)

axs[0].scatter(x, ramp_list)
axs[1].scatter(x, speed_list)
plt.show()