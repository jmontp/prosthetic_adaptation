from operator import sub
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

subject = 'AB02'
joint = 'jointangles_thigh_x'
filename = '../../data/flattened_dataport/dataport_flattened_partial_{}.parquet'

#Conditions that we want to filter by 
ramp = 0
speed = 1.0

#Get data for one subject
data = pd.read_parquet(filename.format(subject))
mask = (data['ramp'] == ramp)
# step_data = data[mask][joint].values.reshape(-1,150)
step_data = data[mask][joint].values.reshape(-1,150).mean(axis=0).reshape(1,150)
# step_data = data[joint].values.reshape(-1,150)



#Get data for all the subjects
# subjects = [f'AB{i:02}' for i in range(1,11)]
# datas = [pd.read_parquet(filename.format(subb)) for subb in subjects]

#All the data, all subjects
# step_data = np.concatenate([d[joint].values.reshape(-1,150) for d in datas], axis=0)

#Filtered by ramp, all subjects
# mask_list = [(d['ramp'] == ramp) & (d['speed'] == speed) for d in datas]
# mask_list = [(d['ramp'] == ramp) for d in datas]
# step_data = np.concatenate([d[mask_i][joint].values.reshape(-1,150) for d,mask_i in zip(datas,mask_list)], axis=0)
# step_data = np.concatenate([d[mask_i][joint].values.reshape(-1,150).mean(axis=0).reshape(1,150) for d,mask_i in zip(datas,mask_list)], axis=0)



phase = np.linspace(0,100,150)

counter = 0

for i in range(step_data.shape[0]):
# for i in range(1):
    curr_data = step_data[i,:]

    #Add low pass filtering
    # fsig = np.fft.fft(curr_data)
    # fsig[20:] = 0
    # curr_data = np.fft.ifft(fsig)


    plt.plot(phase,curr_data)

    if np.max(curr_data) > 78:
        counter += 1 

print(f"bad steps/good steps {counter}/{step_data.shape[0]}")





# mean = step_data.mean(axis=0)
# std = step_data.std(axis=0)
# mean_list = [data[mask][output_name].values.reshape(-1,150).mean(axis=0) for output_name in [joint]]
# std_list = [data[mask][output_name].values.reshape(-1,150).std(axis=0) for output_name in [joint]]
# mean = mean_list[0]
# std = std_list[0]
# plt.plot(phase,mean)
# plt.plot(phase,mean + 2*std)
# plt.plot(phase,mean - 2*std)


# plt.title("Thigh Angle vs Gait Phase")
# plt.title(f"Foot Angle Samples")
plt.xlabel("Gait Phase")
# plt.ylabel("Thigh Torque (N*mm)")
plt.ylabel("Thigh Angles (Degrees)")
plt.show()