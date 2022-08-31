import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

# Configuration of plot 
################################################################################################
#Save file location for file that have phase condition added to them
save_file_location = ("../../data/flattened_dataport/"
                        "dataport_flattened_partial_{}.parquet")


subject_list = [f"AB{i:02d}" for i in range(1,11)]
#Iterate through all the subjects
subject_data_list = [(subject,
                      pd.read_parquet(save_file_location.format(subject)))
                      for subject 
                      in subject_list]

def plot_stride_length():
    
    fig,axs = plt.subplots(10)

    for (subject, data), plot_axs in zip(subject_data_list, axs):
        
        plot_axs.plot(data['stride_length'])
        plot_axs.set_ylabel(f"{subject} Stride Length Fixed")

    axs[-1].set_xlabel("Sample")

    plt.show()


def plot_thigh_foot_shank():
    for subject,data in subject_data_list:
    
        fig, axs = plt.subplots(3)

        joints = ['foot','knee','thigh']

        joint_data_list = [data[f'jointangles_{joint}_x'].values.reshape(-1,150)
                    for joint
                    in joints]
        
        for i,data_2 in enumerate(joint_data_list):
            print(f"{joints[i]} has {np.count_nonzero(np.isnan(data_2))} nans")

        for joint_index, joint_data in enumerate(joint_data_list):
            for i in range(joint_data.shape[0]):
                axs[joint_index].plot(joint_data[i,:])
                axs[joint_index].set_ylabel(joints[joint_index])

    plt.show()
    
def plot_foot_angles():

    #Create three subplots, one for ramp = 10,0,-10
    subject,data = subject_data_list[0]

    ramp_10 = data[data['ramp'] == 10]['jointangles_foot_x'].values.reshape(-1,150)
    ramp_0 = data[data['ramp'] == 0]['jointangles_foot_x'].values.reshape(-1,150)
    ramp_m10 = data[data['ramp'] == -10]['jointangles_foot_x'].values.reshape(-1,150)

    phase = np.linspace(0,1,150)
    #Setup three subplots
    fig, axs = plt.subplots(3)

    for i in range(ramp_10.shape[0]):
        axs[0].plot(phase,ramp_10[i,:])
        axs[0].set_ylabel("Ramp 10")

    for i in range(ramp_0.shape[0]):

        axs[1].plot(phase,ramp_0[i,:])
        axs[1].set_ylabel("Ramp 0")

    for i in range(ramp_m10.shape[0]):

        axs[2].plot(phase,ramp_m10[i,:])
        axs[2].set_ylabel("Ramp -10")

    plt.show()

def plot_hip_knee_with_phase_cond():

    #Define the colors for each phase
    colors = ['red','blue','green']

    #Save file location for file that have phase condition added to them
    save_file_location = ("../../data/flattened_dataport/"
                          "dataport_flattened_partial_{}_phase_cond.parquet")

    # #Create subplots
    # fig,axs = plt.subplots(2)
    
    # #Configure labels
    # axs[0].set_ylabel("hip angles")
    # axs[1].set_ylabel("knee angles")
    # axs[1].set_xlabel('phase')

    #Iterate through all the subjects
    for subject in [f"AB{i:02d}" for i in range(1,2)]:
        #Get the file for the specific subject
        subject_file = save_file_location.format(subject)

        #load the parquet datafile
        subject_data = pd.read_parquet(subject_file)

        #Get the knee angle, hip angle, and phase condition
        hip_angles = subject_data['jointangles_hip_x'].values.reshape(-1,150)
        knee_angles = subject_data['jointangles_knee_x'].values.reshape(-1,150)
        phase_condition = subject_data['phase_condition'].values

        #Plot steps
        # for step in range(hip_angles.shape[0]):
        #     print(f"{step}/{hip_angles.shape[0]}")
        #     for datapoint_in_step in range(hip_angles.shape[1]):

        #         curr_phase = int(phase_condition[step,datapoint_in_step])

        #         axs[0].scatter(datapoint_in_step,
        #                         hip_angles[step,datapoint_in_step],
        #                         c = colors[curr_phase]
        #                     )
        #         axs[1].scatter(datapoint_in_step,
        #                         knee_angles[step,datapoint_in_step],
        #                         c = colors[curr_phase]
        #                     )
        plt.plot(phase_condition)  
        #Run through one person
        plt.show()


def plot_average_hip_knee():
    phase = np.linspace(0,1,150)

    for subject in [f"AB{i:02d}" for i in range(1,2)]:
        filename = '../../data/flattened_dataport/dataport_flattened_partial_{}.parquet'


        data = pd.read_parquet(filename.format(subject))
        
        
        knee_angles = data['jointangles_knee_x'].values.reshape(-1,150)
        avg_knee_angles = np.mean(knee_angles,axis=0)
        
        hip_angles = data['jointangles_hip_x'].values.reshape(-1,150)
        avg_hip_angles = np.mean(hip_angles,axis=0)

        #Plot knee and hip angles
        plt.plot(phase,avg_knee_angles)
        plt.plot(phase,avg_hip_angles)


    # plt.xlabel('stride length')
    # plt.ylabel('phase rate')
    plt.legend(["knee", "hip"])
    plt.show()




if __name__ == "__main__":
    # plot_hip_knee_with_phase_cond()
    # plot_stride_length()
    # plot_foot_angles()
    plot_thigh_foot_shank()