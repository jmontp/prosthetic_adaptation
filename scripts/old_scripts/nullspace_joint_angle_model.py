"""
This is meant to implement the nullspace model in

A novel appreoach for representing and generalising periodic gaits
doi:10.1017/S026357471400188X
"""
#Function type hints
from typing import Callable

# Common imports
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy

class NullspaceModel:
    """
    This class calculates the instantaneous joint angle velocity at the 
    given joint angles and time
    """
    
    # Define constants for stance, pre-swing, and late-swing phases
    STANCE_PHASE = 0
    PRE_SWING_PHASE = 1
    LATE_SWING_PHASE = 2
    # Leg length parameters
    thigh_length = 0.45       # In units ofmeters
    calf_length = 0.45        # In units of meter
    # Fix stride length for now
    # stride_length = 1.2      # In units of meters

    #Get initial leg height
    initial_hip_angle = -40
    initial_knee_angle = 0
    initial_leg_height = (np.sin(np.deg2rad(initial_hip_angle))*thigh_length
                          + np.cos(np.deg2rad(initial_knee_angle)*calf_length))
    #Get initial stride length
    stride_length = (np.cos(np.deg2rad(initial_hip_angle)) * thigh_length
                     + np.sin(np.deg2rad(initial_knee_angle)) * calf_length)

    def __init__(self,
                 task_space_policy:Callable[[int,np.ndarray],np.ndarray],
                 constraint_matrix:np.ndarray,
                 null_space_policy:Callable[[np.ndarray],np.ndarray],
                 phase_transition_func:Callable[[int,np.ndarray],int],
                 initial_phase_condition:int = STANCE_PHASE,
                 ):

        """
        Keywork Arguments:
                    
        task_space_policy -- matrix that determines the task-space velocity
                (joint angles in this case). This is indexed by phase as the
                first entry.
            Data type: np array, shape(3,2,1)
        
        constraint_matrix -- constraint matrix that is used to define the
                constraints and unconstrained space. This matrix is indexed by
                phase as the first entry.
            Data type: np array, shape(3,1,2)

        null_space_policy -- this is the nullspace policy that determines the
                unconstrained behaviour. This function's first argument is the
                phase, the second argument is the joint angles.
            Data type: function, returns np array of shape(2,1)

        phase_transition_matrix -- function that determines the current phase
                condition. The first argument is the current state, and the
                second argument is the current joint angles, third argument is
                stride length.
            Data type: function object
        
        initial_phase_condition -- integer that describres the current phase.
                The definitions for each phase are staticly defined in the
                class. There are 3 phases: Stance, Pre-Swing, Late-Swing.
            Data type: integer
        """

        # Assign local variables to all the input variables
        self.task_space_policy = task_space_policy
        self.constraint_matrix = constraint_matrix
        self.null_space_policy = null_space_policy
        self.phase_transition_func = phase_transition_func
        self.current_phase = initial_phase_condition


    def evaluate(self,t,curr_joint_angles):
        """
        Calculates the current joint angle velocity by calculating
        
        Keyword Arguments
        t -- time in seconds,
            Data type: float

        joint_angles -- hip and knee angles in degrees
            Data type: np array, shape:(2,1)

        Return
        u -- joint velocity at current joint angles
        """
        
        # Wrap angles
        for i in range(2):
            while(curr_joint_angles[i] > 180):
                curr_joint_angles[i] -= 360
            while(curr_joint_angles[i] < -180):
                curr_joint_angles[i] += 360


        # Get the current phase condition
        self.current_phase = self.phase_transition_func(self.current_phase,
                                                        curr_joint_angles)
        
        # Get the task space constraint
        A = self.constraint_matrix[self.current_phase,:,:]

        # Get the task space velocity
        b = self.task_space_policy(self.current_phase, curr_joint_angles)

        # Get the nullspace projection policy
        A_pinv = np.linalg.pinv(A)
        I = np.eye(A.shape[0])
        N = (I - A_pinv @ A)

        # Get the nullspace policy
        pi = self.null_space_policy(curr_joint_angles)

        # Calculate joint velocity
        u = A_pinv @ b + N @ pi

        return u.reshape(-1)


###############################################################################
###############################################################################
## Create functionality required to fit U_null

def add_phase_condition_to_dataset():
    """
    Add the phase condition to the flattened datasets
    """
    #Define the phase name in the dataset
    PHASE_COND = 'phase_condition'
    
    #Define the location of the datasets
    original_file_location = ("../../data/flattened_dataport/"
                              "dataport_flattened_partial_{}.parquet")
    
    #Where to store the modified files
    save_file_location = ("../../data/flattened_dataport/"
                          "dataport_flattened_partial_{}_phase_cond.parquet")

    #Define the list of subjects
    subjects = [f"AB{i:02}" for i in range(1,11)]
    
    #Add phase to each subjects data file
    for subject in subjects:

        #Get the file for the corresponding subject
        filename = original_file_location.format(subject)
        save_filename = save_file_location.format(subject)
        # print(f"Looking for {filename}")

        #Read in the parquet dataframe
        subject_data = pd.read_parquet(filename)

        #No need to add the phase condition if it already has it
        if PHASE_COND in subject_data:
            continue
        
        #Get the numpy dataset for each step
        hip_angles = subject_data['jointangles_hip_x'].values.reshape(-1,150)
        knee_angles = subject_data['jointangles_knee_x'].values.reshape(-1,150)
        stride_length = subject_data['stride_length'].values.reshape(-1,150)

        #The paper outlines that a good approximation for the end stance
        # angles is -40 for hip and 0 for knee. Since there is person-to-person
        # variance in the dataset, use these to offset the angles. The maximum
        # hip and knee flexion should not be modified since this is just an
        # offset
        HIP_EXTENSION_AT_HEEL_TOUCHDOWN = -40
        KNEE_EXTENSION_AT_HEEL_TOUCHDOWN = 0

        #Create the offset by subtracting the original so when it is added we 
        # get the knee angle offset at the end e.g. x + (y - x) = y where
        # the offset is (y-x)
        hip_angles_offset = (HIP_EXTENSION_AT_HEEL_TOUCHDOWN
                              - hip_angles[:,[-1]])
        knee_angles_offset = (KNEE_EXTENSION_AT_HEEL_TOUCHDOWN
                              - knee_angles[:,[-1]])

        #Add the offset to the knee and hip angles
        hip_angles += hip_angles_offset
        knee_angles += knee_angles_offset

        #Invert sign of knee angles
        knee_angles *= -1

        #Create the matrix to store the phase for each point in time
        phase_condition_array =  np.zeros(hip_angles.shape)

        #Iterate through all the steps to get the
        for step_index in range(hip_angles.shape[0]):
            print(f"{subject} step {step_index}")

            #Iterate through all the datapoints in the step
            for datapoint_in_step_index in range(1,150):
                #Get the angles for the current datapoint
                hip_datapoint = hip_angles[step_index,
                                           datapoint_in_step_index]
                knee_datapoint = knee_angles[step_index,
                                             datapoint_in_step_index]
                #Get the current stride length
                stride_length_datapoint = stride_length[step_index,
                                            datapoint_in_step_index]
                #Get the previous phase
                phase_prev = phase_condition_array[step_index,
                                                   datapoint_in_step_index-1]
                #Get the phase
                phase_new = phase_transition_func_paper(phase_prev,
                                [hip_datapoint,knee_datapoint],
                                #stride_length = stride_length_datapoint
                                )

                #Set the phase index
                phase_condition_array[step_index,datapoint_in_step_index] = \
                    phase_new
        
        #Set the subject data in the array
        subject_data[PHASE_COND] = phase_condition_array.ravel()

        #Store the data
        subject_data.to_parquet(save_filename)

        #Print progress message
        print(f"Done with {subject}")


###############################################################################
###############################################################################
## Instantiate the class with the values from the paper

A_paper = np.array([[-0.6, 0.8],   # Stance
              [-0.4, 0.9],   # Early Swing
              [0.1, -0.99]    # Late Swing
             ]).reshape(3,1,2)


def task_space_policy_paper(phase_condition:int,
                            joint_angles:np.ndarray)->np.ndarray:
    """
    This function calculates the task space joint angle velocity based on the 
    current phase and joint angles

    Keywork Arguments
    phase_condition -- integer that describres the current phase.
            The definitions for each phase are staticly defined in the
            class. There are 3 phases: Stance, Pre-Swing, Late-Swing.
        Data type: integer

    joint_angles -- hip and knee angles in degrees
        Data type: np array, shape:(2,1)

    Returns
    task_space_speed -- desired task joint angular velocity
        Data type: np array, shape(2,1)
    """
    # Joint angle target in degrees
    targets = np.array([[-130, -10],  # Stance
                        [-60, -120],  # Early Swing
                        [-40, 0]      # Late Swing
                       ]).reshape(3,2,1)
    
    # Get the specific target for this current phase
    phase_target = targets[phase_condition,:,:].reshape(-1)

    # Take the difference of the current joint angles wit the goal joint angles
    joint_angle_error = np.abs(phase_target - joint_angles)

    #Sum each of the joint angle errors 
    # I guess? they don't say what the task
    # space model is :(
    error = np.sum(joint_angle_error).reshape(1,1)

    #Define the task space policy as constant joint velocity
    task_space_policy = 1

    return np.array(task_space_policy).reshape(1,-1)


def low_pass_filter(adata: np.ndarray,
                     bandlimit: int = 20,
                     sampling_rate: int = 150) -> np.ndarray:
        """
        Low pass filter implementation by Iwohlhart
        https://stackoverflow.com/
        questions/70825086/python-lowpass-filter-with-only-numpy

        Keyword Arguments
        adata -- the data that will be filtered
            Data type: np.ndarray with no specific range
        bandlimit -- the bandwidth limit in Hz
            Data type: int
        sampling_rate -- the sampling rate of the data
            Data type: int

        Returns
        filtered_data
            Data type: np.ndarray
        """
        
        # translate bandlimit from Hz to dataindex according to
        # sampling rate and data size
        bandlimit_index = int(bandlimit * adata.size / sampling_rate)
    
        fsig = np.fft.fft(adata)
        
        for fourier_i in range(bandlimit_index + 1,
                               len(fsig) - bandlimit_index ):
            fsig[fourier_i] = 0
            
        adata_filtered = np.fft.ifft(fsig)
    
        return np.real(adata_filtered)



def phase_transition_func_paper(current_phase:int,
                                joint_angles:np.ndarray,
                                stride_length:float = None)->int:
    """
    This function determies when to transition to the next state given the
    current state, the joint angles, stride length, and leg parameters

    Keyword Arguments:
    current_phase -- integer that describres the current phase.
            The definitions for each phase are staticly defined in the
            class. There are 3 phases: Stance, Pre-Swing, Late-Swing.
        Data type: integer

    joint_angles -- hip and knee angles in degrees
        Data type: np array, shape:(2,1)

    stride_length -- in case the stride length is known a priori

    Returns:
    next_phase_condition -- integer representing the next phase condition
        Data type: integer
    """
    #Print debug text or not
    DEBUG = False

    # Get the hip and knee angles
    hip_angle = joint_angles[0]
    knee_angle = joint_angles[1]

    #Keep an internal state of all the knee angles that we see in order to know
    # when to transition into the late stance phase
    if hasattr(phase_transition_func_paper,'knee_history') is False:
        phase_transition_func_paper.knee_history = []

    # Assume that we stay in the same condition
    next_condition = current_phase

    # Test condition for ending stance phase
    if (current_phase == NullspaceModel.STANCE_PHASE):
        
        # Get the x position of the leg
        r1 = (np.cos(np.deg2rad(hip_angle)) * NullspaceModel.thigh_length
              +  np.sin(np.deg2rad(knee_angle)) * NullspaceModel.calf_length)
        
        #If the stride length is not fed is, use the default one in the 
        # Nullspace model
        if(stride_length is None):
            stride_length = NullspaceModel.stride_length
        #Get the leg displacement goal
        leg_x_position_goal = -0.8*stride_length/2
        
        #Print debug text
        if(DEBUG is True):
            print((f"STANCE -- r1 {r1:.3f}, " 
                f"goal {leg_x_position_goal:.3f},"
                f"hip {hip_angle:.3f} knee {knee_angle:.3f}"))

        # If the leg crosses this distance from the torso, transition into 
        # pre-swing
        if(r1 < leg_x_position_goal):
            next_condition = NullspaceModel.PRE_SWING_PHASE
    
    #Test condition for ending pre swing phase
    if(current_phase == NullspaceModel.PRE_SWING_PHASE):
        #Print debug text
        if(DEBUG is True):
            print("PRE-SWING")

        # The knee angle should reach its maximum flexion at the end of 
        # this phase. To determine this, we can look at if the sign of the 
        # derivative has changes from negative (flexion) to positive 
        # (extension). Since the data is noisy, we'll use an fft to remove the
        # harmonics
        MIN_DATAPOINTS = 5

        if (len(phase_transition_func_paper.knee_history) > MIN_DATAPOINTS):
            
            #Low pass filter previous angle
            prev_knee_angle = low_pass_filter(phase_transition_func_paper\
                                              .knee_history)
            #Remove old angle
            phase_transition_func_paper.knee_history.pop(0)
            #Add new angle
            phase_transition_func_paper.knee_history.append(knee_angle)
            #Low pass filter new angle
            new_knee_angle = low_pass_filter(phase_transition_func_paper\
                                              .knee_history)

            #Using the difference we can get the sign of the derivative without
            # needing to know the dt. If the sign of the difference is
            # positive, the knee is extending. Therefore, we can transition
            # to late swing
            if(new_knee_angle - prev_knee_angle > 0):
                next_condition = NullspaceModel.LATE_SWING_PHASE

    
    # Test condition for ending late swing phase
    if(current_phase == NullspaceModel.LATE_SWING_PHASE):

        #Get the y position of the leg
        r2 = (np.sin(np.deg2rad(hip_angle))*NullspaceModel.thigh_length 
              - np.cos(np.deg2rad(knee_angle))*NullspaceModel.calf_length
              + NullspaceModel.initial_leg_height
              )
        
        #Print debug text
        if(DEBUG is True):    
            print(f"LATE-SWING leg height {r2:02f}")

        #If the leg is very close to the ground, set to zero.
        if (r2 < 0):
            next_condition = NullspaceModel.STANCE_PHASE

            #Clear history at the end of every step
            phase_transition_func_paper.knee_history = []

    return next_condition


def nullspace_policy_paper(joint_angles:np.ndarray)->np.ndarray:
    """
    This function calculates the nullspace policy given a set of joint angles

    Keyword Arguments:

    curr_joint_angles -- hip and knee angles in degrees
        Data type: np array, shape:(2,1)
    """
    #Not implemented, need to do regression

    return np.zeros((2,1))


#Modify the stored data so that it can have ground truth phase condition 
add_phase_condition_to_dataset()


# # Initialize the model
# paper_model = NullspaceModel(task_space_policy_paper,
#                              A_paper,
#                              nullspace_policy_paper,
#                              phase_transition_func_paper,
#                              NullspaceModel.STANCE_PHASE)

# # Get the initial joint angles from the nullspace model
# initial_joint_angles = np.array([NullspaceModel.initial_hip_angle,
#                                    NullspaceModel.initial_knee_angle])\
#                                 .reshape(-1)

# # Define a timespan
# t_span = (0,20)

# #Log message
# print("Starting solution")

# # Get the solution
# solution = solve_ivp(paper_model.evaluate, t_span, initial_joint_angles)

# # Extract the joint angles from the solution
# joint_angles = solution.y
# time_axis = solution.t
# for i in range(2):
#     plt.plot(time_axis,joint_angles[i,:])

# plt.legend(["Hip angle", "Knee angle"])
# plt.xlabel("Time (s)")
# plt.ylabel("Joint Angles")
# plt.show()