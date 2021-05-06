
import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame
import threading
from functools import lru_cache
import os
from scipy.io import loadmat

class Dataport_Dataset:

    def __init__(self):

        self.dataset_name = 'dataport'
        self.data_paths = {'joint_angles':'kinematics/jointangles'}
        self.file_name = './local-storage/InclineExperiment.mat'

        self.data = h5py.File(self.file_name)

    #This will generate the angles 
    def joint_angle_generator(self,subject,joint,left=True,right=True):
        for trial in raw_walking_data['Gaitcycle'][subject].keys():
            #Dont care about the subject details
            if trial == 'subjectdetails':
                continue
        
        #Get all the trials
        for leg in self.data['Gaitcycle'][subject][trial]['kinematics']['jointangles'].keys():
            if (leg == 'left' and left == False):
                continue
            if (leg == 'right' and right == False):
                continue

            for trial in self.data['Gaitcycle'][subject].keys():    
                #Return the numpy array for the trial
                yield self.data['Gaitcycle'][subject][trial]['kinematics']['jointangles'][leg][joint]['x'][:]


class R01_Dataset:

    def __init__(self):

        self.dataset_name = 'r01'
        self.data_paths = {'joint_angles':'kinematics/jointangles'}
        self.file_name = './local-storage/Normalized.mat'

        self.data = h5py.File(self.file_name)


    #This will generate the angles 
    def joint_angle_generator(self,subject,joint,left=True,right=True):
        ambulation_modes = ['Run','Walk']
        for mode in ambulation_modes:
            #Get all the speeds
            for speed in self.data['Normalized'][subject][mode].keys():
                #Get all the inclines
                for incline in self.data['Normalized'][subject][mode][speed].keys():
                    #Return the numpy array for the trial
                    out = np.array(self.data['Normalized'][subject][mode][speed][incline]['jointAngles'][joint][:])
                    print(out.shape)
                    yield out




def flatten(d, parent_key='', sep='/'):
    items = []
    for k,v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v,h5py._hl.group.Group):
            items.extend(flatten(v,new_key,sep=sep).items())
        else:
            items.append((new_key,None))
    return dict(items)




def flatten_list(d, parent_key='', sep='/'):
    items = []
    for k,v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v,h5py._hl.group.Group):
            items.extend(flatten_list(v,new_key,sep=sep))
        else:
            items.append(new_key)
    return items


def get_column_name(column_string_list,num_last_keys):
    filter_strings = ['right','left']
    filtered_list = filter(lambda x: not x in filter_strings, column_string_list)
    column_string = '_'.join(filtered_list)
    return column_string

def get_end_points(d,out_dict,parent_key='', sep='/',num_last_keys=4):

    for k,v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v,h5py._hl.group.Group):
            get_end_points(v,out_dict,new_key,sep=sep)
        #Where the magic happens when you reach an end point
        else:
            column_string_list = (parent_key+sep+k).split(sep)[-num_last_keys:]
            column_string = get_column_name(column_string_list,num_last_keys)
            
            if (column_string not in out_dict):
                out_dict[column_string] = [[new_key],[v]]
            else:
                out_dict[column_string][0].append(new_key)
                out_dict[column_string][1].append(v)


def flatten_dataport_dataset():
    pass
    #%%
    #speed_str_to_float

    file_name = './local-storage/InclineExperiment.mat'
    data = h5py.File(file_name)['Gaitcycle']

    out = flatten(data)

    prev_subject = ''
    prev_trial = ''
    prev_leg = ''
    total_dataframe = DataFrame()
    event_dataframe = DataFrame()

    for key in out.keys():
        #Log the current key, this is the usual cause of errors
        key_shape = data[key].shape
        key_len = len(key_shape)

        #Skip based on bad things happening
        if('subjectdetails' in key or\
            'frame' in key or\
            'stepsout' in key or\
            'description' in key or\
            'mean' in key or\
            'std' in key or\
            'emgdata' in key):
            print("Key " + key + " " + str(key_shape) + " (ignored)")
            continue
        else:
            print("Key " + key + " " + str(key_shape))
            
        
        key_split = key.split('/')
        
        #Get the leg that you are reading on, if you are looking in cycles its different
        if(key_split[2] == 'cycles'):
            leg = key_split.pop(3)
        else:
            leg = key_split.pop(4)
            
        subject = key_split.pop(0)
        trial = key_split.pop(0)
           
        #print(key_split)

        dataframe_column_name = '_'.join(key_split)

        #Save file when switching to another subject to save ram
        if(prev_subject != subject):
            print("Finished subject: " + prev_subject + " Next subject: " + subject)
            total_dataframe.to_parquet('local-storage/test/InclineExperiment_'+prev_subject+'.par')
            total_dataframe = DataFrame()
            prev_subject = subject

        #When you switch leg or trial you run the risk of getting
        #uneven data, therefore add to the total dataset
        if(prev_trial != trial or\
            prev_leg != leg):
                        
            event_dataframe['subject'] = subject
            
            speed_pointer = data[subject][trial]['description'][1][0]
            event_dataframe['speed'] = data[speed_pointer][0][0]
            
            incline_pointer = data[subject][trial]['description'][1][0]
            event_dataframe['incline'] = data[incline_pointer][0][0]
            
            event_dataframe['leg'] = leg
            
            time = data[subject][trial]['cycles'][leg]['time']
            #Todo: need to improve phase dot
            #In this dataset we get the actual time per stride which can be used to calculate
            #immediate phase dot. I dont think we get this in R01
            event_dataframe['phase_dot'] = 1/(time[0][-1]-time[0][0])

            phase = np.linspace(0,1,150)
            event_dataframe['phase'] = np.tile(phase,int(event_dataframe.shape[0]/150))
            

            total_dataframe = pd.concat([total_dataframe,event_dataframe])
            event_dataframe = DataFrame()
            
            prev_trial = trial
            prev_leg = leg

        

        if(key_len == 3):
            cols = data[key]
            for i in range(key_len):
                event_dataframe[dataframe_column_name+"_"+str(i)] = cols[:,i,:].flatten()
        else:
            new_column = np.array(data[key]).flatten()
            if(np.isnan(new_column).any()):
                raise ValueError("This column has nan" + key)
            event_dataframe[dataframe_column_name] = new_column

    total_dataframe.to_parquet('local-storage/test/InclineExperiment_AB10.par')
#%%




def flatten_r01_normalized():


    file_name = './local-storage/Normalized.mat'
    data = h5py.File(file_name)['Normalized']

    out = flatten(data)
    #print(out)


    prev_incline = ''
    prev_speed = ''
    total_dataframe = DataFrame()
    event_dataframe = DataFrame()
    for key in out.keys():
        
        #Log the current key
        key_shape = data[key].shape
        key_len = len(key_shape)
        print("Key " + key + " " + str(key_shape))
        
        #Skip based on bad things happening
        if('SubjectDetails' in key or\
            'markers' in key or
            'CutPoints' in key or\
            #This has uneven data and breaks the import
            'Stair' in key):
            continue

        key_split = key.split('/')
        subject = key_split.pop(0)
        mode = key_split.pop(0)
        #subject = 'AB01'
        speed = key_split.pop(0)
        incline = key_split.pop(0)
        
        #print(key_split)

        dataframe_column_name = '_'.join(key_split)

        if(prev_speed != speed or prev_incline != incline):
            print(prev_incline)
            event_dataframe['subject'] = subject
            event_dataframe['incline'] = incline
            event_dataframe['speed'] = speed
            total_dataframe = pd.concat([total_dataframe,event_dataframe], ignore_index=True)
            event_dataframe = DataFrame()
            prev_speed = speed
            prev_incline = incline

        

        if(key_len == 3):
            cols = data[key]
            for i in range(key_len):
                event_dataframe[dataframe_column_name+"_"+str(i)] = cols[:,i,:].flatten()
        else:
            event_dataframe[dataframe_column_name] = np.array(data[key]).flatten()


    print(total_dataframe)
    return total_dataframe


def quick_flatten_dataport(): 
    
    #%%
    file_name = '../local-storage/InclineExperiment.mat'
    h5py_file = h5py.File(file_name)['Gaitcycle']
    for subject in h5py_file.keys():
       
        data = h5py_file[subject]
        save_name = '../local-storage/test/dataport_flattened_partial_{}.parquet'.format(subject)
        
        @lru_cache(maxsize=5)
        def get_speed(trial):
            return data[data[trial]['description'][1][1]][0][0]
        
        # out = flatten_list(data)
        # print(len(out))
        
        output_dict = {}
        get_end_points(data,output_dict)
        #print(output_dict)
        output_dataframe = DataFrame()
        
        data_frame_dict = {}
        
        first_column = ""
        for column_name,endpoint_list in output_dict.items():
            
            #Pick out the first column
            if(first_column == ""):
                first_column = column_name
            
            if('subjectdetails' in column_name or\
                'cycles' in column_name or\
                'stepsout' in column_name or\
                'description' in column_name or\
                'mean' in column_name or\
                'std' in column_name):
                print(column_name + " " + str(len(endpoint_list[1])) + " (ignored)")
            else:
                print(column_name + " " + str(len(endpoint_list[1])))
                
                #Pick an arbitraty columns to get information about all the rows
                if(column_name == 'jointangles_ankle_x'):
                    #Yikes...
                    #The first list comprehension gives you a double list 
                    #with every trial, the second comprehension flattens out the list
                    trials = [x for experiment_name, dataset in zip(*endpoint_list) for x in [experiment_name.split('/')[0]]*dataset.shape[0]*dataset.shape[1] ]
                    legs = [x for experiment_name, dataset in zip(*endpoint_list) for x in [experiment_name.split('/')[-3]]*dataset.shape[0]*dataset.shape[1] ]
                    #Yikes and then some
                    @lru_cache(maxsize=5)
                    def get_time(trial,leg):
                        return data[trial]['cycles'][leg]['time']
                    
                    phase_dot_list = []
                    step_length_list = []
                    for experiment_name in endpoint_list[0]:
                            endpoint_split = experiment_name.split('/')    
                            trial = endpoint_split[0]
                            leg = endpoint_split[-3]
                            
                            time = data[trial]['cycles'][leg]['time']
                            speed = get_speed(trial)
                            phase_dot_list.append(np.repeat(1/(time[:,-1]-time[:,0]), 150))
                            step_length_list.append(np.repeat(speed*(time[:,-1]-time[:,0]), 150))
                            
                arr = np.concatenate(endpoint_list[1],axis=0).flatten()
                len_arr = arr.shape[0]
                
                len_key = len_arr/150.0
                if(len_key not in data_frame_dict):
                    data_frame_dict[len_key] = DataFrame()
                    
                data_frame_dict[len_key][column_name] = arr
        
        for df in data_frame_dict.values():
            if "jointangles_ankle_x" in df.columns:
                
                print(len(trials))
                print(df.shape[0])
                #They way that this is formatted, there is a data pointer and 
                # the data itself. The first access to "data" gets the pointer
                # and the second gets the incline
                @lru_cache(maxsize=5)
                def get_ramp(trial):
                    return data[data[trial]['description'][0][1]][0][0]
                
                df['ramp'] = [ get_ramp(trial) for trial in trials]
                
                @lru_cache(maxsize=5)
                def get_speed(trial):
                    return data[data[trial]['description'][1][1]][0][0]
    
                df['speed'] = [get_speed(trial) for trial in trials]
                df['trial'] = trials
                phase = np.linspace(0,(1-1/150),150)
                df['leg'] = legs
                df['phase'] = np.tile(phase,int(df.shape[0]/150))
                df['phase_dot'] = np.concatenate(phase_dot_list, axis = 0)
                df['step_length'] = np.concatenate(step_length_list,axis=0)
                df.to_parquet(save_name)
                
        
        

#%%

def get_end_points_R01(d,out_dict,parent_key='', sep='/',num_last_keys=2):

    for k,v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v,h5py._hl.group.Group):
            get_end_points_R01(v,out_dict,new_key,sep=sep)
        #Where the magic happens when you reach an end point
        else:
            column_string_list = (parent_key+sep+k).split(sep)[-num_last_keys:]
            column_string = get_column_name(column_string_list,num_last_keys)
            
            #Verify if we have a 3D matrix. If we do, we need to break it down
            
            if(len(v.shape) == 3):
                str_dict = {0:'x',1:'y',2:'z'}
                for i in range(3):
                    column_string_prepended = column_string+"_"+str_dict[i]
                    v_curr = v[:,i,:]
                    if (column_string_prepended not in out_dict):
                        out_dict[column_string_prepended] = [[new_key+"_"+str_dict[i]],[v_curr]]
                    else:
                        out_dict[column_string_prepended][0].append(new_key+"_"+str_dict[i])
                        out_dict[column_string_prepended][1].append(v_curr)  
            elif(len(v.shape)>1):   
                if (column_string not in out_dict):
                    out_dict[column_string] = [[new_key],[v]]
                else:
                    out_dict[column_string][0].append(new_key)
                    out_dict[column_string][1].append(v)



def test_flatten_R01(): 
    pass
    #%%
    file_name = '../local-storage/Normalized.mat'
    data = h5py.File(file_name, 'r')['Normalized']['AB01']
    
    # out = flatten_list(data)
    # print(len(out))
    
    output_dict = {}
    get_end_points_R01(data,output_dict,num_last_keys=1)
    #print(output_dict)
    output_dataframe = DataFrame()
    problem_children = []
    data_frame_dict = {}
    
    first_column = ""
    for column_name,endpoint_list in output_dict.items():
        
        #Pick out the first column
        if(first_column == ""):
            first_column = column_name
        
        try:
        
            if('subjectdetails' in column_name or\
                'CutPoint' in column_name or\
                'description' in column_name):
                print(column_name + " " + str(len(endpoint_list[1])) + " (ignored)")
            else:
                print(column_name + " " + str(len(endpoint_list[1])))
                arr = np.concatenate(endpoint_list[1],axis=0).flatten()
                len_arr = arr.shape[0]
                
                len_key = len_arr/150.0
                if(len_key not in data_frame_dict):
                    data_frame_dict[len_key] = DataFrame()
                    
                data_frame_dict[len_key][column_name] = arr
        except ValueError:
            problem_children.append(column_name)
   
   
    print(output_dataframe.columns)
    print(output_dataframe.head())
   

#%%


def find_bad_endpoints_for_emma(d,out_list,parent_key='', sep='/'):

    for k,v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v,h5py._hl.group.Group):
            find_bad_endpoints_for_emma(v,out_list,new_key,sep=sep)
        #Where the magic happens when you reach an end point
        else:
            
            #Verify if we have a 3D matrix. If we do, we need to break it down
            if (len(v.shape)==1):
                out_list.append((new_key,v))
            
def find_bad_endpoints_main():
    pass
    #%%
    file_name = '../local-storage/Emma_dataset/latest/Normalized.mat'
    data = h5py.File(file_name, 'r')['Normalized']
    
    # out = flatten_list(data)
    # print(len(out))
    
    output_list = []
    find_bad_endpoints_for_emma(data,output_list)
    
    print(output_list)
    
    
    bad_runs = set()
    for element, dataset in output_list: 
        bad_runs.add("/".join(element.split('/')[:4]))
        
    print(bad_runs)

#%%
def get_endpoints_AY():
    pass
    #%%
    substrings = ['ik','conditions']
    #Get all the directories that match substrings
    end_points = [directory for directory in os.walk('local-storage/AY-dataset/') if any([sub in directory[0] for sub in substrings])]
    
    #f = loadmat(end_points[0][0] +"/" + end_points[0][2][0]) 
    f = pd.read_parquet('local-storage/AY-dataset/AB06/10_09_18/levelground/ik/AY_AB06_test.par') 

    
    #Remove Nones from the list 
    
#%%
import matlab.engine
eng = matlab.engine.start_matlab()
#%%
def convert_mat_to_parquet(mat):
   
    mat_file = eng.load(mat)
    new_file_name = mat + ".parquet"
    eng.parquetwrite(new_file_name,mat_file['data'],nargout=0)
    
#%%
if __name__ == '__main__':
    pass
    #flatten_r01_normalized()
    #flatten_dataport_dataset()
    #main_dataport_threaded()
    #test_flatten()
