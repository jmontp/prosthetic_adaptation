
import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame
import threading
from functools import lru_cache
import os
from scipy.io import loadmat


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
