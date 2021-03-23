
import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame
import threading

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

def flatten_dataport_dataset():

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
			event_dataframe[dataframe_column_name] = np.array(data[key]).flatten()

	total_dataframe.to_parquet('local-storage/test/InclineExperiment_AB10.par')



def main_dataport_threaded():
	file_name = './local-storage/InclineExperiment.mat'
	data = h5py.File(file_name)['Gaitcycle']

	out = list(flatten(data).keys())
	subjects = ['AB01','AB02','AB03','AB04','AB05','AB06','AB07','AB08','AB09','AB10']

	x={}
	for subject in subjects[0:3]:
		filtered_out = filter(lambda x: subject in x, out)
		x[subject] = threading.Thread(target = flatten_dataport_threaded, args=(subject,data,filtered_out))
		x[subject].start()
	
	x['AB01'].join()
	x['AB02'].join()
	x['AB03'].join()

	for subject in subjects[3:6]:
		filtered_out = filter(lambda x: subject in x, out)
		x[subject] = threading.Thread(target = flatten_dataport_threaded, args=(subject,data,filtered_out))
		x[subject].start()
	
	x['AB04'].join()
	x['AB05'].join()
	x['AB06'].join()

	for subject in subjects[6:10]:
		filtered_out = filter(lambda x: subject in x, out)
		x[subject] = threading.Thread(target = flatten_dataport_threaded, args=(subject,data,filtered_out))
		x[subject].start()

	x['AB07'].join()
	x['AB08'].join()
	x['AB09'].join()

	subject = 'AB10'
	x[subject] = threading.Thread(target = flatten_dataport_threaded, args=(subject,data,filtered_out))
	x[subject].start()

	x['AB10'].join()


def flatten_dataport_threaded(subject, data, out):

	#speed_str_to_float
	local_subject = subject
	print("Created thread for " + local_subject)
	prev_trial = ''
	prev_leg = ''
	total_dataframe = DataFrame()
	event_dataframe = DataFrame()

	for key in out:
		#Log the current key, this is the usual cause of errors
		key_shape = data[key].shape
		key_len = len(key_shape)

		#Skip based on bad things happening
		if('subjectdetails' in key or\
			'cycles' in key or\
			'stepsout' in key or\
			'description' in key or\
			'mean' in key or\
			'std' in key or\
			'emgdata' in key):
			#print("Key " + key + " " + str(key_shape) + " (ignored)")
			continue
		else:
			#print("Key " + key + " " + str(key_shape))
			pass

		key_split = key.split('/')
		leg = key_split.pop(4)
		#The normal version uses pop so just use it here too
		subject = key_split.pop(0)
		subject = local_subject
		trial = key_split.pop(0)
		#print(key_split)

		dataframe_column_name = '_'.join(key_split)

		#Add to the total dataset if you changed leg or trial 
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

			#Jodo: Add phase
			phase = np.linspace(0,1,150)
			event_dataframe['phase'] = np.tile(phase,int(event_dataframe.shape[0]/150))
			total_dataframe = pd.concat([total_dataframe,event_dataframe],ignore_index = True)
			
			del event_dataframe
			event_dataframe = DataFrame()
			
			prev_trial = trial
			prev_leg = leg

		
		#Add a new column
		if(key_len == 3):
			cols = data[key]
			for i in range(key_len):
				event_dataframe[dataframe_column_name+"_"+str(i)] = cols[:,i,:].flatten()
		else:
			event_dataframe[dataframe_column_name] = np.array(data[key]).flatten()

	total_dataframe.to_parquet('local-storage/test/InclineExperiment_threaded_'+subject+'.par')
	print("Finalized thread for " + local_subject)

	return 



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

if __name__ == '__main__':
	#flatten_r01_normalized()
	#flatten_dataport_dataset()
	main_dataport_threaded()