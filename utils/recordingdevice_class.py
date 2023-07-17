import asyncio
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
from pylsl import StreamInlet,resolve_stream 
import re
from typing import Tuple
from .breathing_rate_utils import calculate_breathing_rate,calculate_breathing_signal
from .eeg_class import EEG
from .eeg_utils import filter_eeg,get_score_accum
from .hr_utils import get_hr
from .polar_class import PolarH10
from .simulation_utils import update_live_data,fixed_gen

class BiosignalData:
    def __init__(self, data_name:str,source_device:str,aux_dict:dict={},data:pd.DataFrame=pd.DataFrame()):
        self.source_device = source_device # The device which is generating this data {muse2,cyton,polarh10}
        self.aux_dict = aux_dict # contains the name, data columns, fs, etc
        self.computable = self.aux_dict['priority']!= None
        self.data_name = data_name # 'eeg', 'acc', 'eeg_filt', ...
        self.sample_counter=0
        self.start_time =  datetime.now() # To be overwritten later to actual recording start time 
        self.fs = self.aux_dict['sampling_f']
        self.data =  pd.DataFrame(columns=['Timestamp']+self.aux_dict['data_columns']) if data.empty else data
         
    @property
    def empty(self):
      return self.data.empty
    
    def get_last_n_seconds(self, n):
        # If the DataFrame is empty, return it
        if self.data.empty:
            return self.data.drop(columns='Timestamp'), False

        # Get the timestamp for n seconds ago
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'], unit='s')
        n_seconds_ago = self.data['Timestamp'].max() - pd.Timedelta(seconds=n)

        # Calculate the actual range covered by the data
        actual_range = (self.data['Timestamp'].max() - self.data['Timestamp'].min()).total_seconds()

        # Set the flag indicating whether the requested range is fully covered
        fully_elapsed = actual_range >= n

        # Select only the rows in the DataFrame that are from the last n seconds
        last_n_seconds_data = self.data[self.data['Timestamp'] > n_seconds_ago]

        # Drop the 'Timestamp' column and return the DataFrame
        return last_n_seconds_data.drop(columns='Timestamp'), fully_elapsed


    def add_timestamps(self):
        num_new_samples = len(self.data)
        new_timestamps = [self.start_time + timedelta(seconds=self.sample_counter / self.fs + i / self.fs) 
                            for i in range(num_new_samples)]
        
        self.data['Timestamp'] = new_timestamps 

        # Update the sample counter
        self.sample_counter += num_new_samples

    def append(self, other, ignore_index=True):
        self.data = self.data.append(other, ignore_index=ignore_index)
    
    def collect_data(self, new_data_chunk):
        """
        Collect a new chunk of data. The new data should be a list of values.
        Not using the pulled timestamps because Polar's are wrong
        """
        num_new_samples = len(new_data_chunk)

        if num_new_samples > 0:

            # Create a DataFrame from the new data chunk
            if self.data_name == 'eeg' and self.source_device == 'muse2':
                pulled_data = pd.DataFrame(new_data_chunk, columns=self.aux_dict['data_columns']+['aux']).drop(columns='aux')
            
            else:
                pulled_data = pd.DataFrame(new_data_chunk, columns=self.aux_dict['data_columns'])

            if self.data_name == 'eeg' and self.source_device=='cyton':
               
                self.data = self.data.append(new_data_chunk) # Including timestamp
                
                # Drop duplicates
                subset_cols = [col for col in self.data.columns if col != 'Timestamp']
                self.data = self.data.drop_duplicates(subset=subset_cols)

                # Update the corrected timestamps
                # Calculate correct timestamps for the new chunk
                new_timestamps = pd.date_range(start=self.start_time, periods=len(self.data), freq=f'{1/self.fs}S')
                self.data['Timestamp'] = new_timestamps

            else:

                # Calculate correct timestamps for the new chunk
                new_timestamps = [self.start_time + timedelta(seconds=self.sample_counter / self.fs + i / self.fs) 
                            for i in range(num_new_samples)]
                
                # Add the 'Timestamp' column to the DataFrame
                pulled_data['Timestamp'] = new_timestamps

                # Append the new data chunk to the accumulator
                self.data = self.data.append(pulled_data, ignore_index=True)

            # Update the sample counter
            self.sample_counter += num_new_samples
        
    def compute_duration(self):
        self.duration = str(self.data['Timestamp'].iloc[-1] - self.data['Timestamp'].iloc[0])

    def drop_duplicates(self,inplace):
        self.data.drop_duplicates(inplace=inplace)
        
    def save_data(self,filename):
        self.data.to_csv(filename, index=False)

class RecordingDevice:
   
    def __init__(self,input_string):
        # input_str. ex. 'polar ecg hr br'
        #            ex. 'muse eeg eeg_filt'
        self.input_string = input_string
        self.simulated,self.data_sources = analyze_input(input_string)
        self.timestamps_placed = False
        self.start_time =  datetime.now() # To be overwritten later when recording starts
        self.uses_asyncio = False

        if self.simulated:
            self.initialize_simulated_signal()
            # TODO: change the rec_time to the max length of the data
                
    def compute_bmag(self):
        
        self.compute_latest_data('acc','bmag', self.preprocess_data, 
                                 processing_window = 15*self.aux_dict['acc']['sampling_f'],
                                 minimum_start = 0.5*self.aux_dict['acc']['sampling_f'], 
                                 correction_window = 6*self.aux_dict['acc']['sampling_f'],
                                 fs=self.aux_dict['acc']['sampling_f'])

    def compute_br(self):
        if len(self.data['acc'].data)>10:
            # timestamp_seconds = self.data['acc'].data['Timestamp'].apply(timestamp_to_seconds)
            timestamp_seconds = series_timestamp_to_seconds(self.data['acc'].data['Timestamp'])
            self.data['br'].data = self.data['br'].data.drop(self.data['br'].data.index)
            self.data['br'].data['Breathing rate'] = calculate_breathing_rate(self.data['bmag'].data['Breathing magnitude'].to_numpy(),timestamp_seconds)
            # TODO: add Timestamp column 
            self.data['br'].add_timestamps()

    def compute_eeg_score(self):
        # TODO consider adding delay
        score_accum = get_score_accum(self.data['eeg_filt'].data,
                                      self.data['eeg_score'].data,
                                      window_size=512,
                                      channels = self.channels,
                                      fs=self.aux_dict['eeg']['sampling_f'])
        
        self.data['eeg_score'].data = score_accum 

    def compute_derivables(self):
        # Run the derivables
        for data_src,biosignaldata_object in self.data.items():
            if biosignaldata_object.computable:
                getattr(self,f'compute_{data_src}')()

    def compute_eeg_filt(self):
        self.compute_latest_data('eeg','eeg_filt', 
                                 self.preprocess_data, 
                                 10*self.aux_dict['eeg']['sampling_f'],
                                 5*self.aux_dict['eeg']['sampling_f'],
                                 6*self.aux_dict['eeg']['sampling_f'],
                                 self.aux_dict['eeg']['sampling_f'])

    def compute_ecg_hr(self):
        self.compute_latest_data('ecg','ecg_hr', 
                                 self.preprocess_data, 
                                 10*self.aux_dict['ecg_hr']['sampling_f'],#9
                                 4*self.aux_dict['ecg_hr']['sampling_f'],
                                 6*self.aux_dict['ecg_hr']['sampling_f'],
                                 self.aux_dict['ecg_hr']['sampling_f'])

    def compute_latest_data(self, input_label,output_label, processing_func, processing_window, minimum_start, correction_window, fs):
        
        input_data_columns = self.aux_dict[input_label]['data_columns']
        output_data_columns = self.aux_dict[output_label]['data_columns']
        
        # Wait until there is enough data so that processing_func can process it
        if len(self.data[input_label].data)>minimum_start and len(self.data[input_label].data) > processing_window:

            new_data = processing_func(self.data[input_label].data[input_data_columns].dropna().iloc[-processing_window:],
                                       output_label,fs=fs)
            
            # Match the size of the dataframe
            new_data = new_data[...,None] if len(new_data.shape)==1 else new_data

            # Create new Data
            new_data_df = pd.DataFrame(columns=['Timestamp']+output_data_columns)
            new_data_df[output_data_columns] = new_data[-correction_window:]
            
            # Add 'Timestamp' column
            timestamps = self.data[input_label].data['Timestamp']
            timestamps = timestamps.iloc[-len(new_data_df):].reset_index(drop=True)
            new_data_df['Timestamp'] = timestamps
            
            new_length = len(self.data[input_label].data)
            old_data_df_length = len(self.data[output_label].data)
            
            new_points = new_length-old_data_df_length
            points_to_remove = correction_window - new_points
                
            # Remove old data points from the start of data_df
            self.data[output_label].data = self.data[output_label].data.iloc[:-points_to_remove]  

            # Append to dataframe
            self.data[output_label].append(new_data_df, ignore_index=True)
        
        # If no data or not enough datapoints set zeros to rows of the length of input signal
        elif self.data[input_label].empty  or  len(self.data[input_label].data)<minimum_start:
            self.data[output_label].data = pd.DataFrame(np.zeros((len(self.data[input_label].data), len(output_data_columns)+1)), columns=['Timestamp']+output_data_columns)
            
            # Add 'Timestamp' column
            self.data[output_label].data['Timestamp'] =  self.data[input_label].data['Timestamp']

        # If processing window hasn't been reached then just use the full signal
        elif len(self.data[input_label].data) <= processing_window:
            
            self.data[output_label].data = pd.DataFrame(columns=['Timestamp']+ output_data_columns)
            result =  processing_func(self.data[input_label].data[input_data_columns].dropna(), 
                                                                           data_type=output_label,
                                                                           fs=fs)
            # Match the size of the dataframe
            result = result[...,None] if len(result.shape)==1 else result

            self.data[output_label].data[output_data_columns] = result
            
            # Add 'Timestamp' column
            self.data[output_label].data['Timestamp'] =  self.data[input_label].data['Timestamp']
            
    def fetch_data(self):
        # For simulated data fetch from the sim_data dataframe
        if self.simulated:
            self.fetch_simulated_data()
        else:
            self.fetch_streamed_data()

    def fetch_simulated_data(self):
        for data_src,sim_df in self.sim_data.items():

            live_signal, self.sim_starting_point = update_live_data(sim_df.data, 
                                                                    self.data[data_src].data,
                                                                    self.sim_starting_point,
                                                                    fixed_gen(self.aux_dict[data_src]['sampling_f'])
                                                                    # random_gen(self.sim_data_speed)
                                                                    )
            self.data[data_src].data = live_signal
    
    def fetch_streamed_data(self):
        raise "This function needs to be overwritten by a subclass. The selected recording device has no method 'fetch_streamed_data' "

    def initialize_data_dict(self):
        # Data dictionary containing a BiosignalData object as values for each data source
        if not check_priorities(self.data_sources,self.aux_dict):
            raise Exception(f'Priorities are not met check data sources')
        
        # Create dictionary with dataframes to collect data 
        # TODO: rename to signals
        self.data = {data_src:BiosignalData(data_src,self.device_name,self.aux_dict[data_src]) for data_src in self.data_sources}

    def initialize_simulated_signal(self):
        if self.simulated:
            self.device_name,self.sim_data = get_device_name_and_data_dict(file_string=self.input_string, aux_dict = self.aux_dict)
            self.sim_starting_point = 0
            self.timestamps_placed = True
            self.sim_data_speed= 250 # max nb of datapoints to fetch on a given iteration
            
            # Get the fs for the first df
            for data_src_name,data_src_df in self.sim_data.items():
                self.sim_fs = 1/(data_src_df.data.Timestamp.iloc[1]-data_src_df.data.Timestamp.iloc[0]).total_seconds()
                break

    def preprocess_data(self,data,data_type,fs):

        if data_type =='ecg_hr':
            hr = get_hr(data.squeeze(),fs)
            return hr
        elif data_type == 'bmag':
            acc_data = data.to_numpy()/100.0
            acc_data = calculate_breathing_signal(acc_data)
            return acc_data
        elif data_type == 'eeg_filt':
            return filter_eeg(data)
        
    async def stop(self):

        if not self.simulated:
            if asyncio.iscoroutinefunction(self.stream_object.stop):
                await self.stream_object.stop()
            else:
                self.stream_object.stop()
    
    def synchronize_data(self,start_time):
        self.start_time = start_time

        for biosignal_object in self.data.values():
            biosignal_object.start_time = self.start_time

class Muse2(RecordingDevice):
    aux_dict ={'eeg':{'name':'EEG',
                       'data_columns':['TP9','AF7','AF8','TP10'],
                       'sampling_f':256, 
                       'priority':None,
                       'spacing':'\t\t'},
                'eeg_filt':{'name':'EEG filtered',
                       'data_columns':['TP9','AF7','AF8','TP10'],
                       'sampling_f':256,
                       'priority':['eeg'],
                       'spacing':'\t'},
                'eeg_score':{'name':'EEG Score',
                       'data_columns':['TP9','AF7','AF8','TP10'],
                       'sampling_f':0.5,
                       'priority':['eeg','eeg_filt'],
                       'spacing':'\t'},
                'acc':{'name':'Acceleration',
                       'data_columns':['X','Y','Z'],
                       'sampling_f':200, # To confirm
                       'priority':None,
                       'spacing':'\t'},
                'gyro':{'name':'Gyroscope',
                       'data_columns':['X','Y','Z'],
                       'sampling_f':200, # To confirm
                       'priority':None,
                       'spacing':'\t'},
                'ppg':{'name':'Oxymetry',
                       'data_columns':['Ox'],
                       'sampling_f':200, # To confirm
                       'priority':None,
                       'spacing':'\t'}
                }
    
    def __init__(self, input_string):
        # Input string contains the data sources or the stimulation
        super().__init__(input_string)
        self.device_name = 'muse2'
        self.initialize_data_dict() 

        # Verify if the sources match the device's 
        if any(source not in self.aux_dict.keys() for source in self.data_sources):
            raise ValueError(f"Invalid data source: {self.data_sources} for {self.device_name}.")
        
        # Define the stream object, sampling frequency and channels
        self.stream_object = EEG(device = self.device_name) if not self.simulated else None
        self.fs = self.stream_object.sfreq if not self.simulated else self.aux_dict['eeg']
        self.channels =  self.aux_dict['eeg']['data_columns'] # TODO: should get this from the stream_object 

    def connect(self):
        if not self.simulated:

            self.stream_object.start(fn="None",
                                    eeg=True,
                                    ppg= 'ppg' in self.data_sources,
                                    acc='acc' in self.data_sources,
                                    gyro= 'gyro' in self.data_sources)
                
            self.inlet_dict = {data_source:get_inlet(data_source.upper()) for data_source in self.data_sources if self.aux_dict[data_source]['priority']==None }

    def fetch_streamed_data(self):
        
        # Collect data
        for inlet_data_name, inlet in self.inlet_dict.items():
            # self.data[inlet_data_name].collect_data(*inlet.pull_chunk())
            self.data[inlet_data_name].collect_data(inlet.pull_chunk()[0])

        self.compute_derivables()
    
    def start(self):
        # Starts the stream already in the connect
        pass

class Cyton(RecordingDevice):
    aux_dict ={'eeg':{'name':'EEG',
                       'data_columns':['Fp1','Fp2','C3','C4','P7','P8','O1','O2'],#['Fp1','Fp2','T9','T10','CP5','CP6','O1','O2'],
                       'sampling_f':250, 
                       'priority':None,
                       'spacing':'\t\t'},
                'eeg_filt':{'name':'EEG filtered',
                       'data_columns':['Fp1','Fp2','C3','C4','P7','P8','O1','O2'],
                       'sampling_f':250,
                       'priority':['eeg'],
                       'spacing':'\t'},
                'eeg_score':{'name':'EEG Score',
                       'data_columns':['Fp1','Fp2','C3','C4','P7','P8','O1','O2'],
                       'sampling_f':0.5, 
                       'priority':['eeg','eeg_filt'],
                       'spacing':'\t'}
                       }
    def __init__(self, input_string):
        # Input string contains the data sources or the stimulation
        super().__init__(input_string)
        self.device_name = 'cyton'
        self.initialize_data_dict()

        # Verify if the sources match the device's 
        if any(source not in self.aux_dict.keys() for source in self.data_sources):
            raise ValueError(f"Invalid data source: {self.data_sources} for {self.device_name}.")
        
        # Define the stream object, sampling frequency and channels
        self.stream_object = EEG(device = self.device_name) if not self.simulated else None
        self.fs = self.stream_object.sfreq if not self.simulated else self.aux_dict['eeg']
        self.channels =  self.aux_dict['eeg']['data_columns'] # TODO: should get this from the stream_object 

    def connect(self):
        if not self.simulated:

            self.stream_object.start(fn="None",
                                        eeg=True,
                                        ppg= 'ppg' in self.data_sources,
                                        acc='acc' in self.data_sources,
                                        gyro= 'gyro' in self.data_sources)
    
    def fetch_streamed_data(self):
        # Collect streamables
        for data_src,signal in self.data.items():
   
            # Removing the duplicates
            new_data_chunk = self.stream_object.get_recent(max_samples=500).reset_index().rename(columns={'index':'device timestamp'})  # Change max samples if it becomes too slow
            new_data_chunk['device timestamp'] =  pd.to_datetime(new_data_chunk['device timestamp'],unit='s',utc=True).dt.tz_convert(self.tz)

            signal.collect_data(new_data_chunk)
        self.compute_derivables()
        
    def start(self):
        # Starts the stream already in the connect
        pass

class PolarH10_recorder(RecordingDevice):
    aux_dict = {'ecg':{'name':'ECG',
                       'data_columns':['ECG'],
                       'sampling_f':130,
                       'priority':None,
                       'spacing':'\t\t',},
                'ecg_hr':{'name':'HR',
                          'data_columns':['Heart Rate'],
                          'sampling_f':130,
                          'priority':['ecg'],
                          'spacing':'\t\t'},
                'hr':{'name':'HR streamed',
                          'data_columns':['Heart Rate'],
                          'sampling_f':130,
                          'priority':None,
                          'spacing':'\t'},
                'acc':{'name':'Acceleration',
                       'data_columns':['X','Y','Z'],
                       'sampling_f':200,
                       'priority':None,
                       'spacing':'\t'},
                'br':{'name':'Breathing rate',
                      'data_columns':['Breathing rate'],
                       'sampling_f':0.25,
                       'priority':['acc','bmag'],
                       'spacing':'',},
                'bmag':{'name':'Breathing magnitude',
                        'data_columns':['Breathing magnitude'],
                        'sampling_f':200,
                        'priority':['acc'],
                        'spacing':''}
                        }
    def __init__(self, input_string):
        # Input string contains the data sources or the stimulation
        super().__init__(input_string)
        self.device_name = 'polarh10'
        self.initialize_data_dict()

        # Verify if the sources match the device's 
        if any(source not in self.aux_dict.keys() for source in self.data_sources):
            raise ValueError(f"Invalid data source: {self.data_sources} for {self.device_name}.")
        
        # Define the stream object
        self.stream_object = PolarH10() if not self.simulated else None
        self.uses_asyncio = True # Important for connection
        self.inlet_dict = {data_source:get_inlet(data_source.upper()) for data_source in self.data_sources if self.aux_dict[data_source]['priority']==None }

    async def connect(self):
        await self.stream_object.connect()

    async def start(self):
        await self.stream_object.run()

    def fetch_streamed_data(self):

        # Collect if inlet streams present
        for inlet_data_name, inlet in self.inlet_dict.items():
        #    self.data[inlet_data_name].collect_data(*inlet.pull_chunk())
           self.data[inlet_data_name].collect_data(inlet.pull_chunk()[0])
            
        # self.collect_data(pulled_data_dict)
        self.compute_derivables()

def analyze_input(input_string):
    # Split the input string into parts by space
    parts = input_string.split()
    
    # Pattern to match the 'Device_YYYYMMDD_HHMMSS_DataSource' format
    pattern = r'^\w+_\d{8}_\d{6}_\w+$'

    simulated = False
    data_sources = []

    for part in parts:
        # If the part matches the pattern, it's a simulated input
        if re.match(pattern, part):
            simulated = True
            # Extract the data source by splitting on underscore and taking the last part
            data_source = part.split('_')[-1].lower()
        else:
            # If it doesn't match the pattern, it's just a data source
            data_source = part

        data_sources.append(data_source)

    return simulated, data_sources

def check_priorities(input_list,aux_dict):
    for var in input_list:
        priorities = aux_dict[var]['priority']
        if priorities is not None:
            for priority in priorities:
                if priority not in input_list:
                    return False
    return True

def get_device_name_and_data_dict(file_string:str, aux_dict:dict={})->Tuple:
    # file_string like: PolarH10_YYYYMMDD_HHMMSS_DataType
    # example: PolarH10_20230614_145224_EEG

    file_string = file_string.replace(',', ' ').strip()
    data_dict = {}
    device_name_list = []
    for file in file_string.split(' '):
        
        data_source = '_'.join(file.split('_')[3:]).lower() # Splits the name by _ then takes the elements after the hours and joins them again with _
        device_name = file.split('_')[0].lower()
        data_dict[data_source] = BiosignalData(data_source,device_name,aux_dict=aux_dict[data_source],data = read_file(file))
        device_name_list.append(device_name)

    # Check if all elements in device_name_list are the same
    name_check_ok = len(set(device_name_list)) == 1 if device_name_list else False

    if not name_check_ok:
        raise f"Different devices obtained from the data files {device_name_list}"
    else:
        device_name = device_name_list[0]

    return device_name, data_dict

def get_inlet(value):
    #allowed_values = {'ECG','ACC','EEG'}
    # Get a list of all active LSL streams
    stream_ = resolve_stream('type', value)
    inlet_stream = StreamInlet(stream_[0])
    return inlet_stream

def read_file(file_name):
    
    directory = '_'.join( file_name.split('_')[1:3])  # Join the date and time parts together

    # df= pd.read_csv(f'./1-Data/1-Raw/{file_name}.csv')
    df= pd.read_csv(f'./1-Data/1-Raw/{directory}/{file_name}.csv')

    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

def series_timestamp_to_seconds(series):
    return series.dt.hour*3600 + series.dt.minute*60 + series.dt.second + series.dt.microsecond/1e6
