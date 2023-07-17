from IPython.display import clear_output
import numpy as np
import pandas as pd
import pytz
import os
import asyncio
import copy 
import concurrent
from datetime import datetime
from .plotting_utils import plot_data
import matplotlib.pyplot as plt
import _tkinter
from .stimulator_class import StimulationManager
from .decision_models_class import DecisionModel
from .recordingdevice_class import Muse2, Cyton, PolarH10_recorder

class RecordingSession:
    
    # Frequency bands of EEG
    EEG_FREQ_BANDS = {'delta':[0.5,4],'theta':[4,8],'alpha':[8,12],'beta':[12,35],'gamma':[35,119]}

    SAVE_DIRECTORY = os.path.join("1-Data","1-Raw")

    def __init__(self,input_devices,stimulation_devices=[],decision_model=DecisionModel):

        self.input_devices  = input_devices 
        
        # Set the same timezone to all devices
        self.tz = pytz.timezone('America/New_York')
        for device in self.input_devices:
            device.tz = self.tz

        # Overwrite the aux dict with the ones from the device list
        self.aux_dict = merge_dicts([device.aux_dict for device in self.input_devices]) # Can give problems if more than one eeg with different fs and channels
        self.start_datetime = datetime.now()
        self.id = self.start_datetime.strftime("%Y%m%d_%H%M%S")
        self.notes = ""        
        self.data = merge_dicts([device.data for device in self.input_devices])

        # Auxiliary variables for processing
        self.timestamps_placed = False 
        self.durations_added = False
        self.cleaned = False
        self.segmented = False
        self.fig = plt.figure(figsize=[14,8])
        self.is_jupyter = is_jupyter()

        # Stimulation variables
        self.stimulation_devices = stimulation_devices

        # Create the decision model 
        self.decision_model = decision_model(self.input_devices, self.stimulation_devices)

        # Create a StimulationManager, returns None if no stimulation devices
        self.stimulation_periods_dict = {}
        self.stimulation_df = []
        self.stimulation_manager = StimulationManager.create(self.stimulation_devices,self.input_devices, self.decision_model)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __repr__(self):
        """
        Returns a summary of the class instance
        
        Returns: 
        ------------
        - class_info:   str
                         summary of object
        """

        session_data_info = self.get_session_data_info()
        notes_info = f'Notes:\t\t {self.notes}' if self.notes !='' else ''
        class_info = f'Recording start date: \t\t{self.start_datetime.strftime("%A, %B %d, %Y %I:%M:%S %p")}\n'\
                     f'======================         ===================================\n'\
                     f'{session_data_info}\n'\
                     f'{notes_info}\n'

        return class_info
   
    def add_notes(self, note):
        self.notes += f"{datetime.now()}: {note}\n"
    
    async def collect_data(self,rec_time=50,window_length =0):
        
        window_length_valid = validate_window_length(window_length)

        self.loop = asyncio.get_event_loop()

        # Wait until the streaming has started
        streaming_active = await self.detect_streaming()
        self.reset_devices()
        
        self.start_time = datetime.now().astimezone(self.tz)          
        self.current_date = datetime.now(self.tz).date()
        self.synchronize_devices()

        while (datetime.now(self.tz)-self.start_time).total_seconds()<rec_time:# TODO: and (not device_error):

            await self.loop.run_in_executor(self.executor, self.collect_device_data)

            # Take care of the stimulation if any
            if self.stimulation_manager:
                # await self.stimulation_manager.stimulate(self.data)
                stimulation_task = asyncio.create_task(self.stimulation_manager.stimulate(self.data))
                await stimulation_task
                self.stimulation_periods_dict = {stimulator.name:stimulator.stimulation_periods_list for stimulator in self.stimulation_devices} # Could in theory just update every time there's a new value

            # Plot the signals
            self.plot(window_length)
        self.get_durations()
        await self.stop_devices()
        self.generate_stimulation_df()

        if self.stimulation_manager:
            await self.stimulation_manager.stop()  # Stop the StimulationManager
        
    def collect_device_data(self):
        # Get data from devices
        for device in self.input_devices:
            device.fetch_data()
        self.data = merge_dicts([device.data for device in self.input_devices])
        # return 

    async def connect_all(self):
        # Run sync tasks
        self.connect_all_sync()

        # Run async tasks
        await self.connect_all_async()

    async def connect_all_async(self):

        self.async_tasks = [device.connect for device in self.input_devices+self.stimulation_devices if asyncio.iscoroutinefunction(device.connect)]

        # BleakClient can't handle two connections at once so even though it is asyncio it does them one by one
        for task in self.async_tasks:
            await task()

    def connect_all_sync(self):
        self.sync_tasks = [device.connect for device in self.input_devices+self.stimulation_devices if not asyncio.iscoroutinefunction(device.connect)]

        for task in self.sync_tasks:
            task()

    async def detect_streaming(self):
    
        # Initialize streaming detection to False
        streaming_detected = False
        spinners = ['-', '\\', '|', '/']
        spinner_index = 0

        self.current_date = datetime.now().date()

        # Continue the checking process until streaming is detected
        while not streaming_detected:

            # Check for data in each device
            for device in self.input_devices:

                # Fetch data
                device.fetch_data()

                # Get the first key in the data dictionary
                first_key = list(device.data.keys())[0]

                # If first dataframe is empty then wait and start over
                if device.data[first_key].empty:
                    streaming_detected = False
                    print(f'\rWaiting for {device.device_name.capitalize()} to start streaming ' + spinners[spinner_index % len(spinners)], end='', flush=True)
                    spinner_index += 1
                    await asyncio.sleep(1)
                    break
                else:
                    streaming_detected= True

        # Clear the spinner line
        print('\r', end='', flush=True)

        # Return the result of streaming detection
        return streaming_detected

    def generate_stimulation_df(self):
        # TODO: add the datetime 
        if self.stimulation_periods_dict:
            data = []
            for stimulator, times in self.stimulation_periods_dict.items():
                for start, end in times:
                    data.append([stimulator, start, end])
            self.stimulation_df = pd.DataFrame(data, columns=["Stimulator", "Start", "End"])

    def get_data(self):
        return self.data

    def get_durations(self):

        # Only calculate durations if they haven't been set yet
        if not self.durations_added:
            # Create new attributes with the durations of available data
            for data_type in list(self.data.keys()):
                duration_attr = f"{data_type}_duration"
                # setattr(self, duration_attr, str(self.data[data_type]['Timestamp'].iloc[-1] - self.data[data_type]['Timestamp'].iloc[0]))
                self.data[data_type].compute_duration()
                setattr(self, duration_attr, self.data[data_type].duration)

            # Set flag to avoid rerunning
            self.durations_added = True

    def get_session_data_info(self):
        session_data_info = ''
        for device in self.input_devices:
            for data_src,biosignal_object in device.data.items():

                # extra_space = self.aux_dict[key]['spacing']
                extra_space = biosignal_object.aux_dict['spacing']

                if not biosignal_object.empty:
                    session_data_info+= f'{biosignal_object.aux_dict["name"]} duration: {extra_space}\t\t'\
                                        f"{getattr(self,f'{data_src}_duration')}\n"
        return session_data_info
    
    def plot(self,window_length=0):
    
        # Adjust for calling it in jupyter notebook
        if not self.is_jupyter:
            try:
                self.fig.clf()  # Clear the figure
            except _tkinter.TclError:
                pass
        else:
            clear_output(wait=True)

        if len(window_length)>0:
           
            self.fig = plot_data(self.fig,
                               self.data.copy(),
                                sampling_f=self.aux_dict,
                                window_length=window_length,
                                stimulation_areas = copy.deepcopy(self.stimulation_periods_dict),
                                timezone=self.tz)
            
            plt.pause(0.001)
            # plt.show()
        else:   # TODO: maybe print elapsed time?
            pass 

    def reset_devices(self):
        # Deletes the data gathered in the devices
        for device in self.input_devices:
            device.initialize_data_dict()

    def save(self):
        directory = self.SAVE_DIRECTORY
        id_directory = f"{directory}/{self.id}"
        
        # Create new directory if it doesn't exist
        if not os.path.exists(id_directory):
            os.mkdir(id_directory)

        # Save recordings
        for datatype, biosignal_object in self.data.items():
            filename = f"{id_directory}/{biosignal_object.source_device.capitalize()}_{self.id}_{datatype.upper()}.csv"
            biosignal_object.save_data(filename)
        
        # Save stimulation times if any
        if len(self.stimulation_df)>0:
            filename = f"{id_directory}/StimulationPeriods_{self.id}_stims.csv"
            self.stimulation_df.to_csv(filename, index=False)

        # Save notes
        with open(f"{id_directory}/{biosignal_object.source_device.capitalize()}_{self.id}_notes.txt", "w") as f:
            f.write(self.notes)

    async def start_devices(self):
        
        # Start streaming input devices     

        for device in self.input_devices:
            if asyncio.iscoroutinefunction(device.start):
                await device.start()
            else:
                device.start()

    async def stop_devices(self):
        # Stop the input devices
        for device in self.input_devices:
            if asyncio.iscoroutinefunction(device.stop):
                await device.stop()
            else:
                device.stop()

        # Stop the stimulation devices
        if not self.stimulation_manager is None:
            await self.stimulation_manager.stop()
    
    def synchronize_devices(self):
        for stimulator in self.stimulation_devices:
            stimulator.start_time = self.start_time
        for device in self.input_devices:
            device.synchronize_data(self.start_time)

def is_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Just using standard Python interpreter

def merge_dicts(dict_list):
    result = {}
    for d in dict_list:
        result = {**result, **d}
    return result

def validate_window_length(window_length):
    valid_keys = set(list(Muse2.aux_dict.keys()) + list(Cyton.aux_dict.keys()) + list(PolarH10_recorder.aux_dict.keys()))

    # Check keys
    for key in window_length.keys():
        if key not in valid_keys:
            raise ValueError(f"Invalid key: {key}")

    # Check values
    for value in window_length.values():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid value: {value}. Values should be non-negative integers.")
    
    return True