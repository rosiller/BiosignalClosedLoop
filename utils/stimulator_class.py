from datetime import datetime,timedelta
from threading import Condition 
import pytz
import asyncio
import nest_asyncio
nest_asyncio.apply()
from .audio_utils import play_sound
import wave
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class StimulationManager:

    def __init__(self, stimulation_devices:list, input_devices:list=[], decision_model=None):
        self.input_devices = input_devices
        self.stimulation_devices = stimulation_devices
        
        # Initialize the ML model to be used
        self.decision_model = decision_model

    @classmethod
    def create(cls, stimulation_devices,input_devices,decision_model):
        # Only create object if there are stimulation devices
        if stimulation_devices:
            return cls(stimulation_devices,input_devices,decision_model)
        else:
            return None
        
    def convert_data_to_stim(self,data):        
        # Here's where the ML model takes the decision on the parameters of the stimulation and updates the values
        self.decision_model.update_stimulation_params(data)        
        
    def should_stimulate(self, data): 

        # Can also just run the mapping anyways and update the stimulator parameters then let the stimulator decide
        start_permission =  self.decision_model.ready_to_update_parameters(data)

        return start_permission
        
    async def stimulate(self,data=[]):
        if self.should_stimulate(data):
            # Update parameters if ready
            self.convert_data_to_stim(data)

            # Notify output devices of change
            for stimulator in self.stimulation_devices:
                if not stimulator.is_stimulating:
                    if asyncio.iscoroutinefunction(stimulator.stimulate):
                        await stimulator.stimulate()
                    else:
                        stimulator.stimulate()
        
    async def stop(self):
        for stimulator in self.stimulation_devices:
            if asyncio.iscoroutinefunction(stimulator.stop):
                await stimulator.stop()
            else:
                stimulator.stop()
                await asyncio.sleep(1)
class Stimulator: 

    _DEFAULT_PARAMS = {'Playback_speed':{'enabled':False,
                                'magnitude':1,
                               'duration':1},
                'Playback_volume':{'enabled':False,
                                'magnitude':1,
                               'duration':1}}
    
    def __init__(self,stimulator_type):
        """
        Wrapper for output devices. Defines the parameters the output device receives.
        Prepares the device for stimulation by creating a new thread and letting it wait for condition flat and 
            start  stimulation.
        """
        self.name = stimulator_type
        self.is_stimulating=False # Takes care of controlling new parameter updates
        self.start_time = None # To be overwritten by the Recording session once recording begins
        self.timezone = pytz.timezone('America/New_York')
        self.stimulation_periods_list = [] # Collects the stimulation periods
        self.params = self._DEFAULT_PARAMS.get(self.name, {})           
        self.stimulate_condition = Condition() # Flag from the Stimulation manager
        
        # Initialize the stimulation times
        self.stimulation_start_time = datetime.now(self.timezone)
        self.stimulation_end_time =  datetime.now(self.timezone)

    def stimulate(self):
        #Placeholder function
        self.is_stimulating = True
        # Here goes the code to activate stimulation
        raise NotImplementedError("This method should be overridden by subclass")
        # After the stimulation, set is_stimulating back to False
        self.is_stimulating = False

    def update_stimulation_period_list(self):
        # If the stimulation start time is greater than the end time, then it is just updating the start time
        if self.stimulation_start_time>self.stimulation_end_time:
            self.stimulation_periods_list.append([self.stimulation_start_time,None])
        # Otherwise then just update the last entry
        else: 
            self.stimulation_periods_list[-1][1]= self.stimulation_end_time
            # self.stimulation_periods_list.append([self.stimulation_start_time,self.stimulation_end_time])

class SoundStimulator(Stimulator):
       
    def __init__(self, stim_type:str='speed'):
        super().__init__(f'Playback_{stim_type.lower()}')
        self.stim_type = stim_type.lower()

        # Prepare file to play 
        # data_dir = path.abspath(path.join(path.abspath(''), '2-Data/3-Audio/',file))
        file = 'utils/BabyElephantWalk60_extended.wav'
        self.wf = wave.open(file, 'rb')

    async def monitor_stimulation(self):
        while datetime.now(self.timezone) < self.stimulation_end_time:
            await asyncio.sleep(1)  # Wait for 1 second before checking again
        self.is_stimulating = False

    def connect(self):
        # Doesn't connect to anything
        pass
    # def _run(self):
    #     while self.active:
    #         with self.stimulate_condition:
    #             self.stimulate_condition.wait()  # wait for the condition to be notified
    #             self.stimulate()


    async def stimulate(self):
        if self.params['enabled']:
            self.is_stimulating = True
            self.stimulation_start_time = datetime.now(self.timezone)
            self.update_stimulation_period_list()

            # Activate sound stimulation
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            play_sound_func = partial(play_sound, self.wf, self.params['magnitude'], speed=self.stim_type == 'speed', duration=self.params['duration'])
            loop.run_in_executor(executor, play_sound_func)

            # self.is_stimulating = False
            self.stimulation_end_time = datetime.now(self.timezone)+ timedelta(seconds=self.params['duration'])
            asyncio.create_task(self.monitor_stimulation())

            self.update_stimulation_period_list()


    # def stimulate(self):
    #     if self.params['enabled']:
    #         self.is_stimulating = True
    #         self.stimulation_start_time = datetime.now(self.timezone)
    #         self.update_stimulation_period_list()

    #         # Activate sound stimulation
    #         play_sound(self.wf, 
    #                 self.params['magnitude'], 
    #                 speed=self.stim_type == 'speed', 
    #                 duration=self.params['duration'])
            
    #         self.is_stimulating = False
    #         self.stimulation_end_time = datetime.now(self.timezone)
    #         self.update_stimulation_period_list()
    
    def stop(self):
        # Stops the thread
        self.active = False
        # with self.stimulate_condition:
        #     self.stimulate_condition.notify()  # wake up the thread if it's waiting
        # self.thread.join()