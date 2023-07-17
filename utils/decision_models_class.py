from datetime import datetime
import math
import numpy as np

class SignalProcessor:
    # Process the "raw data", compute common features, keep intermediate processed signals perhaps
    def __init__(self,input_dict):
        self.available_inputs = input_dict

    def compute_baselines(self):
        pass

    def compute_mean(self):
        # compute the mean of the last 5 seconds of data
        pass

    def compute_angles_to_north_direction(self):
        pass

class DecisionModel:
    """
    Placeholder class that just sends the same output everytime
    """
    def __init__(self,input_list, output_list):
        # Prepare the signal processor
        self.input_list = input_list # List of recordingDevice
        self.output_list = output_list # List of Stimulator classes
        # self.sig_processor = SignalProcessor(input_list) # keep intermediate processed values
    
    def update_stimulation_params(self,data):
        for stimulator in self.output_list:
            if stimulator.name == 'GVS':
                
                stimulator.params['enabled']=True
                stimulator.params['multiplier']=1
                stimulator.params['duration']=5000

            elif stimulator.name == 'Playback_speed':
                stimulator.params['enabled']=True
                stimulator.params['magnitude']=1
                stimulator.params['duration']=5

            elif stimulator.name == 'Playback_volume':
                stimulator.params['enabled']=True
                stimulator.params['magnitude']=50
                stimulator.params['duration']=5
            else:
                print(f"Unknown stimulator type: {stimulator.name}")
    
    def ready_to_update_parameters(self,data):
        return True
    
class SimpleWait(DecisionModel):
    '''
    Same as Decision Model but with a waiting time before each stimulation
    '''
    def __init__(self,input_list, output_list):
        super().__init__(input_list,output_list)
    
    def check_wait_time(self,wait_time=1):

        self.stimulation_refractory_period_elapsed = False
        
        # TODO: Can set a different wait time for each output (type)
        for output in self.output_list:
            # If the output has a record and is not stimulating
            if output.stimulation_periods_list and not output.is_stimulating:
                last_stim_record = output.stimulation_periods_list[-1][1] # Get the end time of the last period
                
                if (datetime.now(output.timezone)- last_stim_record).total_seconds()>wait_time:
                    self.stimulation_refractory_period_elapsed = True
            
            elif not output.is_stimulating:
                self.stimulation_refractory_period_elapsed = True

    def ready_to_update_parameters(self,data):

        # There has to be at least X seconds after last stimulation finished
        self.check_wait_time(wait_time=10)
        
        return self.stimulation_refractory_period_elapsed
    
class SimpleInterpolator(SimpleWait):

    """
    This class is in charge of mapping the input data to the output devices.
    It consists of several steps:
        1. Decide which output devices will it use depending on the available input/output devices
        2. Process the input data 
        2.1 Save intermediate processed values (like baselines) 
        3. Decide the trigger to update the output parameters
        4. Map the biosignal data to stimulation parameters
        5. Update the output devices' parameters
    """
    # Simple model that uses the averaged EEG_score data of a given frequency band and 
    #  updates the gvs, playback volume's and speed's magnitude parameter to an interpolation
    #  based on a baseline. The durations are all fixed 

    def __init__(self,input_list, output_list):
        super().__init__(input_list,output_list)
        self.baseline_computed= False
        self.stimulation_refractory_period_elapsed = False
    
    def compute_baseline(self,data):
        if not self.baseline_computed:
            # If the hr is present 
            if 'ecg_hr' in data.keys():

                val,fully_elapsed = data['ecg_hr'].get_last_n_seconds(10)
                if fully_elapsed:
                    self.baseline_computed = True

            self.baseline_computed
    
    def interpolate_value(self,value, input_range, target_range):
        # Check if the value is within the correct range
        if not input_range[0] <= value <= input_range[1]:
            raise ValueError(f"Value should be between {input_range[0]} and {input_range[1]}")

        # Calculate the interpolated value
        interpolated_value = ((value - input_range[0]) / 
                            (input_range[1] - input_range[0]) * 
                            (target_range[1] - target_range[0]) + target_range[0])

        return interpolated_value

    def interpolate_value_exponential(self, value, input_range, target_range):
        # Check if the value is within the correct range
        if not input_range[0] <= value <= input_range[1]:
            raise ValueError(f"Value should be between {input_range[0]} and {input_range[1]}")

        # Convert the input and target ranges to a logarithmic scale
        log_input_range = [math.log(x) for x in input_range]
        log_target_range = [math.log(x) for x in target_range]

        # Perform the linear interpolation in the logarithmic space
        interpolated_value_log = ((math.log(value) - log_input_range[0]) / 
                                (log_input_range[1] - log_input_range[0]) * 
                                (log_target_range[1] - log_target_range[0]) + log_target_range[0])

        # Convert the interpolated value back to the original space
        interpolated_value = math.exp(interpolated_value_log)

        return interpolated_value

    def ready_to_update_parameters(self,data):
        # Flag to the stimulation manager to start calling the parameter updates 
        
        # In this case it only waits for the baseline to be computed
        if not self.baseline_computed:
            self.compute_baseline(data)

        # There has to be at least X seconds after last stimulation finished
        self.check_wait_time(wait_time=5)
        
        return self.baseline_computed and self.stimulation_refractory_period_elapsed

    def update_gvs_params(self,stimulator,data):
        if 'ecg_hr' in data.keys():
            # Take the mean last second of hr 
            val = data['ecg_hr'].get_last_n_seconds(1).mean().item()
            hr_range = (50, 140)
            if (val>hr_range[0]) and (val<hr_range[1]):
                stimulator.params['enabled'] = True
                stimulator.params['multiplier'] = self.interpolate_value(val, hr_range, (0.1, 2))
                stimulator.params['duration'] = 2
            else:
                stimulator.params['enabled'] = False

    def update_playback_speed_params(self,stimulator,data):
        if 'ecg_hr' in data.keys():
            # Take the mean last second of hr 
            val = data['ecg_hr'].get_last_n_seconds(1).mean().item()
            hr_range = (50, 140)
            if (val>hr_range[0]) and (val<hr_range[1]):
                stimulator.params['enabled'] = True
                stimulator.params['magnitude'] = self.interpolate_value(val, hr_range, (0.1, 2))
                stimulator.params['duration'] = 2
            else:
                stimulator.params['enabled'] = False
        # Reset the flag 
        # self.stimulation_refractory_period_elapsed= False
        
    def update_playback_volume_params(self,stimulator,data):
        if 'ecg_hr' in data.keys():
            # Take the mean last second of hr 
            val = data['ecg_hr'].get_last_n_seconds(1).mean().item()
            hr_range = (50, 100)
            if (val>hr_range[0]) and (val<hr_range[1]):
                stimulator.params['enabled'] = True
                stimulator.params['magnitude'] = self.interpolate_value_exponential(val, hr_range, (20, 100))
                stimulator.params['duration'] = 2
            else:
                stimulator.params['enabled'] = False

    def update_stimulation_params(self,data):
        for stimulator in self.output_list:
            if stimulator.name == 'GVS':
                self.update_gvs_params(stimulator,data)
            elif stimulator.name == 'Playback_speed':
                self.update_playback_speed_params(stimulator,data)
            elif stimulator.name == 'Playback_volume':
                self.update_playback_volume_params(stimulator,data)
            else:
                print(f"Unknown stimulator type: {stimulator.name}")

class InterpolatorFromBaseline(SimpleInterpolator):
    def __init__(self,input_list, output_list):
        super().__init__(input_list,output_list)
        self.baseline_computed= False
        self.stimulation_refractory_period_elapsed = False

    def compute_baseline(self,data):
        if not self.baseline_computed:
            # If the hr is present 
            # if 'ecg_hr' in data.keys():

            if 'eeg_score' in data.keys():
                val,fully_elapsed = data['eeg_filt'].get_last_n_seconds(10)
                if fully_elapsed:
                    pass
            else: 
                # No other condition here if there is no eeg_score then just behave as a normal interpolator
                pass

            self.baseline_computed = True
