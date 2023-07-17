import numpy as np

from .filtering import butter_bandpass_filter
# from os import path
import pyaudio
# from scipy.signal import find_peaks
import time
# from .plotting_utils import plot_ppg_hr
# from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
from subprocess import call
from multiprocessing.sharedctypes import Synchronized
import os, sys, contextlib

# For the audio in MacOS 
if sys.platform == 'darwin':
    import osascript

# Necessary to ignore warnings of ALSA (soundcard)
@contextlib.contextmanager
def ignoreStderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

def acquire_shared_value(multiprocess_value:float)->float:
    """
    Get the value of the shared variable in another process. Also accepts a float value if the variable isn't shared.
    
    Parameters
    ----------
    multiprocess_value:         multiprocessing.Value, int or float
                                 The shared variable among processors
    
    Returns
    -------
    current_value:              float
                                 The value of the shared variable 
    """
    # Take care of the acquisition and releasing of a shared value
    if type(multiprocess_value)==Synchronized:
        multiprocess_value.acquire()
        current_value = multiprocess_value.value
        multiprocess_value.release()
        
    elif type(multiprocess_value) in [int,float]:
        current_value = multiprocess_value
        
    return current_value

def get_sound_change_from_delta(delta_change:float,mode:str='volume')->float:
    """
    Get the change of the sound stimulation when given a change in delta.
    
    Parameters
    ----------
    delta_change:           float
                             the value of the change (here considered between [-1,1])
    mode:                   str
                             the kind of sound stimuation {volume,speed}

    Returns
    -------
    factor:                 float
                             the value of the sound stimulation 
    """
    
    # The speed stimulation outputs a number betweeen [0.25, 2.0]
    if mode=='speed':
        default_lvl = 1
        min_lvl = 0.25
        max_lvl = 2.0
    # The volume stimulation outputs a number between [0,100]
    elif mode == 'volume':
        default_lvl = 20
        min_lvl = 0
        max_lvl = 100

    # Interpolate a value for the stimulation given the delta_change
    if delta_change >0:
        factor = np.interp(delta_change,[0,1.0],[default_lvl,max_lvl]).item()
    elif delta_change<0:
        factor = np.interp(delta_change,[-1,0],[min_lvl,default_lvl]).item()
    elif delta_change==0:
        factor = default_lvl
    
    # Clip the maximum and minimum values
    if factor > max_lvl:
        factor = max_lvl
    elif factor < min_lvl:
        factor = min_lvl
    return factor

def play_sound(wf,factor:float,speed:bool=False,duration:int=0)->None:
    """
    Play a sound file with a given factor (either volume level or speed of reproduction) which is defined by the speed variable. 
    
    Parameters
    ----------
    factor:                     multiprocessing.Value, int or float
                                 Defines the volume level or rate of reproduction of the sound
    speed:                      bool
                                 defines whether the factor will modify the speed or the volume level of the sound
    duration:                   int
                                 limit the duration of the sound played
    """
    
    # Restart from scratch
    wf.setpos(0)
    
    # Ignore ALSA errors
    with ignoreStderr():
        p = pyaudio.PyAudio()
    
    chunk = 2048#1024

    # Obtain shared variable value
    factor_val = acquire_shared_value(factor)

    # Open the file which is going to be reproduced
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=int(wf.getframerate()*factor_val) if speed else int(wf.getframerate()) ,
                    output=True)

    # Define the first chunk that will be reproduced
    data = wf.readframes(chunk)
    
    # Initialize exit variable
    k=0

    # Condition on duration
    if duration:
        start = time.time()
        condition = lambda x: x-start<duration
    else:
        condition = lambda x: True

        
    while data != '' and condition(time.time()):
        stream.write(data)
        data = wf.readframes(chunk)
        
        # Execute only once as the factor value is updated only once
        if (speed) and (k == 0):
            stream.stop_stream()
            stream.close()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=int(wf.getframerate()*factor_val),
                    output=True)
        elif (not speed) and (k==0):
            # Change the volume to factor
            # Consider the OS
            if sys.platform == 'darwin':
                osascript.osascript(f"set volume output volume {factor_val}")
            elif sys.platform in ['linux','linux2']:
                call(["amixer", "-D", "pulse", "sset", "Master", f"{factor_val}%",' ']) # Empty space removes verbosity

        # Finish stream with counter
        if k==11000:
            data=''
            stream.stop_stream()
            stream.close()
            p.terminate()
        k+=1

# def ppg_audio_controller(accum, mean_hr, samples_ppg,factor):
#     if len(samples_ppg)!=0:
#         accum = np.vstack([accum, samples_ppg])
#         samples_ppg = get_hr(accum[:,1])
#         if len(samples_ppg)!=0:
#             accum_hr = samples_ppg
#             mean_hr_val = np.array(accum_hr)[-3:].mean()
#             mean_hr = np.append(mean_hr,mean_hr_val)
#             plot_ppg_hr(mean_hr)

#             # audio
#             factor.acquire()
#             factor.value = np.interp(mean_hr[-1],[40,60,75],[0.5,1,1.5]).item()
#             factor.release()
        
#     else:
#         time.sleep(0.2)
    
#     return accum,mean_hr