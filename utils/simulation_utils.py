# from .recordingsession_class import RecordingSession
# Ignore some warnings
import logging
logging.getLogger('root').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from IPython.display import clear_output
from .hr_utils import get_hr
from .eeg_utils import filter_eeg,EEG_FREQ_BANDS,get_score_accum
from .breathing_rate_utils import calculate_breathing_signal
from .plotting_utils import plot_eeg_score_accum

def dummy_func(data,a,b,c,d):
    return pd.DataFrame(columns=['Timestamp','data']),pd.DataFrame(columns=['Timestamp','data'])

def hr_sim(signal,collection_df,min_start,proc_wind,corr_wind,measure_run):
    full_df = pd.DataFrame(columns=['Timestamp','data'])
    fs = RecordingSession.aux_dict['hr']['sampling_f']
    
    if len(signal)>min_start and len(signal) > proc_wind:

        data = get_hr(signal['ECG'].iloc[-proc_wind:],fs)

        # Create new data
        new_data = pd.DataFrame(columns=['Timestamp','data'])
        new_data['data'] = data[-corr_wind:]

        new_length = len(signal)
        old_df_length = len(collection_df)

        new_points = new_length-old_df_length
        points_to_remove = corr_wind - new_points

        # Remove old data points from the start 
        collection_df = collection_df.iloc[:-points_to_remove]  
        
        # Append to HR dataframe
        collection_df = collection_df.append(new_data,ignore_index=True)
        full_signal = get_hr(signal['ECG'],fs)
        
    elif len(signal)<min_start:
        collection_df = pd.DataFrame(columns=['Timestamp','data'])
        full_signal = 0

    else:
        # Then just convert the full signal
        collection_df = pd.DataFrame(columns=['Timestamp','data'])

        collection_df['data'] = get_hr(signal['ECG'],fs)
        full_signal = collection_df['data'].copy()
        
    full_df = pd.DataFrame(columns=['Timestamp','data'])
    full_df['data']=full_signal
    return collection_df, full_df

def fixed_gen(int_number):
    yield int_number

def random_gen(max_nb):
    yield np.random.randint(max_nb)

def simulate_bmag(signal,collection_df,min_start,proc_wind,corr_wind,measure_run):
    full_df = pd.DataFrame(columns=['Timestamp','data'])

    if len(signal)>min_start and len(signal) > proc_wind:
        
        acc_data = signal.iloc[-proc_wind:].dropna()[['X','Y','Z']].to_numpy()/100
        data = calculate_breathing_signal(acc_data)

        # Create new data
        new_data = pd.DataFrame(columns=['Timestamp','data'])
        new_data['data'] = data[-corr_wind:]

        new_length = len(signal)
        old_df_length = len(collection_df)

        new_points = new_length-old_df_length
        points_to_remove = corr_wind - new_points

        # Remove old data points from the start 
        collection_df = collection_df.iloc[:-points_to_remove]  
        
        # Append to HR dataframe
        collection_df = collection_df.append(new_data,ignore_index=True)
        full_signal = calculate_breathing_signal(signal.dropna()[['X','Y','Z']].to_numpy()/100)
        
    elif len(signal)<min_start:
        collection_df = pd.DataFrame(columns=['Timestamp','data'])
        full_signal = 0

    else:
        # Then just convert the full signal
        collection_df = pd.DataFrame(columns=['Timestamp','data'])
        collection_df['data'] = calculate_breathing_signal(signal.dropna()[['X','Y','Z']].to_numpy()/100)

        full_signal = calculate_breathing_signal(signal.dropna()[['X','Y','Z']].to_numpy()/100)

    full_df = pd.DataFrame(columns=['Timestamp','data'])
    full_df['data']=full_signal
    return collection_df, full_df

def simulate_filter_eeg(signal,collection_df,min_start,proc_wind,corr_wind,measure_run):
    channels = ['TP9','AF7','AF8','TP10','aux']

    full_df = pd.DataFrame(columns=['Timestamp']+channels)
    if len(signal)>min_start and len(signal) > proc_wind:
        
        data = filter_eeg(signal.iloc[-proc_wind:].dropna()[channels])

        # Create new data
        new_data = pd.DataFrame(columns=['Timestamp']+channels)
        new_data[channels] = data[-corr_wind:]

        new_length = len(signal)
        old_df_length = len(collection_df)

        new_points = new_length-old_df_length
        points_to_remove = corr_wind - new_points

        # Remove old data points from the start 
        collection_df = collection_df.iloc[:-points_to_remove]  
        
        # Append to HR dataframe
        collection_df = collection_df.append(new_data,ignore_index=True)
        
        # if not measuring time: 
        if not measure_run:
            full_signal = filter_eeg(signal.dropna()[channels])
        else:
            full_signal = 0
            
    elif len(signal)<min_start:
        collection_df = pd.DataFrame(columns=['Timestamp']+channels)
        full_signal = 0

    else:
        # Then just convert the full signal
        collection_df = pd.DataFrame(columns=['Timestamp']+channels)
        collection_df[channels] = filter_eeg(signal.dropna()[channels])
        
        full_signal = collection_df[channels].copy()

    full_df = pd.DataFrame(columns=['Timestamp']+channels)
    full_df[channels]=full_signal
    return collection_df, full_df

def simulate_function_run(signal,function_to_run,
                          minimum_start,
                          processing_window,
                          correction_window,
                          data_label,
                          signal_label,
                          measure_run=False):
    starting_point =0
    live_signal = signal[:starting_point].copy()
    fig = plt.figure(figsize=[14,8])
    
    if data_label == 'EEG':
        channels = ['TP9','AF7','AF8','TP10','aux']
        collection_df = pd.DataFrame(columns=['Timestamp']+channels)
    elif data_label == 'EEG_score':
        channels = ['TP9','AF7','AF8','TP10']
        collection_df = np.zeros((0,len(EEG_FREQ_BANDS),len(channels))) 
    else:
        collection_df = pd.DataFrame(columns=['Timestamp','data'])
    
    max_step = 100

    while (len(live_signal)+max_step)<=len(signal):
        
        # FUNCTION HERE
        collection_df,full_signal = function_to_run(live_signal,
                                                          collection_df,
                                                          minimum_start,
                                                          processing_window,
                                                          correction_window,
                                                          measure_run)
        
        # Updating incoming data
        live_signal, starting_point = update_live_data(signal, live_signal, starting_point, random_gen(max_step))

        # Plotting   
        # Input signal  
        ax0 = plt.subplot(311)
        if data_label == 'ECG':
            ax0.plot(live_signal[data_label],label=signal_label)
        elif data_label=='ACC':
            ax0.plot(live_signal[['X','Y','Z']],label=signal_label)
        elif data_label in ['EEG']:
            ax0.plot(live_signal[['TP9','AF7','AF8','TP10','aux']],label=signal_label)
        elif data_label in ['EEG_score']:
            ax0.plot(live_signal[channels],label=signal_label)
        plt.legend(fontsize=10)
        
        # Processed signal
        ax1 = plt.subplot(312)
        if data_label == 'EEG':
            ax1.plot(collection_df['TP9'],c='r',label='Processed')
        elif data_label == 'EEG_score':
            # ax1.plot(collection_df[],c='r',label='Processed')
            plot_eeg_score_accum(collection_df,
                                    ax1,
                                    score_accum_list_areas=[],
                                    ylim_window='alpha',
                                    f_samp=256,
                                    time_offset=0)
        else:
            ax1.plot(collection_df['data'],c='r',label='Processed')

        plt.legend(fontsize=10)
        
        ax2 = plt.subplot(313)
        if data_label == 'EEG':
            ax2.plot(full_signal['TP9'],c='g',label='Full')
        else:
            ax2.plot(full_signal['data'],c='g',label='Full')

        plt.legend(fontsize=10)
        
        plt.show()
        clear_output(True)
        time.sleep(1)

    return collection_df,full_signal

def simulate_score_eeg(signal,collection_df,min_start,proc_wind,corr_wind,measure_run):
    # Assuming signal = filtered signal
    EEG_FREQ_BANDS = {'delta': [0.5, 4],
                             'theta': [4, 8],
                             'alpha': [8, 12],
                             'beta': [12, 35],
                             'gamma': [35, 119]}
    
    dummy_fullsignal_df = pd.DataFrame(columns= ['Timestamp','data'])

    collection_df = get_score_accum(signal,collection_df, window_size=512,fs=256)
    return collection_df,dummy_fullsignal_df

def update_live_data(df, live_df, start_point, random_gen):
    """
    Appends new samples to the live dataframe and updates the starting point.
    This function is useful for simulating live data updates in real-time.

    Args:
    df (pd.DataFrame): The original dataframe that new samples are drawn from.
    live_df (pd.DataFrame): The dataframe being updated with new data samples.
    start_point (int): The index in df where the new samples start.
    sample_incr (callable): Function to call to get the number of new samples.
    random_gen (generator): Generator to produce random increments.

    Returns:
    pd.DataFrame, int: The updated live dataframe and new starting point.
    """
    new_sample_increase = next(random_gen)
    new_samples = df[start_point : start_point + new_sample_increase].reset_index(drop=True)
    live_df = live_df.append(new_samples, ignore_index=True)
    start_point += new_sample_increase

    return live_df, start_point