from .filtering import butter_bandpass_filter, notch_filter
import numpy as np
from scipy.signal import welch
import pandas as pd

EEG_FREQ_BANDS = {'delta':[0.5,4],'theta':[4,8],'alpha':[8,12],'beta':[12,35],'gamma':[35,119]}

def add_virtual_timestamps(df):
    """
    This function adds a new column to a DataFrame that assigns a virtual timestamp to each unique cycle of 
    bands in the 'Band' column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame. It is expected to have a column named 'Band'.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame with the added 'Timestamp' column.
    """
    
    # Get the unique values in the 'Band' column.
    # This gives us the sequence of bands that cycle through in the DataFrame.
    unique_bands = df['Band'].unique()
    
    # 'df['Band'] != df['Band'].shift()' creates a boolean Series that is True whenever the band changes.
    # 'groupby()' groups this Series into groups where the band stays the same.
    # 'cumcount()' returns the cumulative count of each group.
    # Dividing by the number of unique bands ('len(unique_bands)') gives us a new number every time 
    # the band cycles through all unique bands.
    # We then assign this to a new column 'Timestamp'.
    df = df.assign(Timestamp=df.groupby((df['Band'] != df['Band'].shift())).cumcount()//len(unique_bands))
    
    return df

def concat_score_np_to_accum_df(score, score_accum, channels):
    """
    This function takes a numpy array of scores for each frequency band and channel, and 
    appends this data to a running DataFrame (score_accum) that keeps track of these scores over time.

    Parameters
    ----------
    score : numpy.ndarray
        A 2D array with dimensions (frequency bands, channels), where each element represents
        the score for a particular frequency band and channel.

    score_accum : pandas.DataFrame
        A DataFrame that accumulates scores over time. This DataFrame should have columns corresponding to
        each channel, and also 'Band' and 'Timestamp' columns.

    channels : list
        A list of strings representing channel names.

    Returns
    -------
    score_accum : pandas.DataFrame
        The updated DataFrame after appending the new scores.
    """

    # Convert the numpy array of scores into a DataFrame with columns corresponding to each channel
    score_df = pd.DataFrame(score, columns=channels)

    # Check if score_accum is not empty. If it is not, then set the new timestamp to be one more than the last 
    # timestamp in score_accum. If it is empty, then set the new timestamp to 0.
    if score_accum.size != 0:
        score_df['Timestamp'] = score_accum['Timestamp'].iloc[-1] + 1
    else:
        score_df['Timestamp'] = 0

    # Add a new column 'Band' to the DataFrame, which contains the names of each frequency band.
    # The order of the bands is assumed to match the order in the 'score' array.
    score_df['Band'] = EEG_FREQ_BANDS.keys()

    # Define the new order of columns where 'Timestamp' and 'Band' are first and the rest of the columns follow.
    ordered_columns = ['Timestamp', 'Band'] + channels
    score_df = score_df[ordered_columns]

    # Concatenate the new scores with the accumulated scores. This is done row-wise, so the new scores are appended 
    # to the end of the DataFrame.
    score_accum = pd.concat([score_accum, score_df])

    return score_accum

def filter_eeg(eeg,f_low=0.5,f_high=40,fs=256):
    # Takes in only the df with the channel columns
    eeg_filt= notch_filter(eeg,freq=50, quality_factor=80, fs=fs) # Mindmonitor data has somehow a high freq component at this frequency
    eeg_filt = butter_bandpass_filter(eeg_filt,f_low,f_high,fs)
    return eeg_filt

def get_band_frequencies(band_name:str)->list:
    """
    Returns a list with the minimum and maximum frequencies for the given frequency band
    
    Parameters
    ----------
    band_name                       : string
                                        The desired frequency band
        
    Returns
    -------
    FREQ_BANDS[band_name]           : int list 
                                        frequency band
    """
    band_name = band_name.lower()
    
    # assert that band argument is correct
    possible_freq_bands = EEG_FREQ_BANDS
    freq_bands = " -- ".join(possible_freq_bands)
    assert(band_name in possible_freq_bands), "This is not a valid frequency band. Possible frequency bands : "+freq_bands
    
    return EEG_FREQ_BANDS[band_name]

def get_band_score(eeg_fft:np.ndarray, freqs:np.ndarray, band:str='alpha')->np.ndarray:
    """
    Computes the score for the given frequency band
    
    Parameters
    ----------
    eeg_fft                     : numpy array
                                    FFT of the eeg recording (datapoints, channels)
    freqs                       : numpy array
                                    array with frequency labels of the signal
    band                        : string
                                    desired band to process
    
    Returns
    -------
    band_values                 : numpy array
                                    scores for the desired band (channels) 
    """
    # Get a list with the frequency band limits
    band_freq = get_band_frequencies(band)

    # Create a zero array with ones on the given frequency band 
    filterband = np.zeros((freqs.shape))
    filterband[np.where((freqs>=band_freq[0]) & (freqs<=band_freq[1]))] = 1
    
    # Repeat this array in another dimension with the number of channels 
    filterband = np.tile(filterband,(eeg_fft.shape[-1],1)).T

    # Normalize dividing by the total area depending on the channel
    #  converting divisor to numpy to avoid warning
    # eeg_fft = eeg_fft/np.array(eeg_fft.sum(axis=0))[np.newaxis,:]
    divisor = np.array(eeg_fft.sum(axis=0))
    if (divisor==0).any():
        divisor = np.where(divisor == 0, 1, divisor)

    eeg_fft= eeg_fft/divisor[np.newaxis,:]

    # Get the area under the curve in the range of the desired frequencies
    band_values = filterband*eeg_fft
    band_values = band_values.sum(axis=0)

    return band_values

def get_psd(signal,fs,to_df = False,channels=['TP9', 'AF7', 'AF8', 'TP10']):
    psd_array = []
    for channel in signal:
        frequencies, psd = welch(signal[channel], fs, nperseg=fs*2)
        psd_array.append(psd)
    psd_array =np.stack(psd_array)
    if to_df:
        psd_array = pd.DataFrame(psd_array.T,columns= channels)
    return frequencies, psd_array

def get_scores(eeg_fft:np.ndarray, freqs:np.ndarray, weight:bool = True)->np.ndarray:
    """
    Returns array with scores for all the frequency bands
    
    Parameters
    ----------
    eeg_fft                 : numpy array
                                FFT of the eeg recording (datapoints, channels)
    freqs                   : numpy array
                                array with frequency labels of the signal
    weight                  : bool
                                determines whether the frequencies will be weighted by the size of the frequency band
    
    Returns
    -------
    score_array             : numpy array
                                Score of the total EEG signal (frequency bands, channels)
    """
    
    # Creates an empty array of size (frequency bands, channels)
    nb_channels = eeg_fft.shape[-1]
    score_array = np.zeros((len(EEG_FREQ_BANDS),nb_channels))
    for ind,band in enumerate(EEG_FREQ_BANDS):
        
        score = get_band_score(eeg_fft,freqs, band=band)
        score_array[ind] = score
        
        if weight:
            # Get the frequency band
            band_freq = get_band_frequencies(band)

            # weight by width of the band
            score_array[ind] /= (band_freq[1]-band_freq[0])

    return score_array

def get_score_accum(eeg,score_accum, window_size=512,channels=['TP9', 'AF7', 'AF8', 'TP10'],fs=256):
    
    assert isinstance(eeg, pd.DataFrame), "Input must be a pandas DataFrame"
    assert set(channels).issubset(eeg.columns), "Input DataFrame must contain the required channels"
    

    if len(eeg)>=window_size:
        freqs, psd = get_psd(signal=eeg[channels].iloc[-window_size:],fs=fs,channels=channels)
        score = get_scores(psd.T, freqs=freqs, weight= True)
    
    # If not enough data points for PSD then set scores to zero
    else:
        score = np.zeros((len(EEG_FREQ_BANDS),len(channels))) 
    score_accum = concat_score_np_to_accum_df(score,score_accum,channels) # concatenate in dataframe

    return score_accum





