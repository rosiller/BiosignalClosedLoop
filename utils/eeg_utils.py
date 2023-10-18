from .filtering import butter_bandpass_filter, notch_filter
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy
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

def concat_score_df_to_accum_df(score_df: pd.DataFrame, score_accum: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a DataFrame of scores for each frequency band and channel, and 
    appends this data to a running DataFrame (score_accum) that keeps track of these scores over time.

    Parameters
    ----------
    score_df : pandas.DataFrame
        A DataFrame with an index corresponding to frequency bands and columns corresponding to each channel.
        Each element represents the score for a particular frequency band and channel.

    score_accum : pandas.DataFrame
        A DataFrame that accumulates scores over time. This DataFrame should have columns corresponding to
        each channel, and also 'Band' and 'Timestamp' columns.

    Returns
    -------
    score_accum : pandas.DataFrame
        The updated DataFrame after appending the new scores.
    """
    
    # Check if score_accum is not empty. If it is not, then set the new timestamp to be one more than the last 
    # timestamp in score_accum. If it is empty, then set the new timestamp to 0.
    if not score_accum.empty:
        new_timestamp = score_accum['Timestamp'].iloc[-1] + 1
    else:
        new_timestamp = 0

    # Convert index (bands) into a column and add timestamp column.
    score_df = score_df.reset_index()
    score_df.rename(columns={'index': 'Band'}, inplace=True)
    score_df['Timestamp'] = new_timestamp

    # Concatenate the new scores with the accumulated scores. This is done row-wise, so the new scores are appended 
    # to the end of the DataFrame.
    score_accum = pd.concat([score_accum, score_df],ignore_index=True)

    return score_accum

def filter_eeg(eeg,f_low=0.5,f_high=40,fs=256):
    # Takes in only the df with the channel columns
    eeg_filt= notch_filter(eeg,freq=50, quality_factor=80, fs=fs) # Mindmonitor data has somehow a high freq component at this frequency
    eeg_filt = butter_bandpass_filter(eeg_filt,f_low,f_high,fs)
    return eeg_filt

def get_band_score(eeg_fft: pd.DataFrame, band: str) -> pd.Series:
    """
    Computes the score for a given frequency band by summing the power spectral densities 
    within the band's frequency range after normalizing the data.
    
    Parameters
    ----------
    eeg_fft : pandas.DataFrame
        A DataFrame where the index corresponds to frequency values and columns correspond to 
        power spectral densities for those frequencies.
        
    band : str
        The name of the EEG frequency band for which the score should be computed. The band should 
        correspond to one of the keys in the EEG_FREQ_BANDS dictionary.
        
    Returns
    -------
    pandas.Series
        A Series containing the score for the specified band for each column in the input DataFrame.
    """
    
    band_freq = EEG_FREQ_BANDS[band]
    mask = (eeg_fft.index >= band_freq[0]) & (eeg_fft.index <= band_freq[1])

    # Normalize the data
    normalized_data = eeg_fft.divide(eeg_fft.sum())
    normalized_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    normalized_data.fillna(0, inplace=True)

    # Compute the score
    band_values = (normalized_data[mask]).sum()

    return band_values

def get_eeg_randomness(psd):
   
    randomness = measure_randomness(psd)
    left = randomness[['TP9','AF7']].mean(axis=1)
    right = randomness[['AF8','TP10']].mean(axis=1)
    randomness_df = pd.DataFrame({
        'Left Hemisphere': left,
        'Right Hemisphere': right,
        'Delta': left - right
    })  
    return randomness_df

def compute_correlation(data1, data2):
    return np.corrcoef(data1, data2)[0, 1]

def get_eeg_correlation(eeg_data):
    correlation = pd.DataFrame(columns=['Left Hemisphere','Right Hemisphere'])
    left = compute_correlation(eeg_data['TP9'],eeg_data['AF7'])
    right = compute_correlation(eeg_data['TP10'],eeg_data['AF8'])
    correlation = correlation.append({'Left Hemisphere':left,'Right Hemisphere':right},ignore_index=True)
    return correlation

def get_score_accum(psd,score_accum):
    scores = get_scores(psd,weight=True)
    score_accum = concat_score_df_to_accum_df(scores,score_accum)

    return score_accum

def get_scores(eeg_fft: pd.DataFrame, weight: bool = True) -> pd.DataFrame:
    """
    Computes the scores for each specified EEG frequency band. Optionally, it can weight the scores by
    the width of the frequency band.
    
    Parameters
    ----------
    eeg_fft : pandas.DataFrame
        A DataFrame where the index corresponds to frequency values and columns correspond to 
        power spectral densities for those frequencies.
        
    weight : bool, optional (default=True)
        If True, the scores are divided by the width of their respective frequency bands. This gives 
        a form of normalization based on the width of the band.
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame where each column corresponds to a frequency band and values represent the 
        scores for those bands.
    """
    
    scores = {}
    
    for band in EEG_FREQ_BANDS:
        scores[band] = get_band_score(eeg_fft, band)
        
        if weight:
            band_width = EEG_FREQ_BANDS[band][1] - EEG_FREQ_BANDS[band][0]
            scores[band] /= band_width
            
    return pd.DataFrame(scores).T

def rename_columns_based_on_index(df):
    """
    Rename columns based on the index value, prefixing column names with 
    L_ or R_ depending on whether the index contains 'Left_Hemisphere' or 'Right_Hemisphere'.
    
    Parameters:
    df (DataFrame): Input DataFrame.
    
    Returns:
    DataFrame: DataFrame with renamed columns.
    """
    if 'Left_Hemisphere' in df.index:
        prefix = 'L_'
    elif 'Right_Hemisphere' in df.index:
        prefix = 'R_'
    else:
        return df  # Return the original dataframe if index condition is not met
    
    # Renaming columns by adding a prefix based on the index
    df.columns = [prefix + col for col in df.columns]
    return df.reset_index(drop=True)

def get_3d_score_accum(accum_data, df_left,df_right):
    
    df_right = rename_columns_based_on_index(df_right.copy())
    df_left = rename_columns_based_on_index(df_left.copy())

    # accum_data = pd.concat([accum_data, df_left, df_right], axis=0).reset_index(drop=True)
    new_row = pd.concat([df_left,df_right],axis=1)
    accum_data = pd.concat([accum_data,new_row],axis=0).reset_index(drop=True)
    return accum_data

def process_hemispheres_scores(psd, freq_bands=['Delta', 'Theta', 'Alpha']):
    """
    Process the hemispheres and return dataframes for each.
    
    Parameters:
    psd (DataFrame): Input DataFrame containing power spectral densities.
    freq_bands (list): List of frequency bands to keep.
    
    Returns:
    DataFrame, DataFrame: DataFrames for left and right hemispheres.
    """
    
    df_hemispheres = pd.DataFrame()

    # Averaging by hemisphere
    df_hemispheres['Left_Hemisphere'] = psd[['TP9', 'AF7']].mean(axis=1)
    df_hemispheres['Right_Hemisphere'] = psd[['AF8', 'TP10']].mean(axis=1)

    # Getting the score
    scores = get_scores(df_hemispheres)

    # Creating a DataFrame for the Left Hemisphere
    df_left = pd.DataFrame(scores['Left_Hemisphere']).T
    df_left.columns = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    # Creating a DataFrame for the Right Hemisphere
    df_right = pd.DataFrame(scores['Right_Hemisphere']).T
    df_right.columns = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    # Filtering to keep only desired bands
    df_left = df_left[freq_bands]
    df_right = df_right[freq_bands]

    return df_left, df_right

def measure_randomness(psd):
    
    psd_entropy = scipy_entropy(psd)
    
    # Min max normalizing
    # min_val, max_val = np.min(segment), np.max(segment)
    # segment_normalized = (segment - min_val) / (max_val - min_val)

    # Z score normalizing
    segment_normalized = (np.mean(psd)-psd) / (np.std(psd))
    segment_pdf, _ = np.histogram(segment_normalized, bins=100, density=True)
    time_entropy = scipy_entropy(segment_pdf)
    randomness = (psd_entropy + time_entropy) / 2
    randomness = pd.DataFrame(data=randomness[np.newaxis],columns=psd.columns)
    
    return randomness



