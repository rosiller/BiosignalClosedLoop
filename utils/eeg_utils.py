from .filtering import butter_bandpass_filter, notch_filter
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy
import pandas as pd
import asrpy
import mne

EEG_FREQ_BANDS = {'delta':[0.5,4],'theta':[4,8],'alpha':[8,12],'beta':[12,35],'gamma':[35,119]}

def add_virtual_timestamps(df:pd.DataFrame)->pd.DataFrame:
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

def filter_eeg(eeg: pd.DataFrame, f_low: float = 0.5, f_high: float = 40, fs: int = 256, with_mne: bool = False) -> np.array:
    """
    Filters EEG data by applying bandpass and notch filters, and optionally an ASR (Artifact Subspace Reconstruction).
    
    Parameters:
    eeg (pd.DataFrame): Input EEG data as a DataFrame.
    f_low (float, optional): Low cutoff frequency for the bandpass filter. Default is 0.5 Hz.
    f_high (float, optional): High cutoff frequency for the bandpass filter. Default is 40 Hz.
    fs (int, optional): Sampling frequency of the EEG data. Default is 256 Hz.
    with_mne (bool, optional): Whether to use MNE for filtering and ASR. Default is False.
    
    Returns:
    np.array: Filtered EEG data.
    """
    # Check if MNE should be used for filtering and ASR
    if with_mne:
        # Initialize the ASR with given sampling frequency and cutoff
        asr = asrpy.ASR(sfreq=fs, cutoff=20)
        
        # Create MNE info structure with channel names and types
        info = mne.create_info(ch_names=eeg.columns.to_list(), ch_types=["eeg"] * 4, sfreq=fs)
        
        # Create an MNE RawArray and apply bandpass and notch filters
        eeg_mne = mne.io.RawArray(eeg.T, info).filter(l_freq=f_low, h_freq=f_high, verbose=0).notch_filter(50, verbose=0)
        
        # Fit the ASR model
        asr.fit(eeg_mne)
        
        # Transform the EEG data using the ASR model and transpose the result
        eeg_filt = asr.transform(eeg_mne)._data.T
    else:
        # Apply a notch filter to remove 50 Hz component (likely powerline interference)
        eeg_filt = notch_filter(eeg, freq=50, quality_factor=80, fs=fs)
        
        # Apply a bandpass filter to keep frequencies between f_low and f_high
        eeg_filt = butter_bandpass_filter(eeg_filt, f_low, f_high, fs)
    
    # Return the filtered EEG data
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

def get_eeg_randomness(psd: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the randomness of EEG data based on the Power Spectral Density (PSD).
    
    Parameters:
    psd (pd.DataFrame): Input PSD as a DataFrame.
    
    Returns:
    pd.DataFrame: A DataFrame containing the calculated randomness for the left hemisphere,
                  right hemisphere, and the difference (Delta) between them.
    """
    
    # Measure randomness from the PSD
    randomness = measure_randomness(psd)
    
    # Calculate the mean randomness for left hemisphere electrodes (TP9, AF7)
    left = randomness[['TP9', 'AF7']].mean(axis=1)
    
    # Calculate the mean randomness for right hemisphere electrodes (AF8, TP10)
    right = randomness[['AF8', 'TP10']].mean(axis=1)
    
    # Create a new DataFrame to store the calculated randomness measures
    randomness_df = pd.DataFrame({
        'Left Hemisphere': left,  # Randomness of the left hemisphere
        'Right Hemisphere': right,  # Randomness of the right hemisphere
        'Delta': left - right  # Difference between the left and right hemisphere randomness
    })
    
    # Return the resulting DataFrame
    return randomness_df

def compute_correlation(data1: pd.Series, data2: pd.Series) -> float:
    """
    Computes the Pearson correlation coefficient between two data series.
    
    Parameters:
    data1 (pd.Series): The first data series.
    data2 (pd.Series): The second data series.
    
    Returns:
    float: The Pearson correlation coefficient between the two input data series.
    """
    
    # Calculate the Pearson correlation coefficient between the two input data series
    # The function returns a matrix, where the value at position [0, 1] is the correlation coefficient between data1 and data2
    return np.corrcoef(data1, data2)[0, 1]

def get_eeg_correlation(eeg_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Pearson correlation coefficient between pairs of EEG channels,
    one pair for the left hemisphere and one pair for the right hemisphere.
    
    Parameters:
    eeg_data (pd.DataFrame): DataFrame containing the EEG data with columns named after EEG channels.
    
    Returns:
    pd.DataFrame: A DataFrame containing the calculated correlation coefficients for the specified pairs of channels.
    """
    
    # Creating an empty DataFrame with specified columns to store the correlation results
    correlation = pd.DataFrame(columns=['Left Hemisphere', 'Right Hemisphere'])
    
    # Computing the Pearson correlation coefficient between the TP9 and AF7 channels (Left Hemisphere)
    left = compute_correlation(eeg_data['TP9'], eeg_data['AF7'])
    
    # Computing the Pearson correlation coefficient between the TP10 and AF8 channels (Right Hemisphere)
    right = compute_correlation(eeg_data['TP10'], eeg_data['AF8'])
    
    # Appending the calculated correlation coefficients to the correlation DataFrame
    correlation = correlation.append({'Left Hemisphere': left, 'Right Hemisphere': right}, ignore_index=True)
    
    # Returning the DataFrame containing the correlation coefficients
    return correlation

def get_score_accum(psd: pd.DataFrame, score_accum: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves scores from the given Power Spectral Density (PSD) DataFrame and accumulates them in another DataFrame.
    
    Parameters:
    psd (pd.DataFrame): Input DataFrame containing the PSD.
    score_accum (pd.DataFrame): DataFrame where the retrieved scores are to be accumulated.
    
    Returns:
    pd.DataFrame: Updated DataFrame containing the accumulated scores.
    """
    
    # Getting scores from the PSD, weights can be applied to the scores
    scores = get_scores(psd, weight=True)
    
    # Concatenating the newly retrieved scores to the existing accumulated scores DataFrame
    score_accum = concat_score_df_to_accum_df(scores, score_accum)
    
    # Returning the updated DataFrame containing the accumulated scores
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

def rename_columns_based_on_index(df:pd.DataFrame)->pd.DataFrame:
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

def get_3d_score_accum(accum_data: pd.DataFrame, df_left: pd.DataFrame, df_right: pd.DataFrame) -> pd.DataFrame:
    """
    Accumulate 3D scores from two DataFrames (representing left and right data) into a single DataFrame.
    
    Parameters:
    accum_data (pd.DataFrame): DataFrame where the 3D scores are to be accumulated.
    df_left (pd.DataFrame): DataFrame containing scores for the left.
    df_right (pd.DataFrame): DataFrame containing scores for the right.
    
    Returns:
    pd.DataFrame: Updated DataFrame containing the accumulated 3D scores.
    """
    
    # Renaming columns of the input DataFrames based on their indices
    df_right = rename_columns_based_on_index(df_right.copy())
    df_left = rename_columns_based_on_index(df_left.copy())
    
    # Concatenating df_left and df_right horizontally (columns)
    new_row = pd.concat([df_left, df_right], axis=1)
    
    # Adding the concatenated row to the accum_data DataFrame
    accum_data = pd.concat([accum_data, new_row], axis=0).reset_index(drop=True)
    
    # Returning the updated DataFrame containing the accumulated 3D scores
    return accum_data

def process_hemispheres_scores(psd:pd.DataFrame, freq_bands:list=['Delta', 'Theta', 'Alpha'])->pd.DataFrame:
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

def measure_randomness(psd:pd.DataFrame)->pd.DataFrame:
    
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



