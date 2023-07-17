import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import torch as t
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis

def bandpass(input_, low_f, high_f, device=t.device('cuda')):
    """
    Parameters:
     - input:                   numpy
     - low_f:                   float, lowest frequency which will be allowed                         
     - high_f:                  float, highest frequency which will be allowed
    
    Returns: filtered input tensor

    """
    input_ = t.from_numpy(input_)
    pass1 = t.abs(t.fft.rfftfreq(input_.shape[-1],1/160)) > low_f
    pass2 = t.abs(t.fft.rfftfreq(input_.shape[-1],1/160)) < high_f
    fft_input = t.fft.rfft(input_)
    return t.fft.irfft(fft_input.to(device) * pass1.to(device) * pass2.to(device)).cpu().numpy()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #     y = lfilter(b, a, data)
    y = filtfilt(b,a, data,axis=0)

    return y

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq=0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data,axis=0)
    return y

def get_signal_changes(sign_change):
    abs_array = np.abs(np.diff(sign_change,axis=0))
    abs_array /= np.max(abs_array,axis=0)
    total_changes = abs_array.sum(axis=0)
    return total_changes


def notch_filter(data,freq, quality_factor, fs):
    b,a = iirnotch(freq,quality_factor,fs)
    y = filtfilt(b,a,data,axis=0)
    
    return y

def remove_noisy_component_with_ica(eeg):
    n_components=eeg.shape[-1]
    ica = FastICA(n_components=n_components)
    components = ica.fit_transform(eeg)
    
    # Method 1: Keep the component with the lowest kurtosis
    #remove_indices = [i for i in range(n_components) if i != np.argmin(kurtosis(components))]

    # Method 2: Get the indices where there is no change
    # factor=1*components.std(axis=0)
    # factor_1 = 0.1
    # clipped_signal = components-np.clip(components,a_min=-factor,a_max=factor)
    # upper_limit = np.mean(clipped_signal,axis=0)+factor_1*np.max(clipped_signal,axis=0)
    # lower_limit = np.mean(clipped_signal,axis=0)-factor_1*np.max(clipped_signal,axis=0)
    # index_with_less_perturbation = np.argmin((clipped_signal>upper_limit).sum(axis=0)+(clipped_signal<lower_limit).sum(axis=0))
    
    # Method 3:
    # factor=0.2
    # filtered_signal = butter_lowpass_filter(components,1,256,4)
    # limit = np.max(np.vstack([np.abs(np.max(filtered_signal,axis=0)),np.abs(np.min(filtered_signal,axis=0))]),axis=0)
    # signal_mean = np.mean(filtered_signal,axis=0)
    # clipped = filtered_signal-np.clip(filtered_signal,a_min=signal_mean-factor*limit,a_max=signal_mean+factor*limit)
    # index_with_less_perturbation = np.argmin((clipped!=0).sum(axis=0))

    # Method 4:
    factor=0.01
    filtered_signal = butter_lowpass_filter(components,0.5,256,4)
    limit = np.max(np.vstack([np.abs(np.max(filtered_signal,axis=0)),np.abs(np.min(filtered_signal,axis=0))]),axis=0)
    signal_mean = np.mean(filtered_signal,axis=0)
    sign_change_top = np.sign(signal_mean+factor*limit-filtered_signal)
    sign_change_bottom = np.sign(signal_mean-factor*limit-filtered_signal)
    bottom_changes = get_signal_changes(sign_change_bottom)
    top_changes = get_signal_changes(sign_change_top)
    index_with_less_perturbation = np.argmin(bottom_changes+top_changes)

    # Method 5
    # filtered_signal = butter_lowpass_filter(components,1,256,4)
    # indices_of_minimums = np.argmin(np.abs(filtered_signal),axis=1)
    # total_occurrences_minimum = [(indices_of_minimums==i).sum() for i in range(filtered_signal.shape[-1])]
    # ## first_index,second_index = np.argsort(total_occurrences_minimum)[::-1][:2]
    # ## incidences_of_minimum = [(np.argmin(np.abs(filtered_signal[:,[first_index,second_index]]),axis=1)==i).sum() for i in range(2)]
    # ## index_with_less_perturbation = [first_index,second_index][np.argmax(incidences_of_minimum)]
    # index_with_less_perturbation = np.argmax(total_occurrences_minimum)



    remove_indices = [i for i in range(n_components) if i != index_with_less_perturbation]
    components[:, remove_indices] = 0

    # Reconstruct the signal
    eeg_restored = ica.inverse_transform(components)
    
    return eeg_restored