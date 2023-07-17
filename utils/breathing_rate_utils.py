import numpy as np
from scipy.signal import butter, filtfilt,find_peaks

# Code taken from https://github.com/kbre93/dont-hold-your-breath

def calculate_breathing_rate(breathing_signal,acc_times):
        """
        This function calculates the breathing rate from the breathing signal.

        The breathing rate is calculated by finding the peaks in the breathing signal,
        filtering the peaks based on a peak-to-trough difference threshold, and then calculating
        the time difference between valid peaks.
        """
        breath_peaks = []

        # Threshold for peak-to-trough difference
        peak_threshold = 0.01  #0.04

        # Negative of the breathing signal to find the low acceleration points (mid-inhale and mid-exhale)
        breathing_peak_signal = -breathing_signal

        # Find all peak indices in the breathing signal
        breath_peaks_all, _ = find_peaks(breathing_peak_signal)

        # Iterate through each peak index
        for i in range(len(breath_peaks_all)):
            # Get the value of the current peak
            peak_val = breathing_peak_signal[breath_peaks_all[i]]

            if i == 0:
                # If it's the first peak, add it to the list of valid peaks
                breath_peaks.append(breath_peaks_all[i])
            else:
                # For the rest of the peaks, search for the preceding trough between the current peak and the previous peak
                start_idx = breath_peaks_all[i-1]
                end_idx = breath_peaks_all[i]
                trough_idx = np.argmin(breathing_peak_signal[start_idx:end_idx]) + start_idx
                trough_val = breathing_peak_signal[trough_idx]

                # If the difference between the current peak and the preceding trough meets the threshold,
                # add the current peak to the list of valid peaks
                if peak_val - trough_val >= peak_threshold:
                    breath_peaks.append(breath_peaks_all[i])

        # Calculate breathing rate from valid peaks
        br_values = 60/(np.diff(acc_times[breath_peaks])*2)
        br_times = acc_times[breath_peaks[1:]]

        # Smooth the calculated breathing rates using a moving average filter
        window_size = 3
        if len(br_values)>0:
            br_values_smooth = np.convolve(br_values, np.ones(window_size)/window_size, mode='same')
        else:
            br_values_smooth = np.zeros(1)

        return br_values_smooth

def calculate_breathing_signal(acc_values):
    
        # Gravity Filter
        cutoff_freq = 0.04  # Hz
        filter_order = 2
        nyquist_freq = 0.5 * 200 # PolarH10.ACC_SAMPLING_FREQ
        cutoff_norm = cutoff_freq / nyquist_freq
        b, a = butter(filter_order, cutoff_norm, btype='low')
        acc_low_pass = np.zeros_like(acc_values)
        for i in range(3):
            acc_low_pass[:, i] = filtfilt(b, a, acc_values[:, i])
        
        acc_low_pass_norm = np.linalg.norm(acc_low_pass, axis=1)
        acc_values_filt = acc_values - acc_low_pass
        acc_values_filt_norm = np.linalg.norm(acc_values_filt, axis=1)

        # Noise Filter
        nyquist_freq = 0.5 * 200 #PolarH10.ACC_SAMPLING_FREQ  
        cutoff_freq = 0.5  
        filter_order = 2  
        b, a = butter(filter_order, cutoff_freq / nyquist_freq, btype='low')
        breathing_signal = filtfilt(b, a, acc_values_filt_norm)

        return breathing_signal