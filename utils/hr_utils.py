import neurokit2 as nk

def get_hr(ecg,fs):
    # Process the ECG data
    # TODO: It seems this function is generating numpy mean errors due to empty slices
    processed_data, info = nk.ecg_process(ecg, sampling_rate=fs)
    
    # Get the R-peaks
    rpeaks = info['ECG_R_Peaks']
    
    # Get heart rate
    hr = nk.ecg_rate(rpeaks, sampling_rate=fs, desired_length=len(ecg))
    return hr 

def clean_ecg(ecg,fs):
    ecg_signals, info = nk.ecg_process(ecg, sampling_rate=fs)
    ecg_cleaned = ecg_signals['ECG_Clean']
    return ecg_cleaned

def segment_ecg(ecg,fs):
    segmented_hb = nk.ecg_segment(ecg,
                                    sampling_rate=fs,
                                    show=True)
    return segmented_hb