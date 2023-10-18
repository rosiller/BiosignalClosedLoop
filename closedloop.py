
from utils.recordingsession_class import RecordingSession
from utils.recordingdevice_class import Muse2, Cyton, PolarH10_recorder
# Ignore some warnings
import logging
logging.getLogger('root').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import argparse
from utils.stimulator_class import SoundStimulator
from utils.decision_models_class import SimpleInterpolator,SimpleWait
import asyncio

# warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

async def main(duration):
    """Main function for running the recording session.

    Args:
    duration (int): Duration of the recording in seconds.
    """
    await asyncio.gather(
        session.start_devices(),
        session.collect_data(duration)
    )

if __name__ == "__main__":
    # Logging setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)

    # Argument parsing
    parser = argparse.ArgumentParser(description="PolarH10_EEG")
    parser.add_argument('--duration', '-d', default=50, help="Duration of the recording in seconds")
    args = parser.parse_args()

    # Device and stimulator setup (consider moving this to a separate config file or script)
    device_list = [
                    # Muse2('Muse2_20230622_234543_EEG '), # For simulations, runs the file 
                    # PolarH10_recorder('Polarh10_20230625_165341_ECG  '), # For simulations
                #    Cyton('eeg'),  # For openbci
                # Muse2('eeg eeg_filt eeg_score acc gyro'), #eeg_filt eeg_score
                # Muse2('eeg eeg_filt eeg_score eeg_randomness eeg_time_correlation eeg_frequency_correlation'), #eeg_filt eeg_score
                Muse2('eeg eeg_filt eeg_randomness eeg_time_correlation eeg_3d_score'),
                PolarH10_recorder('ecg ecg_hr acc bmag br'), # Available: ecg, ecg_hr, acc, bmag, br, "hr"
                ]    
    
    stimulation_devices = [ 
                            # SoundStimulator(stim_type='speed'),  
                            # SoundStimulator(stim_type='volume')
                        ]    
    decision_model =  SimpleWait# SimpleInterpolator


    # Window length setup
    # window_length = {'eeg':10,'eeg_filt':5,'eeg_score':0,'ecg':10,'ecg_hr':50,'hr':50,'br':10}
    window_length = {'eeg_filt':5,'ecg':10,'ecg_hr':10,'acc':10,'gyro':10,
                     'eeg_randomness':50,'eeg_score':30,'eeg_time_correlation':50,
                     'eeg_frequency_correlation':30,'eeg_3d_score':100}
    # window_length = {} # No plotting 

    # RecordingSession setup
    session = RecordingSession(input_devices = device_list,               
                               stimulation_devices=stimulation_devices,  
                               decision_model = decision_model,
                               window_length=window_length)

    # Start devices and data collection
    asyncio.run(session.connect_all())
    try:
        asyncio.run(main(int(args.duration)))
    except KeyboardInterrupt:
        logger.info("Interrupt received, stopping devices...")
        asyncio.run(session.stop_devices())
        asyncio.sleep(4)
        session.get_durations()

   # Post-session summary and data saving
    print(session)

    while True:
        save_data = input("Would you like to save the data? (y/n): ").lower()
        if save_data == 'y':
            note = input("Add some notes?:")
            session.add_notes(note)
            session.save()
            print(f"Saved data with id: {session.id}")
            break  # exit the loop once valid input is received and data is saved
        elif save_data == 'n':
            print("Data will not be saved.")
            break  # exit the loop once valid input is received
        else:
            print("Invalid input. Please enter 'y' or 'n'.")  # prompt the user for a valid input and continue the loop

