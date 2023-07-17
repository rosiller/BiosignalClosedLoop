
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

async def main(duration, window_length):
    """Main function for running the recording session.

    Args:
    duration (int): Duration of the recording in seconds.
    window_length (dict): Dictionary specifying the display window length for each data type.
    """
    await asyncio.gather(
        session.start_devices(),
        session.collect_data(duration, window_length)
    )

if __name__ == "__main__":
    # Logging setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)

    # Argument parsing
    parser = argparse.ArgumentParser(description="PolarH10_EEG")
    parser.add_argument('--duration', '-d', default=40, help="Duration of the recording in seconds")
    args = parser.parse_args()

    # Device and stimulator setup (consider moving this to a separate config file or script)
    device_list = [
                    # Muse2('Muse2_20230622_234543_EEG '), # For simulations, runs the file 
                    # PolarH10_recorder('Polarh10_20230625_165341_ECG  '), # For simulations
                #    Cyton('eeg'),  # For openbci
                # Muse2('eeg eeg_filt eeg_score'), #eeg_filt eeg_score
                PolarH10_recorder('ecg ecg_hr'), # Available: ecg, ecg_hr, acc, bmag, br, "hr"
                ]    
    
    stimulation_devices = [ 
                            # SoundStimulator(stim_type='speed'),  
                            # SoundStimulator(stim_type='volume')
                        ]    
    decision_model =  SimpleWait# SimpleInterpolator

    # RecordingSession setup
    session = RecordingSession(input_devices = device_list,               
                               stimulation_devices=stimulation_devices,  
                               decision_model = decision_model)

    # Window length setup
    window_length= {'eeg':10,'eeg_filt':5,'eeg_score':0,'ecg':10,'ecg_hr':0,'hr':20,'br':0}

    # Start devices and data collection
    asyncio.run(session.connect_all())
    try:
        asyncio.run(main(int(args.duration), window_length))
    except KeyboardInterrupt:
        logger.info("Interrupt received, stopping devices...")
        asyncio.run(session.stop_devices())

    # Post-session summary and data saving
    print(session)
    save_data = input("Would you like to save the data? (y/n): ").lower()
    if save_data == 'y':
        note = input("Add some notes?:")
        session.add_notes(note)
        session.save()
        print(f"Saved data with id: {session.id}")
    elif save_data == 'n':
        print("Data will not be saved.")
    else:
        print("Invalid input. Data will not be saved.")
