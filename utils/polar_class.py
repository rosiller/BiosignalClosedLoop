import asyncio

from pylsl import StreamInfo, StreamOutlet,StreamInlet,resolve_stream
from bleak.uuids import uuid16_dict
from math import ceil,floor
import nest_asyncio
import time
from .ble_utils import disconnect_ble,connect_ble
nest_asyncio.apply()

class PolarH10:
    uuid16_dict = {v: k for k, v in uuid16_dict.items()}

    ## UUID for Request of stream settings ##
    PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"

    ## UUID for Request of start stream ##
    PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

    ## UUID for model number ##
    MODEL_NBR_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(uuid16_dict.get("Model Number String"))
    
    ## UUID for manufacturer name ##
    MANUFACTURER_NAME_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(uuid16_dict.get("Manufacturer Name String"))

    ## UUID for battery level ##
    BATTERY_LEVEL_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(uuid16_dict.get("Battery Level"))

    ## For Polar H10  sampling frequency ##
    ECG_SAMPLING_FREQ = 130
    ACC_SAMPLING_FREQ = 200

    ## UUID for Request of ECG Stream ##
    ECG_WRITE = bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])
    ACC_WRITE = bytearray([0x02, 0x02, 0x00, 0x01, 0xC8, 0x00, 0x01, 0x01, 0x10, 0x00, 0x02, 0x01, 0x08, 0x00])

    def __init__(self,address=None,data_type=['ecg','acc']):
        self.device_name = 'Polar H10'
        self.address = None#'C4:57:25:A1:20:0E'
        self.thread = None
        self.client = None
        self.model_number = None
        self.manufacturer_name = None
        self.battery_level = None
        self.fs = 130 #Hz
        self.session_list = []
        self.OUTLET_ECG = self.prepareStreamOutlet('ECG')
        self.OUTLET_ACC = self.prepareStreamOutlet('ACC')
    
    async def connect(self):
        self.client, self.address = await connect_ble(self.client, self.address, self.device_name, self.device_name)
        if not self.model_number:
            self.model_number = await self.client.read_gatt_char(PolarH10.MODEL_NBR_UUID)
            print("Model Number: {0}".format("".join(map(chr, self.model_number))), flush=True)
        
        if not self.manufacturer_name:
            self.manufacturer_name = await self.client.read_gatt_char(PolarH10.MANUFACTURER_NAME_UUID)
            print("Manufacturer Name: {0}".format("".join(map(chr, self.manufacturer_name))), flush=True)

        if not self.battery_level:
            self.battery_level = await self.client.read_gatt_char(PolarH10.BATTERY_LEVEL_UUID)
            print("Battery Level: {0}%".format(int(self.battery_level[0])), flush=True)

    def data_conv(self,sender,data):

        if data[0] == 0x00:
            # print(".", end = '', flush=True)
            step = 3
            samples = data[10:]
            offset = 0
            while offset < len(samples):
                ecg = PolarH10.convert_array_to_signed_int(samples, offset, step)
                offset += step
                self.OUTLET_ECG.push_sample([ecg])

        elif data[0] == 0x02:

            timestamp = PolarH10.convert_to_unsigned_long(data, 1, 8)/1.0e9 # timestamp of the last sample
            frame_type = data[9]
            resolution = (frame_type + 1) * 8 # 16 bit
            time_step = 0.005 # 200 Hz sample rate
            step = ceil(resolution / 8.0)
            samples = data[10:] 
            n_samples = floor(len(samples)/(step*3))
            sample_timestamp = timestamp - (n_samples-1)*time_step
            offset = 0
            while offset < len(samples):
                x = PolarH10.convert_array_to_signed_int(samples, offset, step)
                offset += step
                y = PolarH10.convert_array_to_signed_int(samples, offset, step) 
                offset += step
                z = PolarH10.convert_array_to_signed_int(samples, offset, step) 
                offset += step
                sample_timestamp += time_step
                self.OUTLET_ACC.push_sample([x,y,z])
    
    async def run(self):

        self.stop_event = asyncio.Event(loop = asyncio.get_event_loop())

        await self.client.read_gatt_char(PolarH10.PMD_CONTROL)
        print("Collecting GATT data...", flush=True)

        await self.client.write_gatt_char(PolarH10.PMD_CONTROL, PolarH10.ECG_WRITE)
        print("Writing GATT data...", flush=True)
        await self.client.write_gatt_char(PolarH10.PMD_CONTROL, PolarH10.ACC_WRITE)

        await self.client.start_notify(PolarH10.PMD_DATA, lambda sender, data: self.data_conv(sender, data))
        
        print("Collecting ECG data...", flush=True)
        await self.stop_event.wait()  # wait here until stop_event is set

        # self.OUTLET_ECG = None
        # self.OUTLET_ACC = None
        
        print("Stopping ECG data...", flush=True)
        # print("[CLOSED] application closed.", flush=True)

        return 0
    
    def pause_stream(self):
        # Not really doing anything except closing the stream outlets and receiving the data, the device is still connected,
        #  and the thread and loops are still running 
        self.stop_event.set()
        self.stop_event.clear()

    def resume_stream(self):
        # After pausing the stream, can again start the stream outlets and start reading from device again
        _ = asyncio.run_coroutine_threadsafe(self.start_polar(), self.loop)

    async def stop(self):
        self.stop_event.set()

        await self.client.stop_notify(PolarH10.PMD_DATA)
        self.disconnected = await disconnect_ble(self.client)

        # Wait a bit for the coroutine to actually stop
        time.sleep(5)

    def prepareStreamOutlet(self,data_type = 'ECG'):
        data_type = data_type.upper()
        if data_type=='ECG':
            nb_channels = 1
            fs=PolarH10.ECG_SAMPLING_FREQ
            
        elif data_type=='ACC':
            nb_channels = 3
            fs = PolarH10.ACC_SAMPLING_FREQ
        else:
            raise Exception(f'Data type {data_type} not recognized')

        info = StreamInfo(self.device_name, data_type, nb_channels,fs, 'float32', 'myuid2424')

        info.desc().append_child_value("manufacturer", "Polar")
        channels = info.desc().append_child("channels")
        for c in ["ECG"]:
            channels.append_child("channel")\
                .append_child_value("name", c)\
                .append_child_value("unit", "microvolts")\
                .append_child_value("type", data_type)
        
        # next make an outlet; we set the transmission chunk size to 74 samples and
        # the outgoing buffer size to 360 seconds (max.)
        return StreamOutlet(info, 74, 360)
    
    @staticmethod
    def convert_array_to_signed_int(data, offset, length):
        return int.from_bytes(
            bytearray(data[offset : offset + length]), byteorder="little", signed=True,)
    
    @staticmethod
    def convert_to_unsigned_long(data, offset, length):
        return int.from_bytes(
            bytearray(data[offset : offset + length]), byteorder="little", signed=False,)
        




