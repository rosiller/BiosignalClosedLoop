from bleak import BleakClient,BleakScanner
import logging
import asyncio

async def connect_ble(client, address, device_name, device_type):
     # Confirm the address
    address = await discover(device_name,address,device_type)

    if client is None:
        client = BleakClient(address)
        await client.connect()
        print("---------Device connected--------------", flush=True)
    return client,address

async def connect_and_run(device_name,address,device_type, client, run_func):
    
    # Confirm the address
    address = await discover(device_name,address,device_type)
    print(f'Attempting to connect to {address}')

    try:
        if client is None:
            client = BleakClient(address)
            print(f'Connecting to {address}')
            await client.connect()
            print("---------Device connected--------------", flush=True)

        await run_func(client)
    except Exception as e:
        logging.error("Exception in connect_and_run: %s", e)
    finally:
        if client.is_connected:
            await client.disconnect()

async def disconnect_ble(client):
    try:
        await client.disconnect()
        return not client.is_connected
    
    except Exception as e:
        print(e)
        return not client.is_connected
        
async def discover(device_name, address=None,device_type='GVS'):
    print("---------Looking for Device------------ ", flush=True)
    scanner = BleakScanner()
    devices = await scanner.discover(timeout=15)

    devices_list = []

    for d in devices:
        # print(d.name)
        if device_name in d.name:
            print(f'Found {d.name} with address: {d.address}')
            devices_list.append(d)

    if devices_list:
        device_address = devices_list[0].address
    else:
        print(f'No {device_type} devices found')
        device_address = None
        
    if address is not None and address != device_address:
        if device_address is not None:
            raise Exception(f'Address: {address} not found, found {device_type} with {device_address} instead')
        else:
            print(f"Couldn't discover {device_type} with address {address} attempting to connect directly")
            device_address = address
            
    return device_address