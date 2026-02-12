# test_umyo.py - Verify your setup in 30 seconds
import umyo_parser
import serial
from serial.tools import list_ports
import time
#import display_stuff
# Connect to uMyo sensor
ports = list(list_ports.comports())
ser = serial.Serial(ports[-1].device, 921600, timeout=0)
print("🔍 Searching for uMyo sensors...")
for _ in range(100): # 10 seconds at ~10Hz
    if ser.in_waiting > 0:
        data = ser.read(ser.in_waiting)
        umyo_parser.umyo_parse_preprocessor(data)

        devices = umyo_parser.umyo_get_list()
        if devices:
            device = devices[0]
            print(f"✅ Found sensor {device.unit_id:08X}")
            print(f"🔋 Battery: {device.batt}mV | 📶 RSSI: {device.rssi}")
            print(f"💪 EMG: {device.data_array[-1]} | 🌐 Orientation: {device.Qsg}")
            break
    time.sleep(0.1)