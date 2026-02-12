import pandas as pd
import os
import time
from datetime import datetime
import serial
from serial.tools import list_ports

# Import core components from your provided files
from umyo_logger import uMyoDataLogger
import umyo_parser


# --- 1. EXPORT FUNCTION (MUST BE DEFINED FIRST) ---

def export_sorted_emg_to_excel(input_csv_filename):
    """
    1. Reads data from the generated CSV.
    2. Sorts the data by 'device_id'.
    3. Adds a File_Timestamp column.
    4. Exports the final DataFrame to an Excel file.
    """
    output_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_excel_filename = f"EMG_Export_{output_timestamp}.xlsx"
    SENSOR_ID_COLUMN = 'device_id'

    try:
        print(f"Reading data from: {input_csv_filename}")

        # 1. Read the data into a Pandas DataFrame
        df = pd.read_csv(input_csv_filename)

        # --- Fulfilling Requirements ---

        # 3. Gives a timestamp for the data (File Processing Timestamp)
        df['File_Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 2. Sorts the data according to the uMyo sensor from which it came
        print(f"Sorting data by: '{SENSOR_ID_COLUMN}'")
        df.sort_values(by=SENSOR_ID_COLUMN, inplace=True)

        # 1. exports EMG data to an excel file
        print(f"Exporting sorted data to: {output_excel_filename}")
        df.to_excel(output_excel_filename, index=False, sheet_name='uMyo_EMG_Data')

        print("\n✅ **Success!**")
        print(f"Final Excel file created: **{output_excel_filename}**")

    except FileNotFoundError:
        print(f"❌ Error: Input CSV file not found at {input_csv_filename}. Did the logger run successfully?")
    except KeyError:
        print(f"❌ Error: Column '{SENSOR_ID_COLUMN}' not found in the CSV data. Check the column headers.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during export: {e}")

    finally:
        # Clean up the temporary CSV file
        if os.path.exists(input_csv_filename):
            os.remove(input_csv_filename)
            print(f"🧹 Cleaned up temporary file: {input_csv_filename}")


# --- 2. MAIN CONTROL FUNCTION ---

def run_data_collection_and_export(collection_duration_seconds=5):
    """
    Main function to orchestrate data collection and export.
    """
    temp_filename = f"temp_umyo_data_{int(time.time())}.csv"

    print(f"Starting uMyo data collection for {collection_duration_seconds} seconds...")

    # --- Serial Setup ---
    ports = list(list_ports.comports())
    device = None

    for p in ports:
        # Prioritize devices with "usbserial" in the name, common for uMyo adapters
        if "usbserial" in p.device:
            device = p.device
            break
        # Fallback to the last detected port if no usbserial is found
        device = p.device

    if not device:
        print("No serial device found!")
        return

    # Standard serial setup using parameters from your serial_test.py
    temp_ser = serial.Serial(device, timeout=1)
    temp_ser.close()
    time.sleep(0.5)

    ser = serial.Serial(
        port=device,
        baudrate=921600,
        parity=serial.PARITY_NONE,
        stopbits=1,
        bytesize=8,
        timeout=0,  # Non-blocking read for real-time operation
    )

    # Initialize data logger
    logger = uMyoDataLogger(filename=temp_filename)
    print(f"Logging temporary data to: {logger.filename}")

    # --- Data Collection Loop ---
    try:
        start_time = time.time()
        while (time.time() - start_time) < collection_duration_seconds:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)

                # Parse raw data and update device objects
                umyo_parser.umyo_parse_preprocessor(data)

                # Log data from all active device objects to CSV
                logger.log_all_devices()

            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    except KeyboardInterrupt:
        print("\nData collection interrupted.")
    finally:
        logger.close()
        ser.close()
        print("Data collection finished. Starting export process...")

    # --- Export Step (Function Call is now valid) ---
    export_sorted_emg_to_excel(temp_filename)


# --- 3. SCRIPT ENTRY POINT ---

if __name__ == "__main__":
    # You can adjust the collection time here (in seconds)
    COLLECTION_TIME = 10

    try:
        run_data_collection_and_export(COLLECTION_TIME)
    except ImportError as e:
        print(
            f"❌ Critical Error: Could not import a required module. Please ensure all your files are in the same directory and all dependencies (pandas, openpyxl, pyserial) are installed.")
        print(f"Missing module: {e}")