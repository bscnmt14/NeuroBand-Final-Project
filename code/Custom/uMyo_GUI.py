import tkinter as tk
from tkinter import ttk, filedialog
import threading
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import serial
from serial.tools import list_ports

# Import core components from your provided files
# These must exist in the same directory as this script.
import umyo_parser
from umyo_logger import uMyoDataLogger

# --- GLOBAL PLOTTING DATA AND CONFIGURATION ---
# Key: device_id (hex string), Value: list of EMG samples
PLOT_DATA = {}
# Define buffer size for plotting history
PLOT_BUFFER_SIZE = 500


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
        print(f"❌ Error: Input CSV file not found at {input_csv_filename}.")
    except KeyError:
        print(f"❌ Error: Column '{SENSOR_ID_COLUMN}' not found in the CSV data. Check the column headers.")
    except Exception as e:
        # This catches the 'No module named 'openpyxl'' error if not installed
        print(f"❌ An unexpected error occurred during export: {e}")

    finally:
        # Clean up the temporary CSV file
        if os.path.exists(input_csv_filename):
            os.remove(input_csv_filename)
            print(f"🧹 Cleaned up temporary file: {input_csv_filename}")


# --- 2. GUI APPLICATION CLASS ---

class uMyoApp:
    def __init__(self, master):
        self.master = master
        master.title("uMyo EMG Exporter GUI")

        # --- State Variables ---
        self.is_collecting = False
        self.serial_port = None
        self.logger = None
        self.collection_duration = tk.DoubleVar(value=10.0)
        self.selected_device_id = tk.StringVar(value="No Device")
        self.current_status = tk.StringVar(value="Status: Ready")

        # --- GUI Setup ---
        self.create_widgets()
        self.setup_plotting()

        # Start a thread to continuously update the plots (daemon=True means it closes with the main program)
        self.plot_update_thread = threading.Thread(target=self.update_plots_loop, daemon=True)
        self.plot_update_thread.start()

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Settings Frame ---
        settings_frame = ttk.LabelFrame(main_frame, text="Settings & Status", padding="10")
        settings_frame.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        # Collection Duration Input
        ttk.Label(settings_frame, text="Duration (s):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(settings_frame, textvariable=self.collection_duration, width=8).grid(row=0, column=1, sticky=tk.W)

        # Control Buttons
        self.start_button = ttk.Button(settings_frame, text="Start/Export", command=self.start_collection)
        self.start_button.grid(row=1, column=0, columnspan=2, pady=10)

        # Status
        self.status_label = ttk.Label(settings_frame, textvariable=self.current_status)
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W)

        # --- Device Selector Frame (Updates based on detected devices) ---
        device_frame = ttk.LabelFrame(main_frame, text="Device Data Viewer", padding="10")
        device_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(device_frame, text="Viewing Sensor ID:").grid(row=0, column=0, sticky=tk.W)
        self.device_dropdown = ttk.Combobox(device_frame, textvariable=self.selected_device_id, state="readonly",
                                            width=15)
        self.device_dropdown.grid(row=0, column=1, sticky=tk.W)
        self.device_dropdown.bind('<<ComboboxSelected>>', self.update_plots)

        # --- Plotting Frame (Matplotlib will be embedded here) ---
        self.plot_frame = ttk.LabelFrame(main_frame, text="Real-time EMG Plots (Last 500 Samples)", padding="10")
        self.plot_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

    def setup_plotting(self):
        # Set up Matplotlib figure with two subplots
        self.fig, (self.ax_time, self.ax_freq) = plt.subplots(1, 2, figsize=(9, 4))
        self.fig.tight_layout(pad=3.0)

        # Time Domain Plot Setup (Requirement: time graph)
        self.ax_time.set_title('Time-Domain EMG Signal')
        self.ax_time.set_xlabel('Sample Index')
        self.ax_time.set_ylabel('EMG Amplitude')
        self.ax_time.set_ylim(-5000, 5000)  # Adjust based on typical uMyo output range
        self.line_time, = self.ax_time.plot([], [], 'r-')

        # Frequency Domain Plot Setup (Requirement: frequency/magnitude graph)
        self.ax_freq.set_title('Frequency Spectrum (Bands 0-3)')
        self.ax_freq.set_xlabel('Frequency Band Index')
        self.ax_freq.set_ylabel('Magnitude')
        self.ax_freq.set_ylim(0, 5000)
        # Assuming the first 4 bands of device_spectr are most relevant
        self.bar_freq = self.ax_freq.bar(range(4), [0] * 4, color='b')
        self.ax_freq.set_xticks(range(4))

        # Embed Matplotlib into Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def update_plots(self, event=None):
        """Updates the plot data based on the currently selected device,
           including a computed FFT graph."""
        device_id_str = self.selected_device_id.get()
        if device_id_str == "No Device":
            return

        # --- Time Domain Update (Unchanged) ---
        emg_data = PLOT_DATA.get(device_id_str, [])
        self.line_time.set_xdata(np.arange(len(emg_data)))
        self.line_time.set_ydata(emg_data)
        self.ax_time.set_xlim(max(0, len(emg_data) - PLOT_BUFFER_SIZE), len(emg_data))

        # --- Frequency Domain Update: Calculating FFT ---

        # 1. Get enough data for FFT (must be power of 2, e.g., 256 or 512)
        fft_size = 256
        data_subset = emg_data[-fft_size:]

        if len(data_subset) == fft_size:
            # 2. Apply FFT
            # The FFT returns complex numbers; np.abs() gets the magnitude/power.
            # We use rfft for real input, which is faster.
            fft_result = np.fft.rfft(data_subset)

            # 3. Get magnitudes (power spectrum)
            magnitude = np.abs(fft_result)

            # 4. Generate frequency axis (assuming a sample rate, e.g., 1000 Hz)
            # NOTE: You may need to adjust the sample rate (fs) based on your uMyo setup!
            fs = 1000  # Assume 1000 Hz sampling rate (typical for raw EMG)
            freqs = np.fft.rfftfreq(fft_size, 1.0 / fs)

            # 5. Plot the result (using the first line object for the frequency plot)
            # If you want to keep the bar plot logic, you would need to define a new line object
            # self.line_freq, = self.ax_freq.plot([], [], 'b-') in setup_plotting

            # Clear bar plot items if switching to line plot
            for bar in self.bar_freq:
                bar.set_height(0)

            if not hasattr(self, 'line_freq'):
                self.line_freq, = self.ax_freq.plot(freqs, magnitude, 'b-')
            else:
                self.line_freq.set_xdata(freqs)
                self.line_freq.set_ydata(magnitude)

            # Adjust frequency plot limits
            self.ax_freq.set_xlim(0, fs / 2)  # Max frequency is half the sample rate (Nyquist)
            self.ax_freq.set_ylim(0, np.max(magnitude) * 1.1)
            self.ax_freq.set_title('Computed FFT Magnitude (Line Graph)')
            self.ax_freq.set_xlabel('Frequency (Hz)')

        else:
            # Not enough data yet to run FFT
            self.ax_freq.set_title('Frequency Spectrum (Needs 256 samples)')

        self.canvas.draw_idle()

    def update_plots_loop(self):
        """Runs in a separate thread to ensure plots refresh continuously."""
        while True:
            # Only update if data is being collected or a device is selected
            if self.is_collecting or self.selected_device_id.get() != "No Device":
                self.master.after(0, self.update_plots)  # Use master.after to safely update GUI from thread
            time.sleep(0.1)  # Refresh rate

    def start_collection(self):
        if self.is_collecting:
            # This should ideally stop the thread immediately but we use the loop check
            self.stop_collection()
        else:
            self.current_status.set("Status: Connecting & Collecting Data...")
            self.start_button.config(text="STOP", state=tk.DISABLED)
            self.is_collecting = True

            # Start the data collection in a separate thread
            self.data_thread = threading.Thread(target=self.data_collection_thread, daemon=True)
            self.data_thread.start()

    def stop_collection(self):
        self.is_collecting = False
        # The data_collection_thread will exit its loop and proceed to export

    def data_collection_thread(self):
        temp_filename = f"temp_umyo_data_{int(time.time())}.csv"
        duration = self.collection_duration.get()

        # --- Serial Setup (Same logic as previous versions) ---
        ports = list(list_ports.comports())
        device = next((p.device for p in ports if "usbserial" in p.device), None)
        if not device:
            device = ports[-1].device if ports else None

        if not device:
            self.current_status.set("Status: ERROR - No serial device found!")
            self.master.after(100, lambda: self.start_button.config(state=tk.NORMAL, text="Start/Export"))
            self.is_collecting = False
            return

        ser = None
        try:
            # Ensure serial port is not blocked before opening it for non-blocking read
            temp_ser = serial.Serial(device, timeout=1)
            temp_ser.close()
            time.sleep(0.5)

            ser = serial.Serial(port=device, baudrate=921600, timeout=0)
            self.logger = uMyoDataLogger(filename=temp_filename)
            self.current_status.set(f"Status: Collecting Data for {duration}s from {device}...")

            # --- Data Collection Loop ---
            start_time = time.time()
            device_ids_set = set()

            while self.is_collecting and (time.time() - start_time) < duration:
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting)
                    umyo_parser.umyo_parse_preprocessor(data)

                    devices = umyo_parser.umyo_get_list()
                    for d in devices:
                        d_id_hex = f"{d.unit_id:08X}"
                        device_ids_set.add(d_id_hex)

                        # Update global PLOT_DATA for visualization
                        for sample in d.data_array[:d.data_count]:
                            PLOT_DATA.setdefault(d_id_hex, []).append(sample)
                            PLOT_DATA[d_id_hex] = PLOT_DATA[d_id_hex][-PLOT_BUFFER_SIZE:]

                    self.logger.log_all_devices()

                    # Update device dropdown if new sensors are found
                    if len(device_ids_set) > len(self.device_dropdown['values']):
                        # Use self.master.after to safely update GUI widgets
                        self.master.after(0, lambda: self.update_device_dropdown(list(device_ids_set)))

                time.sleep(0.005)  # Loop faster to catch data

            # --- Finalization and Export ---
            self.current_status.set("Status: Data collection finished. Exporting to Excel...")
            self.logger.close()

            self.export_sorted_emg_to_excel(temp_filename)
            self.current_status.set("Status: Export Complete! Check output folder.")

        except Exception as e:
            self.current_status.set(f"Status: ERROR - {e}")
        finally:
            if ser and ser.is_open:
                ser.close()
            self.is_collecting = False
            self.master.after(100, lambda: self.start_button.config(state=tk.NORMAL, text="Start/Export"))

    def update_device_dropdown(self, device_ids):
        """Updates the dropdown list with discovered device IDs."""
        current_selection = self.selected_device_id.get()
        self.device_dropdown['values'] = device_ids

        if device_ids and (current_selection not in device_ids or current_selection == "No Device"):
            # Set selection to the first device found or keep current if it's still present
            self.selected_device_id.set(device_ids[0])
            self.update_plots()


# --- 3. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    root = tk.Tk()
    app = uMyoApp(root)


    # Gracefully handle window closing
    def on_closing():
        app.is_collecting = False
        if app.plot_update_thread.is_alive():
            app.plot_update_thread.join(0.2)
        root.destroy()


    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()