import display_stuff

 # Initialize the plotting system
display_stuff.plot_init()

 # Add EMG data for device 0
display_stuff.plot_append_emg(device_id=0, emg_value=1024.5)

 # Add accelerometer data
display_stuff.plot_append_acc(device_id=0, ax=0.2, ay=-0.1, az=0.9)

 # Update the display
display_stuff.draw_all()