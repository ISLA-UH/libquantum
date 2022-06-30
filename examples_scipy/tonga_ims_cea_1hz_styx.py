"""
Test Styx against instrument-corrected CEA - all stations
Crashed at Station ID: IM.I01H8..BDF
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libquantum.styx_stx import tfr_stx_fft
from libquantum.benchmark_signals import plot_tfr_bits

DIR_PATH = "/Users/mgarces/Documents/DATA_2022/Tonga/CEA"
INPUT_PICKLE_PATH = DIR_PATH
file_name_pickle = "ims_bdf_1hz_df" + ".pickle"
MAKE_PICKLE = False

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = SECONDS_PER_HOUR*24
EARTH_RADIUS_KM = 6371

# Set number of days to run
# NUM_DAYS = 7.5
# NUM_DAYS = 9
NUM_DAYS = 2
# TODO: Automatically build directory
OUTPUT_FIGURE_PATH = os.path.join(DIR_PATH, 'STYX_2D')
figure_format = "png"

NUM_SECONDS_EXACT = int(NUM_DAYS*SECONDS_PER_DAY)
NUM_SECONDS = NUM_SECONDS_EXACT

# STATION_ID = 'I59H1'
# OUTPUT_PICKLE_PATH = DIR_PATH
# file_name_pickle = "ims_cea_" + STATION_ID + "_df" + ".pickle"

order_nth = 6
frequency_stx_min_hz = 0.0001
frequency_stx_max_hz = 0.1

station_dq = ["IM.I11L4..BDF"]
compute_tfr = True

if __name__ == "__main__":
    # Load file, reset index
    df = pd.read_pickle(os.path.join(INPUT_PICKLE_PATH, file_name_pickle))
    print("Columns:", df.columns)

    # Extract waveform for specified station
    # df = df0[df0['station_id'].str.contains(STATION_ID)]
    # df.index[0] returns the integer index
    for idx in df.index:
        station_id_string = df['station_id'][idx]
        print("Station ID:", station_id_string)
        station_id_short = station_id_string[3:8]
        sig_wf = df['wf_raw_pa'][idx][0:NUM_SECONDS]
        sig_epoch_s = df['epoch_s'][idx][0:NUM_SECONDS]
        sig_sample_rate_hz = df['sample_rate_hz'][idx]
        sig_sample_interval_s = 1/sig_sample_rate_hz
        sig_range = df['range_km'][idx]
        sig_degrees_int = int(10*sig_range/EARTH_RADIUS_KM*180/np.pi)/10.

        fig_title = station_id_short + ", r = " + str(sig_degrees_int) + " degrees"

        if station_id_string in station_dq:
            continue
            # plt.figure()
            # plt.plot(sig_wf)
            # plt.title(station_id_string)
            # plt.show()

        # TODO: Correct for case of no data at beginning or end
        if np.any(np.isnan(sig_wf)):
            print("Zero filled NaNs in station ", station_id_string)
            sig_wf = np.nan_to_num(sig_wf, nan=0.0)

        if np.ma.is_masked(sig_wf):
            print("Zero-filled masks in station ", station_id_string)
            sig_wf = np.ma.filled(sig_wf, fill_value=0.0)

        sig_time_s = np.arange(sig_wf.shape[-1])*sig_sample_interval_s
        sig_time_hours = sig_time_s/SECONDS_PER_HOUR
        sig_time_days = sig_time_s/SECONDS_PER_DAY

        if compute_tfr:
            [tfr_stx, psd_stx, frequency, frequency_fft, W] = \
                tfr_stx_fft(sig_wf=sig_wf,
                            time_sample_interval=sig_sample_interval_s,
                            frequency_min=frequency_stx_min_hz,
                            frequency_max=frequency_stx_max_hz,
                            scale_order_input=order_nth,
                            is_geometric=True,
                            is_inferno=True)
            # print("Number SX frequencies:", frequency.shape)
            # print("Shape of W:", W.shape)
            # print("Shape of SX:", psd_stx.shape)

            # Show period in minutes
            period = 1/frequency/60
            # TODO: Fix plots, standardize units - go to libquantum plot templates
            fig = plot_tfr_bits(tfr_power=psd_stx, tfr_frequency=period, tfr_time=sig_time_hours,
                                bits_min=-12, y_scale='log', tfr_x_str="Hours from 2022-01-15 0Z",
                                tfr_y_str="Period, minutes", title_str=fig_title, tfr_y_flip=True)

            range_sta_id = 'R_' + str(int(sig_range)) + '_km_' + station_id_string
            figure_filename = os.path.join(OUTPUT_FIGURE_PATH, range_sta_id)
            plt.savefig(figure_filename + '.' + figure_format, format=figure_format)
            fig.clear()
            plt.close()

            # plt.show()




