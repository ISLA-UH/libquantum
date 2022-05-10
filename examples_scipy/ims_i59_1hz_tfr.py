"""
Test Styx against instrument-corrected CEA IS59

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libquantum.styx import tfr_stx_fft
from libquantum.benchmark_signals import plot_tfr_bits

DIR_PATH = "/Users/mgarces/Documents/DATA_2022/Tonga/CEA"
MAKE_PICKLE = False

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = SECONDS_PER_HOUR*24
# NUM_DAYS = 0.5
# NUM_DAYS = 7.5
NUM_DAYS = 9
NUM_SECONDS_EXACT = int(NUM_DAYS*SECONDS_PER_DAY)
NUM_SECONDS = NUM_SECONDS_EXACT

STATION_ID = 'I59H1'
OUTPUT_PICKLE_PATH = DIR_PATH
file_name_pickle = "ims_cea_" + STATION_ID + "_df" + ".pickle"
OUTPUT_FIGURE_PATH = os.path.join(DIR_PATH, 'STYX_9D')

order_nth = 6
frequency_stx_min_hz = 0.0001
frequency_stx_max_hz = 0.1

if __name__ == "__main__":
    # Load file, reset index
    df0 = pd.read_pickle(os.path.join(OUTPUT_PICKLE_PATH, file_name_pickle))
    print("Columns:", df0.columns)

    # Extract waveform for specified station
    df = df0[df0['station_id'].str.contains(STATION_ID)]
    # df.index[0] returns the integer index
    print("Index:", df.index[0])
    station_id_string = df['station_id'][df.index[0]]
    print("Station ID:", station_id_string)
    sig_wf = df['wf_raw_pa'][df.index[0]][0:NUM_SECONDS]
    sig_epoch_s = df['epoch_s'][df.index[0]][0:NUM_SECONDS]
    sig_sample_rate_hz = df['sample_rate_hz'][df.index[0]]
    sig_sample_interval_s = 1/sig_sample_rate_hz

    # TODO: Correct for case of no data at beginning or end
    if np.any(np.isnan(sig_wf)):
        print("Zero filled NaNs in station, : ", station_id_string)
        sig_wf = np.nan_to_num(sig_wf, nan=0.0)

    if np.ma.is_masked(sig_wf):
        print("Zero-filled masks in station, : ", station_id_string)
        sig_wf = np.ma.filled(sig_wf, fill_value=0.0)

    print(sig_wf.shape[-1])
    sig_time_s = np.arange(sig_wf.shape[-1])*sig_sample_interval_s
    sig_time_hours = sig_time_s/SECONDS_PER_HOUR
    sig_time_days = sig_time_s/SECONDS_PER_DAY

    [tfr_stx, psd_stx, frequency, frequency_fft, W] = \
        tfr_stx_fft(sig_wf=sig_wf,
                    time_sample_interval=sig_sample_interval_s,
                    frequency_min=frequency_stx_min_hz,
                    frequency_max=frequency_stx_max_hz,
                    scale_order_input=order_nth,
                    is_geometric=True,
                    is_inferno=True)
    print("Number SX frequencies:", frequency.shape)
    print("Shape of W:", W.shape)
    print("Shape of SX:", psd_stx.shape)

    # Show period in minutes
    period = 1/frequency/60
    # TODO: Fix plots, standardize units - go to libquantum plot templates
    fig = plot_tfr_bits(tfr_power=psd_stx, tfr_frequency=period, tfr_time=sig_time_days,
                  bits_min=-12, y_scale='log', tfr_x_str="Days from 2022-01-15 0Z",
                  tfr_y_str="Period, minutes", title_str=station_id_string, tfr_y_flip=True)
    figure_filename = os.path.join(OUTPUT_FIGURE_PATH, station_id_string)
    figure_format = "png"
    plt.savefig(figure_filename + '.' + figure_format, format=figure_format)
    fig.clear()
    plt.close()
    # plt.show()




