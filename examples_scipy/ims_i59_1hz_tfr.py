import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from libquantum.stockwell import tfr_array_stockwell_isla
from libquantum.styx import tfr_stx_fft
from libquantum.benchmark_signals import plot_tdr_rms, plot_tfr_lin, plot_tfr_bits


DIR_PATH = "/Users/mgarces/Documents/DATA_2022/Tonga/CEA"
MAKE_PICKLE = False

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = SECONDS_PER_HOUR*24
NUM_DAYS = 1
# NUM_DAYS = 7.5
# NUM_DAYS = 10
NUM_SECONDS_EXACT = int(NUM_DAYS*SECONDS_PER_DAY)
NUM_SECONDS = NUM_SECONDS_EXACT

STATION_ID = 'I59H1'
OUTPUT_PICKLE_PATH = DIR_PATH
file_name_pickle = "ims_cea_" + STATION_ID + "_df" + ".pickle"


if __name__ == "__main__":
    # Load file, reset index
    df0 = pd.read_pickle(os.path.join(OUTPUT_PICKLE_PATH, file_name_pickle))
    print("Columns:", df0.columns)

    # Extract waveform for specified station
    df = df0[df0['station_id'].str.contains(STATION_ID)]
    # df.index[0] returns the integer index
    print("Index:", df.index[0])
    sig_wf = df['wf_raw_pa'][df.index[0]][0:NUM_SECONDS]
    sig_epoch_s = df['epoch_s'][df.index[0]][0:NUM_SECONDS]
    sig_time_hours = (sig_epoch_s - sig_epoch_s[0])/SECONDS_PER_HOUR
    sig_time_days = (sig_epoch_s - sig_epoch_s[0])/SECONDS_PER_DAY
    #
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(sig_wf)
    # plt.subplot(212)
    # plt.plot(sig_time_hours, sig_wf)
    # plt.xlabel('Days')
    # plt.show()

    # rms_sig_wf, rms_sig_time = calculate_rms_sig_test(sig_wf=sig_wf, sig_time=sig_days, points_per_seg=16)

    fmin = 0.0001
    fmax = 0.1

    [tfr_stx, psd_stx, frequency, frequency_fft, W] = \
        tfr_stx_fft(sig_wf=sig_wf,
                    time_sample_interval=1,
                    frequency_min=fmin,
                    frequency_max=fmax,
                    nth_order=6,
                    is_geometric=True,
                    is_inferno=True)
    print("Number SX frequencies:", frequency.shape)
    print("Shape of W:", W.shape)
    print("Shape of SX:", psd_stx.shape)

    # TODO: Fix plots, standardize units - go to libquantum plot templates
    plot_tfr_bits(tfr_power=psd_stx, tfr_frequency=frequency, tfr_time=sig_time_hours,
                  y_scale='log', signal_time_base="hours")
    plt.show()





