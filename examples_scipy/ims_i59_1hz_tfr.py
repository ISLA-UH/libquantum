import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from libquantum.stockwell import tfr_array_stockwell, calculate_rms_sig_test
from libquantum.benchmark_signals import plot_synth_tfr, plot_synth


DIR_PATH = "/Users/mgarces/Documents/DATA_2022/Tonga/CEA"
MAKE_PICKLE = False

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = SECONDS_PER_HOUR*24
NUM_DAYS = 1
# NUM_DAYS = 7.5
# NUM_DAYS = 10
# NUM_SECONDS = int(NUM_DAYS*SECONDS_PER_DAY)
NUM_SECONDS = 2**12

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
    sig_days = (sig_epoch_s - sig_epoch_s[0])/SECONDS_PER_DAY

    # plt.subplot(211)
    # plt.plot(sig_wf)
    # plt.subplot(212)
    # plt.plot(sig_days, sig_wf)
    # plt.xlabel('Days')
    # plt.show()

    rms_sig_wf, rms_sig_time = calculate_rms_sig_test(sig_wf=sig_wf, sig_time=sig_days, points_per_seg=16)

    plot_synth(sig_wf=sig_wf, sig_time=sig_days,
               sig_rms_wf=rms_sig_wf, sig_rms_time=rms_sig_time)

    freqs = np.arange(0.001, 0.1, 0.001)
    fmin, fmax = freqs[[0, -1]]

    # Stockwell
    [st_power, frequency] = tfr_array_stockwell(data=sig_wf, sfreq=1, fmin=fmin, fmax=fmax, width=3)

    plot_synth_tfr(tfr_power=st_power, tfr_frequency=frequency, tfr_time=sig_days)
    plt.show()





