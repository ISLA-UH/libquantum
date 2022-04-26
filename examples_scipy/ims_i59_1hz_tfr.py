import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from redpandas.redpd_plot.wiggles import plot_wiggles_pandas
from scipy.io.wavfile import write
from libquantum import utils, spectra
from scipy.signal import detrend


DIR_PATH = "/Users/mgarces/Documents/DATA_2022/Tonga/CEA"
MAKE_PICKLE = False

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = SECONDS_PER_HOUR*24
NUM_DAYS = 7.5
# NUM_DAYS = 10
NUM_SECONDS = int(NUM_DAYS*SECONDS_PER_DAY)

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

    plt.subplot(211)
    plt.plot(sig_wf)
    plt.subplot(212)
    plt.plot(sig_days, sig_wf)
    plt.xlabel('Days')
    plt.show()





