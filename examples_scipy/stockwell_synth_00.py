"""
Compute Stockwell TFR as in Moukadem et al. 2022
First test synthetic: synth_00
"""

import numpy as np
from libquantum.stockwell import tfr_array_stockwell, calculate_rms_sig_test
from libquantum.benchmark_signals import synth_00, synth_01, synth_02, synth_03

from matplotlib import pyplot as plt
print(__doc__)


def plot_synth(sig_wf, sig_time,
                    sig_rms_wf, sig_rms_time,
                    signal_time_base:str='seconds'):
    """
    Waveform
    :param sig_wf:
    :param sig_time:
    :param sig_rms_wf:
    :param sig_rms_time:
    :param signal_time_base:
    :return:
    """

    plt.figure()
    plt.plot(sig_time, sig_wf)
    plt.plot(sig_rms_time, sig_rms_wf)
    plt.title('Input waveform and RMS')
    plt.xlabel("Time, " + signal_time_base)


def plot_synth_tfr(tfr_power, tfr_frequency, tfr_time,
                   title_str: str='TFR',
                   signal_time_base:str='seconds'):
    """
    TFR
    :param sig_tfr:
    :param sig_tfr_frequency:
    :param sig_tfr_time:
    :param signal_time_base:
    :return:
    """

    plt.figure()
    plt.pcolormesh(tfr_time, tfr_frequency, tfr_power, cmap='RdBu_r')
    plt.title(title_str)
    plt.ylabel("Frequency, samples per " + signal_time_base)
    plt.xlabel("Time, " + signal_time_base)


def main(sample_rate, signal_time_base:str='seconds'):
    """
    Evaluate synthetics
    :param sample_rate:
    :param signal_time_base:
    :return:
    """
    sample_interval = 1/sample_rate
    sig_in, time_in = synth_00(time_sample_interval=sample_interval, time_duration=1.024)
    print("Sig n:", sig_in.shape)

    # Compute strided RMS
    # TODO: Fix hop time and add offset on time from start
    rms_sig_wf, rms_sig_time = calculate_rms_sig_test(sig_wf=sig_in, sig_time=time_in, points_per_seg=16)

    plot_synth(sig_wf=sig_in, sig_time=time_in,
               sig_rms_wf=rms_sig_wf, sig_rms_time=rms_sig_time)


    freqs = np.arange(5., 500., 2.)
    fmin, fmax = freqs[[0, -1]]

    # Stockwell
    [st_power, frequency] = tfr_array_stockwell(data=sig_in, sfreq=sample_rate, fmin=fmin, fmax=fmax, width=3.0)

    plot_synth_tfr(tfr_power=st_power, tfr_frequency=frequency, tfr_time=time_in)
    plt.show()


if __name__ == "__main__":
    sample_rate_hz: float = 1000.0  # Sample frequency
    main(sample_rate=sample_rate_hz)
