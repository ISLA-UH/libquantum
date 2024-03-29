"""
Compute Stockwell TFR as in Moukadem et al. 2022
First test synthetic: synth_00
"""

import numpy as np
from libquantum.stockwell_orig import tfr_array_stockwell, calculate_rms_sig_test
from libquantum2.benchmark_signals import synth_02
from examples_scipy.stockwell_synth_03 import plot_synth, plot_synth_tfr

from matplotlib import pyplot as plt
print(__doc__)


def main(sample_rate, signal_time_base:str='seconds'):
    """
    Evaluate synthetics
    :param sample_rate:
    :param signal_time_base:
    :return:
    """
    sample_interval = 1/sample_rate
    sig_in, time_in = synth_02(time_sample_interval=sample_interval, time_duration=1.024)
    print("Sig n:", sig_in.shape)

    # Compute strided RMS
    # TODO: Fix hop time and add offset on time from start
    rms_sig_wf, rms_sig_time = calculate_rms_sig_test(sig_wf=sig_in, sig_time=time_in, points_per_seg=16)

    plot_synth(sig_wf=sig_in, sig_time=time_in,
               sig_rms_wf=rms_sig_wf, sig_rms_time=rms_sig_time)


    freqs = np.arange(1., 60., 1)
    fmin, fmax = freqs[[0, -1]]

    # Stockwell
    [st_power, frequency, _] = tfr_array_stockwell(data=sig_in, sfreq=sample_rate, fmin=fmin, fmax=fmax, width=1.0)

    plot_synth_tfr(tfr_power=st_power, tfr_frequency=frequency, tfr_time=time_in)
    plt.show()


if __name__ == "__main__":
    sample_rate_hz: float = 1000.0  # Sample frequency
    main(sample_rate=sample_rate_hz)
