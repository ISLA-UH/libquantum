"""
Compute Stockwell TFR as in Moukadem et al. 2022
First test synthetic: synth_00
"""

import numpy as np
from matplotlib import pyplot as plt
from libquantum.stockwell_orig import tfr_array_stockwell, calculate_rms_sig_test
from libquantum2.benchmark_signals import synth_01
from libquantum2.benchmark_signals import plot_tdr_rms, plot_tfr_lin, plot_tfr_bits

print(__doc__)


def main(sample_rate: float,
         frequency_center_min: float,
         frequency_center_max: float,
         order: float = 3.,
         signal_time_base: str = 'seconds'):
    """
    Evaluate synthetics
    :param sample_rate:
    :param frequency_center_min:
    :param frequency_center_max:
    :param order:
    :param signal_time_base:
    :return:
    """
    sample_interval = 1/sample_rate
    sig_in, time_in = synth_01(time_sample_interval=sample_interval, time_duration=1.024)
    print("Sig n:", sig_in.shape)

    # Compute strided RMS
    # TODO: Fix hop time and add offset on time from start
    rms_sig_wf, rms_sig_time = calculate_rms_sig_test(sig_wf=sig_in, sig_time=time_in, points_per_seg=16)

    # Stockwell transform
    [st_power, frequency, W] = tfr_array_stockwell(data=sig_in, sfreq=sample_rate,
                                                fmin=frequency_center_min,
                                                fmax=frequency_center_max, width=3.0)

    # TODO: Figure this out!
    print("Shape of W:", W.shape)
    plt.plot(np.abs(W))
    plt.show()

    # exit()

    plot_tdr_rms(sig_wf=sig_in, sig_time=time_in,
                 sig_rms_wf=rms_sig_wf, sig_rms_time=rms_sig_time)

    plot_tfr_lin(tfr_power=st_power, tfr_frequency=frequency, tfr_time=time_in)
    plot_tfr_bits(tfr_power=st_power, tfr_frequency=frequency, tfr_time=time_in)

    plt.show()


if __name__ == "__main__":
    # Specify synthetic
    time_base: str = 'seconds'
    sample_rate_hz: float = 1000.0  # Sample frequency
    frequency_center_min_hz: float = 5.
    frequency_center_max_hz: float = 100.
    order_nominal = 3.

    main(sample_rate=sample_rate_hz,
         frequency_center_min=frequency_center_min_hz,
         frequency_center_max=frequency_center_max_hz,
         order=order_nominal)
