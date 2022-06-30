import os
import numpy as np
import matplotlib.pyplot as plt
from libquantum.spectra import butter_bandpass
from libquantum.styx_stx import tfr_stx_fft
from libquantum.benchmark_signals import plot_tfr_bits

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = SECONDS_PER_HOUR*24

input_dir = "/Users/mgarces/Documents/DATA_2022/Tonga/RDVX/META/redvoxcore_main/kona_stacks"
file_name_stack = "stack_17days.npy"

sig_sample_interval_s = 60
sample_rate_sps = 1. / sig_sample_interval_s
cutoff_low_hours = 7
cutoff_low_s = SECONDS_PER_HOUR*cutoff_low_hours
cutoff_high_s = 2 * sig_sample_interval_s  # Nyquist

order_nth = 6

period_stx_max_hours = 3*24
period_stx_max_s = SECONDS_PER_HOUR*period_stx_max_hours
frequency_stx_min_hz = 1/period_stx_max_s
# frequency_stx_min_hz = 0.0001
frequency_stx_max_hz = sample_rate_sps/2.

compute_tfr = True

if __name__ == "__main__":
    """
    Bandpass and process demeaned stack
    :return:
    """

    sig_wf = np.load(os.path.join(input_dir, file_name_stack))
    sig_wf_bp = butter_bandpass(sig_wf=sig_wf, sample_rate_hz=sample_rate_sps,
                                frequency_cut_low_hz=1/cutoff_low_s,
                                frequency_cut_high_hz=1/cutoff_high_s,
                                tukey_alpha=0.1)

    sig_time_s = np.arange(sig_wf.shape[-1])*sig_sample_interval_s
    sig_time_min = sig_time_s/SECONDS_PER_MINUTE
    sig_time_hours = sig_time_s/SECONDS_PER_HOUR
    sig_time_days = sig_time_s/SECONDS_PER_DAY

    plt.figure()
    plt.subplot(211)
    plt.plot(sig_time_days, sig_wf)
    plt.grid(True)
    plt.title('Kona RedVox Barometer Stack')
    plt.subplot(212)
    plt.plot(sig_time_days, sig_wf_bp)
    plt.xlabel('Days from 2022-01-14')
    plt.grid(True)
    # plt.show()

    if compute_tfr:
        [tfr_stx, psd_stx, frequency, frequency_fft, W] = \
            tfr_stx_fft(sig_wf=sig_wf,
                        time_sample_interval=sig_sample_interval_s,
                        frequency_min=frequency_stx_min_hz,
                        frequency_max=frequency_stx_max_hz,
                        scale_order_input=order_nth,
                        is_geometric=True,
                        is_inferno=True)

        [tfr_stx2, psd_stx2, frequency2, frequency_fft2, W2] = \
            tfr_stx_fft(sig_wf=sig_wf,
                        time_sample_interval=sig_sample_interval_s,
                        frequency_min=1/cutoff_low_s,
                        frequency_max=frequency_stx_max_hz,
                        scale_order_input=order_nth,
                        is_geometric=True,
                        is_inferno=True)

        # Show period in minutes
        period_min = 1/frequency/60
        period_hours = period_min/60
        frequency_cycles_per_day = frequency*SECONDS_PER_DAY

        fig_title = "Kona RedVox Barometer Stack"
        # TODO: Fix plots, standardize units - go to libquantum plot templates
        fig1 = plot_tfr_bits(tfr_power=psd_stx, tfr_frequency=period_min, tfr_time=sig_time_days,
                             bits_min=-10, y_scale='log', tfr_x_str="Days from 2022-01-13 0Z",
                             tfr_y_str="Period, min", title_str=fig_title, tfr_y_flip=True)
        fig2 = plot_tfr_bits(tfr_power=psd_stx, tfr_frequency=frequency_cycles_per_day, tfr_time=sig_time_days,
                             bits_min=-10, y_scale='log', tfr_x_str="Days from 2022-01-13 0Z",
                             tfr_y_str="Cycles per day", title_str=fig_title)

        fig3 = plot_tfr_bits(tfr_power=psd_stx2, tfr_frequency=frequency2, tfr_time=sig_time_days,
                             bits_min=-10, y_scale='log', tfr_x_str="Days from 2022-01-13 0Z",
                             tfr_y_str="Hz", title_str=fig_title)

    plt.show()
