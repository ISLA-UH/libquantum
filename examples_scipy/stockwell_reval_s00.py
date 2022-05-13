"""
Compute Stockwell TFR as in Moukadem et al. 2022
First test synthetic: synth_00
"""

import numpy as np
from scipy.fft import fft, rfft, ifft, fftfreq, fftshift
from libquantum.scales import EPSILON
# from libquantum.stockwell_orig import tfr_array_stockwell, calculate_rms_sig_test
from libquantum.benchmark_signals import synth_00
from libquantum.benchmark_signals import plot_tdr_sig, plot_tfr_lin, plot_tfr_bits, \
    plot_st_window_tdr_lin, plot_st_window_tfr_bits, plot_st_window_tfr_lin

from matplotlib import pyplot as plt
print(__doc__)


def check_input_st_isla(x_in, n_fft):
    """Aux function."""
    # flatten to 2 D and memorize original shape
    n_times = x_in.shape[-1]

    def _is_power_of_two(n):
        return not (n > 0 and (n & (n - 1)))

    if n_fft is None or (not _is_power_of_two(n_fft) and n_times > n_fft):
        # Compute next power of 2
        n_fft = 2 ** int(np.ceil(np.log2(n_times)))
    elif n_fft < n_times:
        raise ValueError("n_fft cannot be smaller than signal size. "
                         "Got %s < %s." % (n_fft, n_times))
    if n_times < n_fft:
        print('The input signal is shorter ({}) than "n_fft" ({}). '
              'Applying zero padding.'.format(x_in.shape[-1], n_fft))
        zero_pad = n_fft - n_times
        pad_array = np.zeros(x_in.shape[:-1] + (zero_pad,), x_in.dtype)
        x_in = np.concatenate((x_in, pad_array), axis=-1)
    else:
        zero_pad = 0
    return x_in, n_fft, zero_pad


def tfr_array_stockwell_isla(data, sample_rate, fmin=None, fmax=None, n_fft=None, df=None,
                             order=8.0, binary_order: str = False):

    M = 12/5*order
    sample_interval = 1/sample_rate
    if n_fft is None:
        n_fft = data.shape[-1]

    Tw = n_fft/sample_rate
    fmin_nth = 12/5 * order/Tw

    if fmin is None:
        fmin = fmin_nth
    if fmax is None:
        fmax = sample_rate/2.
    if df is None:
        df = 1.

    # Take FFT and concatenate
    X = fft(data)
    XX = np.concatenate([X, X], axis=-1)
    freqs_fft = fftfreq(n_fft, sample_interval)
    mu_fft = 2*np.pi*freqs_fft/sample_rate

    # TODO: Standardize
    start_f_idx = np.abs(freqs_fft - fmin).argmin()
    stop_f_idx = np.abs(freqs_fft - fmax).argmin()
    f_start = freqs_fft[start_f_idx]
    f_stop = freqs_fft[stop_f_idx]
    if binary_order is True:
        num_octaves = np.log2(f_stop/f_start)
        num_bands = int(num_octaves*order)
        print("Number of bands:", num_bands)
        freqs_s = np.logspace(np.log2(f_start), np.log2(f_stop), num=num_bands, base=2.)
    else:
        freqs_s = np.arange(f_start, f_stop, df)
    print("Shape of freqs_s", freqs_s.shape)
    print("SX Band edges:", freqs_s[0], freqs_s[-1])

    # TODO: Consolidate all the loops over freqs_s
    # Construct shifting frequency indexes
    freqs_s_index = np.empty(len(freqs_s), dtype=int)
    for isx, fsx in enumerate(freqs_s):
        freqs_s_index[isx] = np.abs(freqs_fft - freqs_s[isx]).argmin()

    # TODO: find fft frequencies that exactly match - necessary for shifting FFT
    # Gabor window
    # TODO: Reconcile with above granularity
    nu_s = 2*np.pi*freqs_s/sample_rate  # Nondimensionalized angular frequency
    windows_f3 = np.empty((len(freqs_s), len(data)), dtype=np.complex128)

    # TODO: These can be constructed as a matrix w/o for loop
    for i_f, nu in enumerate(nu_s):
        if nu == 0.:
            windows_f3[i_f] = np.ones(len(data))
        else:
            sigma = M/nu
            windows_f3[i_f] = np.exp(-0.5 * (sigma ** 2.)*(mu_fft ** 2.))

    psd3 = np.empty((len(windows_f3), n_fft))

    # Frequency shift the FFT, then compute the IFFT
    for i_f, window in enumerate(windows_f3):
        # this is f_fft + f_s
        f_idx = freqs_s_index[i_f]
        TFR = ifft(XX[f_idx:f_idx + n_fft] * window) + EPSILON
        TFR_abs = np.abs(TFR)
        TFR_abs *= TFR_abs  # Square
        psd3[i_f, :] = TFR_abs

    return psd3, freqs_s, freqs_fft, windows_f3


def main(sample_rate, signal_time_base: str='seconds'):
    """
    Evaluate synthetics
    :param sample_rate:
    :param signal_time_base:
    :return:
    """

    order = 8
    # k = 3*order/8  # 1 for classic stockwell transform
    k = 3

    sample_interval = 1/sample_rate
    sig_in, time_in = synth_00(time_sample_interval=sample_interval, time_duration=1.024)

    n_fft = sig_in.shape[-1]
    Tw = n_fft/sample_rate
    fmin_nth = 12/5 * order/Tw
    print("Signal number of points:", n_fft)
    print("Min frequency, hz:", fmin_nth)
    fmin = fmin_nth
    fmax = sample_rate/2.

    # Stockwell, canned version; breaks if df not 1
    # The default df = 1
    [st_power, frequency, W] = \
        tfr_array_stockwell(data=sig_in,
                            sfreq=sample_rate,
                            fmin=fmin, fmax=fmax,
                            width=k, delta_f=1)
    [psd3, freqs_s, freqs_fft, windows_f3] = \
        tfr_array_stockwell_isla(data=sig_in,
                                 sample_rate=sample_rate,
                                 fmin=fmin, fmax=fmax,
                                 order=order, df=10)
    # # plot_st_window_tfr_lin(window=W, frequency_sx=frequency, frequency_fft=freqs_fft)
    # # plot_st_window_tfr_lin(window=windows_f3, frequency_sx=freqs_s, frequency_fft=freqs_fft)
    # plot_st_window_tfr_lin(window=fftshift(W), frequency_sx=frequency, frequency_fft=fftshift(freqs_fft))
    # plot_st_window_tfr_lin(window=fftshift(windows_f3), frequency_sx=freqs_s, frequency_fft=fftshift(freqs_fft))
    # plt.show()

    print("Shape of W:", st_power.shape)
    print("Shape of frequency:", frequency.shape)
    print("Shape of psd3:", psd3.shape)
    print("Shape of freqs_s", freqs_s.shape)
    # print(freqs_s)
    # plot_tdr_sig(sig_wf=sig_in, sig_time=time_in)
    # plot_st_window_tfr_bits(window=W, frequency_sx=freqs_s, frequency_fft=freqs_fft)
    # plot_tfr_bits(tfr_power=st_power, tfr_frequency=frequency, tfr_time=time_in)
    # plot_tfr_bits(tfr_power=psd3, tfr_frequency=freqs_s, tfr_time=time_in)
    plot_tfr_lin(tfr_power=st_power, tfr_frequency=frequency, tfr_time=time_in)
    plot_tfr_lin(tfr_power=psd3, tfr_frequency=freqs_s, tfr_time=time_in)
    plt.show()


if __name__ == "__main__":
    sample_rate_hz: float = 1000.0  # Sample frequency
    main(sample_rate=sample_rate_hz)
