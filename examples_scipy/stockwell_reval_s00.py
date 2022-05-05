"""
Compute Stockwell TFR as in Moukadem et al. 2022
First test synthetic: synth_00
"""

import numpy as np
from scipy.fft import fft, rfft, ifft, fftfreq, fftshift
from libquantum.scales import EPSILON
from libquantum.stockwell import tfr_array_stockwell, calculate_rms_sig_test
from libquantum.benchmark_signals import synth_00
from libquantum.benchmark_signals import plot_tdr_sig, plot_tfr_lin, plot_tfr_bits, plot_st_window

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


def precompute_st_windows_isla(n_samp, sfreq, f_range, width, delta_f: int = None):
    """Precompute stockwell Gaussian windows (in the freq domain).
    From mne-python, _stockwell.py
    """

    tw = fftfreq(n_samp, 1. / sfreq) / n_samp
    tw = np.r_[tw[:1], tw[1:][::-1]]

    # TODO: FIX THIS!! Width sets the frequency interval
    k = width  # 1 for classical stockwell transform

    windows = np.empty((len(f_range), len(tw)), dtype=np.complex128)
    for i_f, f in enumerate(f_range):
        if f == 0.:
            window = np.ones(len(tw))
        else:
            window = ((f / (np.sqrt(2. * np.pi) * k)) *
                      np.exp(-0.5 * (1. / k ** 2.) * (f ** 2.) * tw ** 2.))
        window /= window.sum()  # normalisation
        windows[i_f] = fft(window)
    return windows


def st_power_isla(x, start_f, zero_pad, W):
    """Aux function."""
    n_samp = x.shape[-1]
    n_out = (n_samp - zero_pad)
    psd = np.empty((len(W), n_out))
    X = fft(x)
    XX = np.concatenate([X, X], axis=-1)
    print("XX:", XX.shape)
    for i_f, window in enumerate(W):
        f = start_f + i_f
        ST = ifft(XX[f:f + n_samp] * window)
        if zero_pad > 0:
            TFR = ST[:-zero_pad:1]
        else:
            TFR = ST[::1]
        TFR_abs = np.abs(TFR)
        # TODO: Correct default value
        TFR_abs[TFR_abs == 0] = 1.
        TFR_abs *= TFR_abs  # Square
        psd[i_f, :] = TFR_abs
    return psd


def tfr_array_stockwell_isla(data, sfreq, fmin=None, fmax=None, n_fft=None, delta_f=None,
                        width=1.0):
    """Compute power and intertrial coherence using Stockwell (S) transform.
    Same computation as `~mne.time_frequency.tfr_stockwell`, but operates on
    :class:`NumPy arrays <numpy.ndarray>` instead of `~mne.Epochs` objects.
    See :footcite:`Stockwell2007,MoukademEtAl2014,WheatEtAl2010,JonesEtAl2006`
    for more information.
    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_times)
        The signal to transform.
    sfreq : float
        The sampling frequency.
    fmin : None, float
        The minimum frequency to include. If None defaults to the minimum fft
        frequency greater than zero.
    fmax : None, float
        The maximum frequency to include. If None defaults to the maximum fft.
    n_fft : int | None
        The length of the windows used for FFT. If None, it defaults to the
        next power of 2 larger than the signal length.
    width : float
        The width of the Gaussian window. If < 1, increased temporal
        resolution, if > 1, increased frequency resolution. Defaults to 1.
        (classical S-Transform).

    Returns
    -------
    st_power : ndarray
        The multitaper power of the Stockwell transformed data.
    freqs : ndarray
        The frequencies.

    """

    # TODO: Test
    data, n_fft_, zero_pad = check_input_st_isla(data, n_fft)
    freqs = fftfreq(n_fft_, 1. / sfreq)

    # freqs = fftfreq(n_fft, 1. / sfreq)
    if fmin is None:
        fmin = freqs[freqs > 0][0]
    if fmax is None:
        fmax = freqs.max()

    start_f_idx = np.abs(freqs - fmin).argmin()
    stop_f_idx = np.abs(freqs - fmax).argmin()
    f_start = freqs[start_f_idx]
    f_stop = freqs[stop_f_idx]

    # TODO: Standardize (replace 12)
    if delta_f is None:
        if (f_stop - f_start) > 12:  # To reproduce examples
            delta_f = 1.
        else:
            delta_f = (f_stop - f_start)/12.

    # TODO: Add log space
    f_range = np.arange(f_start, f_stop, delta_f)

    W = precompute_st_windows_isla(data.shape[0], sfreq, f_range, width)
    psd = st_power_isla(data, start_f_idx, zero_pad=0, W=W)

    return psd, f_range, W


def main(sample_rate, signal_time_base: str='seconds'):
    """
    Evaluate synthetics
    :param sample_rate:
    :param signal_time_base:
    :return:
    """
    sample_interval = 1/sample_rate
    sig_in, time_in = synth_00(time_sample_interval=sample_interval, time_duration=1.024)
    n_fft = sig_in.shape[-1]
    print("Sig n:", n_fft)

    # Take FFT and concatenate
    X = fft(sig_in)
    XX = np.concatenate([X, X], axis=-1)
    freqs_fft = fftfreq(n_fft, sample_interval)
    # Make symmetric about origin
    freqs_fft_symmetric = fftshift(freqs_fft)
    print(freqs_fft.shape, X.shape, XX.shape)

    # f =  fs*t / n_fft
    # tw = freqs_fft / n_fft
    # print(tw)
    # # Keeps zero frequency [first element] at start, reverses all elements afterwards
    # # x[::-1] reverses all elements
    # tw = np.r_[tw[:1], tw[1:][::-1]]
    tw = freqs_fft_symmetric / n_fft
    print(tw)

    fmin = 10.
    fmax = 500.
    df = 50.

    # Find edges
    # freqs_s = np.arange(fmin, fmax, df)
    start_f_idx = np.abs(freqs_fft - fmin).argmin()
    stop_f_idx = np.abs(freqs_fft - fmax).argmin()
    f_start = freqs_fft[start_f_idx]
    f_stop = freqs_fft[stop_f_idx]
    freqs_s = np.arange(f_start, f_stop, df)

    print("TW.shape", tw.shape)
    # TODO: FIX THIS!! Width sets the frequency interval
    order = 8
    k = 3*order/8  # 1 for classical stockwell transform

    windows_t = np.empty((len(freqs_s), len(tw)), dtype=np.complex128)
    windows_f = np.empty((len(freqs_s), len(tw)), dtype=np.complex128)

    # TODO: Too low a frequency leads to instability - won't fit
    for i_f, f in enumerate(freqs_s):
        if f == 0.:
            window = np.ones(len(tw))
        else:
            window = ((f / (np.sqrt(2. * np.pi) * k)) *
                      np.exp(-0.5 * (1. / k ** 2.) * (f ** 2.) * tw ** 2.))

        windows_t[i_f] = window/window.sum()  # normalisation
        windows_f[i_f] = fft(windows_t[i_f])

    # Time domain window
    plot_st_window(window=windows_t, frequency_s=freqs_s, frequency_fft=tw)
    # Frequency domain window
    plot_st_window(window=windows_f, frequency_s=freqs_s, frequency_fft=freqs_fft)

    # plt.figure()
    # plt.plot(freqs_fft_symmetric, np.abs(X))
    # plt.title('X')
    #
    # plt.figure()
    # plt.plot(np.abs(XX))
    # plt.title('XX')
    # plt.show()

    # Stockwell
    [st_power, frequency, W] = tfr_array_stockwell_isla(data=sig_in,
                                                        sfreq=sample_rate,
                                                        fmin=fmin, fmax=fmax,
                                                        width=k, delta_f=df)

    plot_st_window(window=W, frequency_s=freqs_s, frequency_fft=freqs_fft)
    plt.show()

    exit()


    # TODO: Construct function
    print("Shape of W:", W.shape)
    print("Shape of frequency:", frequency.shape)
    plot_tdr_sig(sig_wf=sig_in, sig_time=time_in)
    plot_st_window(window=W, frequency_s=freqs_s, frequency_fft=freqs_fft)
    plot_tfr_lin(tfr_power=st_power, tfr_frequency=frequency, tfr_time=time_in)
    # plot_tfr_bits(tfr_power=st_power, tfr_frequency=frequency, tfr_time=time_in)
    plt.show()


if __name__ == "__main__":
    sample_rate_hz: float = 1000.0  # Sample frequency
    main(sample_rate=sample_rate_hz)
