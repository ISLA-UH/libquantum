"""
Compute Stockwell TFR as in Moukadem et al. 2022
First test synthetic: synth_00
"""

import numpy as np
from scipy.fft import fft, rfft, ifft, fftfreq
from libquantum.stockwell import tfr_array_stockwell, calculate_rms_sig_test
from libquantum.benchmark_signals import synth_00
from libquantum.benchmark_signals import plot_tdr, plot_tfr_lin, plot_tfr_log

from matplotlib import pyplot as plt
print(__doc__)


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
    # data, n_fft_, zero_pad = check_input_st(data, n_fft)
    # freqs = fftfreq(n_fft_, 1. / sfreq)

    freqs = fftfreq(n_fft, 1. / sfreq)
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

    # # Compute strided RMS
    # # TODO: Fix hop time and add offset on time from start
    # rms_sig_wf, rms_sig_time = calculate_rms_sig_test(sig_wf=sig_in, sig_time=time_in, points_per_seg=16)
    #
    # # TODO: Test convolution envelope
    # plot_tdr(sig_wf=sig_in, sig_time=time_in,
    #          sig_rms_wf=rms_sig_wf, sig_rms_time=rms_sig_time)


    freqs = np.arange(5., 500., 2.)
    fmin, fmax = freqs[[0, -1]]

    # Stockwell
    [st_power, frequency, W] = tfr_array_stockwell(data=sig_in, sfreq=sample_rate, fmin=fmin, fmax=fmax, width=3.0)

    # TODO: Construct function
    print("Shape of W:", W.shape)
    plt.figure()
    plt.plot(np.abs(W))
    plt.show()

    plot_tfr_lin(tfr_power=st_power, tfr_frequency=frequency, tfr_time=time_in)
    plot_tfr_log(tfr_power=st_power, tfr_frequency=frequency, tfr_time=time_in)
    plt.show()


if __name__ == "__main__":
    sample_rate_hz: float = 1000.0  # Sample frequency
    main(sample_rate=sample_rate_hz)
