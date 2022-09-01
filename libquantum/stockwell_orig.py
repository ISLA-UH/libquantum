"""
Stockwell
Compute Stockwell TFR from numpy array/tensor
https://mne.tools/stable/generated/mne.time_frequency.tfr_array_stockwell.html#mne.time_frequency.tfr_array_stockwell

"""

import numpy as np
from typing import List, Tuple, Optional, Union
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft, fftfreq
# print(__doc__)


def calculate_rms_sig(sig_wf: np.array,
                      sig_time_s: np.array,
                      points_per_seg: int = None,
                      points_hop: int = None) -> Tuple[np.array, np.array]:
    """
    RMS Audio
    # Adapted from https://stackoverflow.com/questions/45730504/how-do-i-create-a-sliding-window-with-a-50-overlap-with-a-numpy-array
    :param sig_wf: audio waveform
    :param sig_time_s: audio timestamps in seconds
    :param points_per_seg: number of points. Default is
    :param points_hop: number of points overlap per window. Default is 50%

    :return: rms_sig_wf, rms_sig_time_s
    """

    if points_per_seg is None:
        points_per_seg: int = int(len(sig_wf)/8)
    if points_hop is None:
        points_hop: int = int(0.5 * points_per_seg)

    # RMS sig wf
    # Initialize
    sh = (sig_wf.size - points_per_seg + 1, points_per_seg)  # shape of new array
    # Tuple of bytes to step in each dimension when traversing an array
    st = sig_wf.strides * 2

    # strided copy sig_wf, note stride spec
    sig_wf_windowed = np.lib.stride_tricks.as_strided(sig_wf, shape=sh, strides=st)[0::points_hop].copy()

    # List comprehension without intialization
    rms_sig_wf: List = [np.nanstd(sub_array) for sub_array in sig_wf_windowed]

    # sig time
    rms_sig_time_s = sig_time_s[0::points_hop]

    # check dims
    diff = abs(len(rms_sig_time_s) - len(rms_sig_wf))

    if (diff % 2) != 0:
        rms_sig_time_s = rms_sig_time_s[0:-1]
    else:
        rms_sig_time_s = rms_sig_time_s[1:-1]

    return rms_sig_wf, rms_sig_time_s


def calculate_rms_sig_test(sig_wf: np.array,
                           sig_time: np.array,
                           points_per_seg: int = None,
                           points_hop: int = None) -> Tuple[np.array, np.array]:
    """
    Look at
    https://localcoder.org/rolling-window-for-1d-arrays-in-numpy
    :param sig_wf: audio waveform
    :param sig_time: audio timestamps in seconds
    :param points_per_seg: number of points. Default is
    :param points_hop: number of points overlap per window. Default is 50%

    :return: rms_sig_wf, rms_sig_time_s
    """

    if points_per_seg is None:
        points_per_seg: int = int(len(sig_wf)/8)
    if points_hop is None:
        points_hop: int = int(0.5 * points_per_seg)

    # https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    sig_wf_windowed = \
        np.lib.stride_tricks.sliding_window_view(sig_wf, window_shape=points_per_seg)[0::points_hop, :].copy()

    # TODO: Build nan support
    rms_sig_wf = sig_wf_windowed.std(axis=-1)

    # sig time
    rms_sig_time_s = sig_time[0::points_hop].copy()

    # check dims
    diff = abs(len(rms_sig_time_s) - len(rms_sig_wf))

    if (diff % 2) != 0:
        rms_sig_time_s = rms_sig_time_s[0:-1]
    else:
        rms_sig_time_s = rms_sig_time_s[1:-1]

    return rms_sig_wf, rms_sig_time_s


def centers_to_edges(*arrays):
    """Convert center points to edges.
    Parameters
    ----------
    *arrays : list of ndarray
        Each input array should be 1D monotonically increasing,
        and will be cast to float.
    Returns
    -------
    arrays : list of ndarray
        Given each input of shape (N,), the output will have shape (N+1,).
    Examples
    --------
    > x = [0., 0.1, 0.2, 0.3]
    > y = [20, 30, 40]
    > centers_to_edges(x, y)  # doctest: +SKIP
    [array([-0.05, 0.05, 0.15, 0.25, 0.35]), array([15., 25., 35., 45.])]
    """
    out = list()
    for ai, arr in enumerate(arrays):
        arr = np.asarray(arr, dtype=float)
        # _check_option(f'arrays[{ai}].ndim', arr.ndim, (1,))
        if len(arr) > 1:
            arr_diff = np.diff(arr) / 2.
        else:
            arr_diff = [abs(arr[0]) * 0.001] if arr[0] != 0 else [0.001]
        out.append(np.concatenate([
            [arr[0] - arr_diff[0]],
            arr[:-1] + arr_diff,
            [arr[-1] + arr_diff[-1]]]))
    return out


def check_input_st(x_in, n_fft):
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


def precompute_st_windows_orig(n_samp, start_f, stop_f, sfreq, width):
    """Precompute stockwell Gaussian windows (in the freq domain).
    From mne-python, _stockwell.py
    """

    tw = fftfreq(n_samp, 1. / sfreq) / n_samp
    tw = np.r_[tw[:1], tw[1:][::-1]]

    # TODO: FIX THIS!!
    k = width  # 1 for classical stockwell transform
    # TODO: FIX THIS!!
    f_range = np.arange(start_f, stop_f, 1)
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


def st_power_orig(x, start_f, zero_pad, W):
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


def precompute_st_windows(n_samp, sfreq, f_range, width):
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


def st_power(x, start_f, zero_pad, W):
    """Aux function."""
    n_samp = x.shape[-1]
    n_out = (n_samp - zero_pad)
    psd = np.empty((len(W), n_out))
    X = fft(x)
    XX = np.concatenate([X, X], axis=-1)
    # print("XX:", XX.shape)
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


def tfr_array_stockwell(data, sfreq, fmin=None, fmax=None, n_fft=None, delta_f=None,
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

    data, n_fft_, zero_pad = check_input_st(data, n_fft)


    freqs = fftfreq(n_fft_, 1. / sfreq)
    if fmin is None:
        fmin = freqs[freqs > 0][0]
    if fmax is None:
        fmax = freqs.max()

    start_f_idx = np.abs(freqs - fmin).argmin()
    stop_f_idx = np.abs(freqs - fmax).argmin()
    f_start = freqs[start_f_idx]
    f_stop = freqs[stop_f_idx]

    # TODO: Standardize (replace 12), use ratios
    if delta_f is None:
        if (f_stop - f_start) > 8:  # To reproduce examples
            delta_f = 1.
        else:
            delta_f = (f_stop - f_start)/12.

    # TODO: Add log space
    f_range = np.arange(f_start, f_stop, delta_f)

    W = precompute_st_windows(data.shape[0], sfreq, f_range, width)
    psd = st_power(data, start_f_idx, zero_pad, W)

    return psd, f_range, W

