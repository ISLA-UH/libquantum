"""
Envelope and edge functions

"""

import numpy as np
from typing import List, Tuple, Optional, Union


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