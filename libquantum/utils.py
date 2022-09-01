"""
This module contains general utilities that can work with values containing nans.
"""

from enum import Enum
from typing import Tuple, Union

import numpy as np
from scipy import interpolate, signal

from scipy.integrate import cumulative_trapezoid
from libquantum.scales import EPSILON
from redvox.common import date_time_utils as dt

""" Logical power of two flag """


def is_power_of_two(n):
    """ Returns not if not positive and a power of two """
    return not (n > 0 and (n & (n - 1)))


""" Time/Sample Duration Utils """


def duration_points(sample_rate_hz: float, time_s: float) -> Tuple[int, int, int]:
    """
    Compute number of points
    :param sample_rate_hz: sample rate in Hz
    :param time_s: time scale, period or duration
    :return: number of points, floor and ceiling of log2 of number of points
    """
    points_float: float = sample_rate_hz * time_s
    points_int: int = int(points_float)
    points_floor_log2: int = int(np.floor(np.log2(points_float)))
    points_ceil_log2: int = int(np.ceil(np.log2(points_float)))

    return points_int, points_floor_log2, points_ceil_log2


def duration_ceil(sample_rate_hz: float, time_s: float) -> Tuple[int, int, float]:
    """
    Compute ceiling of the number of points, and convert to seconds
    :param sample_rate_hz: sample rate in Hz
    :param time_s: time scale, period or duration
    :return: ceil of log 2 of number of points, power of two number of points, corresponding time in s
    """
    points_float: float = sample_rate_hz * time_s
    points_ceil_log2: int = int(np.ceil(np.log2(points_float)))
    points_ceil_pow2: int = 2**points_ceil_log2
    time_ceil_pow2_s: float = points_ceil_pow2 / sample_rate_hz

    return points_ceil_log2, points_ceil_pow2, time_ceil_pow2_s


def duration_floor(sample_rate_hz: float, time_s: float) -> Tuple[int, int, float]:
    """
    Compute floor of the number of points, and convert to seconds
    :param sample_rate_hz: sample rate in Hz
    :param time_s: time scale, period or duration
    :return: floor of log 2 of number of points, power of two number of points, corresponding time in s
    """
    points_float: float = sample_rate_hz * time_s
    points_floor_log2: int = int(np.floor(np.log2(points_float)))
    points_floor_pow2: int = 2**points_floor_log2
    time_floor_pow2_s: float = points_floor_pow2 / sample_rate_hz

    return points_floor_log2, points_floor_pow2, time_floor_pow2_s


""" Sampling Utils """


def resample_uneven_signal(sig_wf: np.ndarray,
                           sig_epoch_s: np.ndarray,
                           sample_rate_new_hz: float = None):
    """

    :param sig_wf:
    :param sig_epoch_s:
    :param sample_rate_new_hz:
    :return:
    """

    if sample_rate_new_hz is None:
        interval_from_epoch_s = np.mean(np.diff(sig_epoch_s))
        # Round up
        sample_rate_new_hz = np.ceil(1/interval_from_epoch_s)

    interval_s = 1/sample_rate_new_hz
    sig_new_epoch_s = np.arange(sig_epoch_s[0], sig_epoch_s[-1], interval_s)
    f = interpolate.interp1d(sig_epoch_s, sig_wf)
    sig_new_wf = f(sig_new_epoch_s)
    return sig_new_wf, sig_new_epoch_s


def upsample_fourier(sig_wf: np.ndarray,
                     sig_sample_rate_hz: float,
                     new_sample_rate_hz: float = 8000.) -> np.ndarray:
    """
    Upsample the Fourier way.

    :param sig_wf: input signal waveform, reasonably well preprocessed
    :param sig_sample_rate_hz: signal sample rate
    :param new_sample_rate_hz: resampling sample rate
    :return: resampled signal
    """
    sig_len = len(sig_wf)
    new_len = int(sig_len * new_sample_rate_hz / sig_sample_rate_hz)
    sig_resampled = signal.resample(x=sig_wf, num=new_len)
    return sig_resampled


def taper_tukey(sig_wf_or_time: np.ndarray,
                fraction_cosine: float) -> np.ndarray:
    """
    Constructs a symmetric Tukey window with the same dimensions as a time or signal numpy array.
    fraction_cosine = 0 is a rectangular window, 1 is a Hann window

    :param sig_wf_or_time: input signal or time
    :param fraction_cosine: fraction of the window inside the cosine tapered window, shared between the head and tail
    :return: tukey taper window amplitude
    """
    return signal.windows.tukey(M=np.size(sig_wf_or_time), alpha=fraction_cosine, sym=True)


# def taper_tukey_array(number_signals: int,
#                       number_samples: int,
#                       fraction_cosine: float = 0.1) -> np.ndarray:
#     """
#     Construct a teper matrix
#     :param number_signals:
#     :param number_samples:
#     :param fraction_cosine:
#     :return:
#     """
#     tukey_nsamples = signal.windows.tukey(M=number_samples, alpha=fraction_cosine, sym=True)
#     tukey_array = just_tile(tukey_nsamples)


def datetime_now_epoch_s() -> float:
    """
    Returns the invocation Unix time in seconds

    :return: The current epoch timestamp as seconds since the epoch UTC
    """
    return dt.datetime_to_epoch_seconds_utc(dt.now())


def datetime_now_epoch_micros() -> float:
    """
    Returns the invocation Unix time in microseconds

    :return: The current epoch timestamp as microseconds since the epoch UTC
    """
    return dt.datetime_to_epoch_microseconds_utc(dt.now())


# Integrals and derivatives
def integrate_cumtrapz(timestamps_s: np.ndarray,
                       sensor_wf: np.ndarray,
                       initial_value: float = 0) -> np.ndarray:
    """
    Cumulative trapazoid integration using scipy.integrate.cumulative_trapezoid
    Initiated by Kei 2106, work in progress. See blast_derivative_integral for validation.

    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :param initial_value: the value to add in the initial of the integrated data to match length of input (default is 0)
    :return: integrated data with the same length as the input
    """
    integrated_data = cumulative_trapezoid(x=timestamps_s,
                                           y=sensor_wf,
                                           initial=initial_value)
    return integrated_data


def derivative_gradient(timestamps_s: np.ndarray,
                        sensor_wf: np.ndarray) -> np.ndarray:
    """
    Derivative using gradient

    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :return: derivative data with the same length as the input
    """
    # derivative_data = np.gradient(sensor_wf)/np.gradient(timestamps_s)
    derivative_data = np.gradient(sensor_wf, timestamps_s)

    return derivative_data


def derivative_diff(timestamps_s: np.ndarray,
                    sensor_wf: np.ndarray) -> np.ndarray:
    """
    Derivative using diff

    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :return: derivative data with the same length as the input. Hold/repeat last value
    """

    derivative_data0 = np.diff(sensor_wf)/np.diff(timestamps_s)
    derivative_data = np.append(derivative_data0, derivative_data0[-1])

    return derivative_data


class ExtractionType(Enum):
    """
    Enumeration of valid extraction types.

    ARGMAX = max of the absolute value of the signal
    SIGMAX = fancier signal picker, from POSITIVE max
    BITMAX = fancier signal picker, from ABSOLUTE max
    """
    ARGMAX: str = "argmax"
    SIGMAX: str = "sigmax"
    BITMAX: str = "bitmax"


def sig_extract(sig: np.ndarray,
                time_epoch_s: np.ndarray,
                intro_s: float,
                outro_s: float,
                pick_bits_below_max: float = 1.,
                pick_time_interval_s: float = 1.,
                extract_type: ExtractionType = ExtractionType.ARGMAX) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract signal and time relative to reference index

    :param sig: input signal
    :param time_epoch_s: signal epoch time
    :param intro_s: time before pick
    :param outro_s: time after pick
    :param pick_bits_below_max: pick treshold in bits below max
    :param pick_time_interval_s: pick time interval between adjacent max point
    :param extract_type: Type of extraction, see ExtractionType class
    :return: extracted signal np.ndarray, extracted signal timestamps np.ndarray, and pick time
    """

    sig_sample_interval_s = np.mean(np.diff(time_epoch_s))

    if extract_type == ExtractionType.ARGMAX:
        index_max = np.argmax(np.abs(sig))
        # Max pick
        pick_time_epoch_s = time_epoch_s[index_max]
    elif extract_type == ExtractionType.SIGMAX:
        time_index_pick_all = \
            picker_signal_max_index(sig=sig, sig_sample_rate_hz=1./sig_sample_interval_s,
                                    bits_pick=pick_bits_below_max, time_interval_s=pick_time_interval_s)
        # First pick
        pick_time_epoch_s = time_epoch_s[time_index_pick_all[0]]
    elif extract_type == ExtractionType.BITMAX:
        time_index_pick_all = \
            picker_signal_bit_index(sig=sig, sig_sample_rate_hz=1./sig_sample_interval_s,
                                    bits_pick=pick_bits_below_max, time_interval_s=pick_time_interval_s)
        # First pick
        pick_time_epoch_s = time_epoch_s[time_index_pick_all[0]]
    else:
        print('Unexpected extraction type to sig_extract, return max')
        index_max = np.argmax(np.abs(sig))
        # Max pick
        pick_time_epoch_s = time_epoch_s[index_max]

    epoch_s_start = pick_time_epoch_s - intro_s
    epoch_s_stop = pick_time_epoch_s + outro_s

    intro_index = np.argmin(np.abs(time_epoch_s - epoch_s_start))
    outro_index = np.argmin(np.abs(time_epoch_s - epoch_s_stop))
    sig_wf = sig[intro_index: outro_index]
    sig_epoch_s = time_epoch_s[intro_index: outro_index]

    return sig_wf, sig_epoch_s, pick_time_epoch_s


def sig_frame(sig: np.ndarray,
              time_epoch_s: np.ndarray,
              epoch_s_start: float,
              epoch_s_stop: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frame one-component signal within start and stop epoch times

    :param sig: input signal
    :param time_epoch_s: input epoch time in seconds
    :param epoch_s_start: start epoch time
    :param epoch_s_stop: stop epoch time
    :return: truncated time series and time
    """
    intro_index = np.argmin(np.abs(time_epoch_s - epoch_s_start))
    outro_index = np.argmin(np.abs(time_epoch_s - epoch_s_stop))
    sig_wf = sig[intro_index: outro_index]
    sig_epoch_s = time_epoch_s[intro_index: outro_index]

    return sig_wf, sig_epoch_s


def sig3c_frame(sig3c: np.ndarray,
                time_epoch_s: np.ndarray,
                epoch_s_start: float,
                epoch_s_stop: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frame three-component signal within start and stop epoch times

    :param sig3c: input signal with three components
    :param time_epoch_s: input epoch time in seconds
    :param epoch_s_start: start epoch time
    :param epoch_s_stop: stop epoch time
    :return: truncated time series and time
    """
    intro_index = np.argmin(np.abs(time_epoch_s - epoch_s_start))
    outro_index = np.argmin(np.abs(time_epoch_s - epoch_s_stop))
    sig_wf = sig3c[:, intro_index: outro_index]
    sig_epoch_s = time_epoch_s[intro_index: outro_index]

    return sig_wf, sig_epoch_s


def dbepsilon(x: np.ndarray) -> np.ndarray:
    """
    Converts the absolute value of a time series to dB

    :param x: time series
    :return: ndarray
    """
    y = 10 * np.log10(np.abs(x ** 2) + EPSILON)
    return y


def dbepsilon_max(x: np.ndarray) -> float:
    """
    Returns the max of the absolute value of a time series to dB

    :param x: time series
    :return: float
    """
    y = 10 * np.log10(np.max(np.abs(x ** 2)) + EPSILON)
    return y


def log2epsilon(x: np.ndarray) -> np.ndarray:
    """
    log 2 of the absolute value of linear amplitude, with EPSILON to avoid singularities

    :param x: time series or fft - not power
    :return: ndarray
    """
    y = np.log2(np.abs(x) + EPSILON)
    return y


def log2epsilon_max(x: np.ndarray) -> float:
    """
    max of the log 2 of absolute value of linear amplitude, with EPSILON to avoid singularities

    :param x: time series or fft - not power
    :return: float
    """
    y = np.max(np.log2(np.abs(x) + EPSILON))
    return y


"""
Picker modules
"""


def picker_signal_max_index(sig: np.array,
                            sig_sample_rate_hz: float,
                            bits_pick: float,
                            time_interval_s: float) -> np.array:
    """
    Finds the picker index for the POSITIVE max of a signal

    :param sig: array of waveform data
    :param sig_sample_rate_hz: float sample rate of reference sensor
    :param bits_pick: detection threshold in bits loss
    :param time_interval_s: min time interval between events
    :return: picker index
    """

    # Compute the distance
    distance_points = int(time_interval_s * sig_sample_rate_hz)
    height_min = np.max(sig) - 2 ** bits_pick
    time_index_pick, _ = signal.find_peaks(sig, height=height_min, distance=distance_points)

    return time_index_pick


def picker_signal_bit_index(sig: np.array,
                            sig_sample_rate_hz: float,
                            bits_pick: float,
                            time_interval_s: float) -> np.ndarray:
    """
    Finds the picker index from the ABSOLUTE max in bits

    :param sig: array of waveform data
    :param sig_sample_rate_hz: float sample rate of reference sensor
    :param bits_pick: detection treshold in db loss
    :param time_interval_s: min time interval between events
    :return: picker index
    """

    # Compute the distance
    distance_points = int(time_interval_s * sig_sample_rate_hz)
    sig_bit = log2epsilon(sig)
    height_min = log2epsilon_max(sig) - bits_pick
    index_pick, _ = signal.find_peaks(sig_bit, height=height_min, distance=distance_points)

    return index_pick


def picker_comb(sig_pick: np.ndarray,
                index_pick: np.ndarray) -> np.ndarray:
    """
    Constructs a comb function from the picks

    :param sig_pick: 1D record corresponding to the picks
    :param index_pick: indexes for the picks
    :return: comb with unit amplitude
    """
    comb = np.zeros(sig_pick.shape)
    comb[index_pick] = np.ones(index_pick.shape)
    return comb


"""
Matrix transformations
"""


def sum_columns(sxx: np.ndarray) -> np.ndarray:
    """
    Sum over all the columns in a 1D or 2D array

    :param sxx: input vector or matrix
    :return: ndarray with sum
    """
    if not isinstance(sxx, np.ndarray):
        raise TypeError('Input must be array.')
    elif len(sxx) == 0:
        raise ValueError('Cannot compute on empty array.')

    if np.isnan(sxx).any() or np.isinf(sxx).any():
        sxx = np.nan_to_num(sxx)

    if len(sxx.shape) == 1:
        sum_c = np.sum(sxx, axis=0)
    elif len(sxx.shape) == 2:
        sum_c = np.sum(sxx, axis=1)
    else:
        raise TypeError('Cannot handle an array of shape {}.'.format(str(sxx.shape)))

    return sum_c


def mean_columns(sxx: np.ndarray) -> np.ndarray:
    """
    Compute the mean of the columns in a 1D or 2D array

    :param sxx: input vector or matrix
    :return: ndarray with mean
    """
    if not isinstance(sxx, np.ndarray):
        raise TypeError('Input must be array.')
    elif len(sxx) == 0:
        raise ValueError('Cannot compute on empty array.')

    if np.isnan(sxx).any() or np.isinf(sxx).any():
        sxx = np.nan_to_num(sxx)

    if len(sxx.shape) == 1:
        sum_c = np.mean(sxx, axis=0)
    elif len(sxx.shape) == 2:
        sum_c = np.mean(sxx, axis=1)
    else:
        raise TypeError('Cannot handle an array of shape {}.'.format(str(sxx.shape)))

    return sum_c


def just_tile(array1d_in: Union[float, np.ndarray],
              shape_out: tuple) -> np.ndarray:
    """
    Constructs tiled array from 1D array to the shape specified by shape_out

    :param array1d_in: 1D array or vector
    :param shape_out: Tuple with output array shape.
    :return: ndarray
    """
    if len(shape_out) == 1:
        tiled_matrix = np.tile(array1d_in, (shape_out[0]))
    elif len(shape_out) == 2:
        tiled_matrix = np.tile(array1d_in, (shape_out[1], 1)).T
    else:
        raise TypeError('Cannot handle an array of shape {}.'.format(str(array1d_in.shape)))

    return tiled_matrix


def sum_tile(sxx: np.ndarray) -> np.ndarray:
    """
    Compute the sum of the columns in a 1D or 2D array and then re-tile to the original size

    :param sxx: input vector or matrix
    :return: ndarray of sum
    """
    sum_c = sum_columns(sxx)

    # create array of repeated values of PSD with dimensions that match those of energy array
    if len(sxx.shape) == 1:
        sum_c_matrix = np.tile(sum_c, (sxx.shape[0]))
    elif len(sxx.shape) == 2:
        sum_c_matrix = np.tile(sum_c, (sxx.shape[1], 1)).T
    else:
        raise TypeError('Cannot handle an array of shape {}.'.format(str(sxx.shape)))

    return sum_c_matrix


def mean_tile(sxx: np.ndarray,
              shape_out: np.ndarray) -> np.ndarray:
    """
    Compute the mean of the columns in a 1D or 2D array and then re-tile to the original size

    :param sxx: input vector or matrix
    :param shape_out: shape of output vector or matrix
    :return: ndarray of mean
    """
    sum_c = mean_columns(sxx)

    # create array of repeated values of PSD with dimensions that match those of energy array
    if len(shape_out) == 1:
        sum_c_matrix = np.tile(sum_c, (shape_out[0]))
    elif len(shape_out) == 2:
        sum_c_matrix = np.tile(sum_c, (shape_out[1], 1)).T
    else:
        raise TypeError('Cannot handle an array of shape {}.'.format(str(sxx.shape)))

    return sum_c_matrix


def d1tile_x_d2(d1: Union[float, np.ndarray],
                d2: np.ndarray) -> np.ndarray:
    """
    Create array of repeated values with dimensions that match those of energy array
    Useful to multiply frequency-dependent values to frequency-time matrices

    :param d1: 1D input vector, nominally frequency/scale multipliers
    :param d2: 2D array, first dimension should be that same as d1
    :return: array with matching values
    """
    shape_out = d2.shape

    if len(shape_out) == 1:
        d1_matrix = np.tile(d1, (shape_out[0]))
    elif len(shape_out) == 2:
        d1_matrix = np.tile(d1, (shape_out[1], 1)).T
    else:
        raise TypeError('Cannot handle an array of shape {}.'.format(str(d1.shape)))

    if d1_matrix.shape == d2.shape:
        d1_x_d2 = d1_matrix * d2
    else:
        raise TypeError('Cannot handle an array of shape {}.'.format(str(d1.shape)))
    return d1_x_d2


def decimate_array(sig_wf: np.array,
                   downsampling_factor: int) -> np.ndarray:
    """
    Decimate data and timestamps for an individual station
    All signals MUST have the same sample rate
    :param sig_wf: signal waveform
    :param downsampling_factor: the downsampling factor
    :param filter_order: the order of the filter
    :return: np.array decimated data
    """
    # decimate signal data
    decimated_data = signal.decimate(x=sig_wf,
                                    q=downsampling_factor,
                                    axis=1,
                                    zero_phase=True)

    return decimated_data
