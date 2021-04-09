"""
This module contains general utilities that can work with values containing nans.
"""

from enum import Enum
from typing import Tuple

import numpy as np
from scipy import signal

from libquantum.scales import EPSILON, MICROS_TO_S, KPA_TO_PA
from redvox.common import date_time_utils as dt
from redvox.common.station import Station


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


class NormType(Enum):
    """
    Enumeration of normalization types.
    """
    MAX: str = "max"
    L1: str = "l1"
    L2: str = "l2"
    OTHER: str = "other"


def normalize(sig: np.ndarray, scaling: float = 1., norm_type: NormType = NormType.MAX) -> np.ndarray:
    """
    Scale a 1D time series
    :param sig: time series signature
    :param scaling: scaling parameter, division
    :param norm_type: {'max', l1, l2}, optional
    :return: The scaled series
    """
    if norm_type == NormType.MAX:
        return sig / np.nanmax(np.abs(sig))
    elif norm_type == NormType.L1:
        return sig / np.nansum(sig)
    elif norm_type == NormType.L2:
        return sig / np.sqrt(np.nansum(sig * sig))
    else:  # Must be NormType.Other
        return sig / scaling


def demean_nan(sig: np.ndarray) -> np.ndarray:
    """
    Detrend and normalize a 1D time series
    :param sig: time series with (possibly) non-zero mean
    :return: Detrended and normalized time series
    """""
    return np.nan_to_num(sig - np.nanmean(sig))


def detrend_nan(sig: np.ndarray) -> np.ndarray:
    """
    Detrend and normalize a 1D time series
    :param sig: time series with (possibly) non-zero mean
    :return: Detrended and normalized time series
    """""
    return signal.detrend(demean_nan(sig))


def demean_nan_norm(sig: np.ndarray, scaling: float = 1., norm_type: NormType = NormType.MAX) -> np.ndarray:
    """
    Detrend and normalize a 1D time series
    :param sig: time series with (possibly) non-zero mean
    :param scaling: scaling parameter, division
    :param norm_type: {'max', l1, l2}, overrides scikit default of 'l2' by 'max'
    :return: The detrended and denormalized series.
    """""
    sig_detrend = demean_nan(sig)
    return normalize(sig_detrend, scaling=scaling, norm_type=norm_type)


def demean_nan_matrix(sig: np.ndarray) -> np.ndarray:
    """
    Detrend and normalize a matrix of time series
    :param sig: time series with (possibly) non-zero mean
    :return: The detrended and normalized signature
    """""
    return np.nan_to_num(np.subtract(sig.transpose(), np.nanmean(sig, axis=1))).transpose()


def taper_tukey(sig_or_time: np.ndarray, fraction_cosine: float) -> np.ndarray:
    """
    Constructs a symmetric Tukey window with the same dimensions as a time or signal numpy array.
    fraction_cosine = 0 is a rectangular window, 1 is a Hann window
    :param sig_or_time: input signal or time
    :param fraction_cosine: fraction of the window inside the cosine tapered window, shared between the head and tail
    :return: tukey taper window amplitude
    """
    number_points = np.size(sig_or_time)
    amplitude = signal.windows.tukey(M=number_points, alpha=fraction_cosine, sym=True)
    return amplitude


def mic_wf_time_build_station(station: Station,
                              mean_type: str = "simple",
                              raw: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Builds mic waveform and times
    :param station: the station with data
    :param mean_type: under development
    :param raw: if false (default), boolean or nan mean removed
    :return:
    """
    mic_sample_rate_hz = station.audio_sensor().sample_rate_hz
    mic_wf_raw = station.audio_sensor().get_data_channel("microphone")
    mic_epoch_s = station.audio_sensor().data_timestamps() * MICROS_TO_S

    if raw:
        mic_wf = np.array(mic_wf_raw)
    else:
        if mean_type == "simple":
            # Simple demean and replace nans with zeros. OK for mic, not OK for all other DC-biased sensors
            mic_wf = demean_nan(mic_wf_raw)
        else:
            # Remove linear trend
            mic_wf = detrend_nan(mic_wf_raw)

    return mic_wf, mic_epoch_s, mic_sample_rate_hz


def barometer_build_station(station: Station, raw: bool = True) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    gets barometer data from a station. Returns raw data
    :param station: the station with data
    :param raw: if false (default), boolean or nan mean removed
    :return: the barometer data, the timestamps, the estimated sample rate, and the indexes of the nans
    The nan indexes should be preserved throughout the computation and used in all the plots.
    """
    barometer_sample_rate_hz = station.barometer_sensor().sample_rate_hz
    barometer_raw = station.barometer_sensor().get_data_channel("pressure")
    barometer_epoch_s = station.barometer_sensor().data_timestamps() * MICROS_TO_S
    barometer_nans = np.argwhere(np.isnan(barometer_raw))
    if raw:
        barometer_wf = np.array(barometer_raw)
    else:
        # Detrend
        barometer_wf = detrend_nan(np.array(barometer_raw))

    return barometer_wf, barometer_epoch_s, barometer_sample_rate_hz, barometer_nans


def accelerometer_build_station(station: Station,
                                mean_type: str = "simple",
                                raw: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    gets accelerometer data from a station
    :param station: the station with data
    :param mean_type: under development
    :param raw: if false (default), boolean or nan mean removed
    :return: the accelerometer data, the timestamps, and the estimated sample rate
    """
    accelerometer_sample_rate_hz = station.accelerometer_sensor().sample_rate_hz
    accelerometer_raw = station.accelerometer_sensor().samples()
    accelerometer_epoch_s = station.accelerometer_sensor().data_timestamps() * MICROS_TO_S

    if raw:
        accelerometer_wf = np.array(accelerometer_raw)
    else:
        if mean_type == "simple":
            # Demeans and replaces nans with zeros for 3C sensors
            # TODO: Write function
            accelerometer_wf = np.nan_to_num(np.subtract(accelerometer_raw.transpose(), np.nanmean(accelerometer_raw, axis=1))).transpose()
        else:
            # Placeholder for diff solution with nans
            accelerometer_wf = np.nan_to_num(np.subtract(accelerometer_raw.transpose(), np.nanmean(accelerometer_raw, axis=1))).transpose()

    return accelerometer_wf, accelerometer_epoch_s, accelerometer_sample_rate_hz


def gyroscope_build_station(station: Station,
                            mean_type: str = "simple",
                            raw: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    gets gyroscope data from a station
    :param station: the station with data
    :param mean_type: under development
    :param raw: if false (default), boolean or nan mean removed
    :return: the gyroscope data, the timestamps, and the estimated sample rate
    """
    gyroscope_sample_rate_hz = station.gyroscope_sensor().sample_rate_hz
    gyroscope_raw = station.gyroscope_sensor().samples()
    gyroscope_epoch_s = station.gyroscope_sensor().data_timestamps() * MICROS_TO_S

    if raw:
        gyroscope_wf = np.array(gyroscope_raw)
    else:
        if mean_type == "simple":
            # Demeans and replaces nans with zeros for 3C sensors
            # TODO: Write function
            gyroscope_wf = np.nan_to_num(np.subtract(gyroscope_raw.transpose(), np.nanmean(gyroscope_raw, axis=1))).transpose()
        else:
            # Placeholder for diff solution with nans
            gyroscope_wf = np.nan_to_num(np.subtract(gyroscope_raw.transpose(), np.nanmean(gyroscope_raw, axis=1))).transpose()

    return gyroscope_wf, gyroscope_epoch_s, gyroscope_sample_rate_hz


def magnetometer_build_station(station: Station,
                               mean_type: str = "simple",
                               raw: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    gets magnetometer data from a station
    :param station: the station with data
    :param mean_type: under development
    :param raw: if false (default), boolean or nan mean removed
    :return: the magnetometer data, the timestamps, and the estimated sample rate
    """
    magnetometer_sample_rate_hz = station.magnetometer_sensor().sample_rate_hz
    magnetometer_raw = station.magnetometer_sensor().samples()
    magnetometer_epoch_s = station.magnetometer_sensor().data_timestamps() * MICROS_TO_S

    if raw:
        magnetometer_wf = np.array(magnetometer_raw)
    else:
        if mean_type == "simple":
            # Demeans and replaces nans with zeros for 3C sensors
            # TODO: Write function
            magnetometer_wf = \
                np.nan_to_num(np.subtract(magnetometer_raw.transpose(), np.nanmean(magnetometer_raw, axis=1))).transpose()
        else:
            # Placeholder for diff solution with nans
            magnetometer_wf = \
                np.nan_to_num(np.subtract(magnetometer_raw.transpose(), np.nanmean(magnetometer_raw, axis=1))).transpose()

    return magnetometer_wf, magnetometer_epoch_s, magnetometer_sample_rate_hz


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


def sig_extract(sig: np.ndarray, time_epoch_s: np.ndarray,
                intro_s: float, outro_s: float,
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
    :return:
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


def sig_frame(sig: np.ndarray, time_epoch_s: np.ndarray,
              epoch_s_start: float, epoch_s_stop: float) -> Tuple[np.ndarray, np.ndarray]:
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


def sig3c_frame(sig3c: np.ndarray, time_epoch_s: np.ndarray,
                epoch_s_start: float, epoch_s_stop: float):
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


# TODO: Migrate to their own modules
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


def picker_signal_bit_index(sig: np.array, sig_sample_rate_hz: float,
                            bits_pick: float, time_interval_s: float) -> np.array:
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


def picker_comb(sig_pick, index_pick):
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
    :return: ndarray
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
    :return: ndarray
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


def just_tile(array1d_in: np.ndarray, shape_out: tuple) -> np.ndarray:
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
    :return: ndarray
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


def mean_tile(sxx: np.ndarray, shape_out) -> np.ndarray:
    """
    Compute the mean of the columns in a 1D or 2D array and then re-tile to the original size
    :param sxx: input vector or matrix
    :param shape_out: shape of output vector or matrix
    :return: ndarray
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


def d1tile_x_d2(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    Create array of repeated values with dimensions that match those of energy array
    Useful to multiply frequency-dependent values to frequency-time matrices
    :param d1: 1D input vector, nominally frequency/scale multipliers
    :param d2: 2D array, first dimension should be that same as d1
    :return:
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
