"""
Compute temporal powers of two from sample rate

"""

import numpy as np
from typing import Tuple


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


def duration_from_order_and_frequency(sig_frequency_hz: float, order_number: int = 6):
    """
    Return minimum duration in seconds for a specified signal frequency and order
    :param sig_frequency_hz: characteristic signal frequency in hz, recommend lower bound
    :param order_number: fractional octave band, default of 6th octave for transients
    :return: minimum window duration in seconds
    """
    min_number_cycles = 2*np.sqrt(2*np.log(2))*order_number
    min_duration_s = min_number_cycles/sig_frequency_hz
    return min_duration_s


def duration_from_order_and_period(sig_period_s: float, order_number: int = 6):
    """
    Return minimum duration in seconds for a specified signal frequency and order
    :param sig_frequency_hz: characteristic signal frequency in hz, recommend lower bound
    :param order_number: fractional octave band, default of 6th octave for transients
    :return: minimum window duration in seconds
    """
    min_number_cycles = 2*np.sqrt(2*np.log(2))*order_number
    min_duration_s = min_number_cycles*sig_period_s
    return min_duration_s
