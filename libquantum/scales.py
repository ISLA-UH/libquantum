"""
Construct standardized scales
"""

import numpy as np
from typing import Tuple, Union

""" Smallest number for 64-bit floats. Deploy to avoid division by zero or log zero singularities"""
EPSILON = np.finfo(np.float64).eps
MICROS_TO_S = 1E-6
MICROS_TO_MILLIS = 1E-3
KPA_TO_PA = 1E3
DEGREES_TO_KM = 111.
"""
Standardized scales
"""


class Slice:
    """
    Constants for slice calculations, supersedes inferno/slice
    """
    # Constant Q Base
    G2 = 2.  # Octaves
    G3 = 10. ** 0.3  # Reconciles base2 and base10
    # Time
    T_PLANCK = 5.4E-44  # 2.**(-144)   Planck time in seconds
    T0S = 1E-42  # Universal Scale
    T1S = 1.    # 1 second
    T100S = 100.  # 1 hectosecond, IMS low band edge
    T1000S = 1000.  # 1 kiloseconds = 1 mHz
    T1M = 60.  # 1 minute in seconds
    T1H = T1M*60.  # 1 hour in seconds
    T1D = T1H*24.   # 1 day in seconds
    TU = 2.**58  # Estimated age of the known universe in seconds
    # Frequency
    F1 = 1.  # 1 Hz
    F1000 = 1000.  # 1 kHz
    F0 = 1.E42  # 1/Universal Scale
    FU = 2.**-58  # Estimated age of the universe in Hz
    # Pressure
    PREF_KPA = 101.325  # sea level pressure, kPa
    PREF_PA = 10132500.  # sea level pressure, kPa


def planck_scale_s(scale_order: float) -> Tuple[float, float, float, float]:
    """
    Calculate Planck scale

    :param scale_order: scale order
    :return: center in seconds, minimum in seconds, maximum in seconds, quality factor Q
    """
    # Assumes base 2, Slice.G2
    cycles_M, quality_factor_Q = wavelet_MQ_from_N(scale_order)
    scale_edge = Slice.G2 ** (1.0 / (2.0 * scale_order))
    # # Q = center/bandwidth
    # planck_scale_zero_s = planck_scale_bandwidth_s*quality_factor_Q
    # Must be greater than Planck scale
    planck_scale_center_s = Slice.T0S
    # Smallest scale
    planck_scale_min_s = planck_scale_center_s/scale_edge
    planck_scale_max_s = planck_scale_center_s*scale_edge
    return planck_scale_center_s, planck_scale_min_s, planck_scale_max_s, quality_factor_Q


def musical_scale_hz():
    """
    Returns frequencies for equal tempered scale
    base2, ref = 440 Hz, 12th octaves
    """
    return band_frequencies_nyquist(frequency_order_input=12,
                                    frequency_base_input=2,
                                    frequency_ref_input=440,
                                    frequency_low_input=16.35,
                                    frequency_sample_rate_input=48000)


def band_periods_nyquist(scale_order_input: float,
                         scale_base_input: float,
                         scale_ref_input: float,
                         scale_sample_interval_input: float,
                         scale_high_input: float) -> Tuple[float, float, np.ndarray, float, np.ndarray,
                                                           np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Standard Logarithmic Interval Time Parameters: ALWAYS USE SECONDS

    Parameters
    ----------
    scale_order_input: float
        Band order N, for ISO3 use N = 1.0 or 3.0 or 6.0 or 12.0 or 24.0
    scale_base_input: float
        reference base G; i.e. G3 = 10.**0.3 or G2 = 2.0
    scale_ref_input: float
        time reference: in seconds
    scale_sample_interval_input: float
        Sample interval for scale: float
    scale_high_input: float
        Highest scale of interest in seconds

    Returns
    -------
    scale_order: float
        Band order N > 1, defaults to 1.
    scale_base: float
        positive reference Base G > 1, defaults to G3
    scale_ref: float
        positive reference scale
    scale_band_number: numpy ndarray
        Band number n
    scale_center_algebraic: numpy ndarray
        Algebraic center of band scale
    scale_center_geometric: numpy ndarray
        Geometric center of band scale
    scale_start: numpy ndarray
        Lower band edge scale
    scale_end: numpy ndarray
        Upper band edge scale
    """
    scale_nyquist_input = 2*scale_sample_interval_input
    scale_order, scale_base, scale_band_number, scale_ref, scale_center_algebraic, scale_center_geometric, scale_start, \
    scale_end = band_intervals_periods(scale_order_input=scale_order_input,
                                       scale_base_input=scale_base_input,
                                       scale_ref_input=scale_ref_input,
                                       scale_low_input=scale_nyquist_input,
                                       scale_high_input=scale_high_input)

    return scale_order, scale_base, scale_band_number, scale_ref, scale_center_algebraic, scale_center_geometric, \
           scale_start, scale_end


def band_frequencies_nyquist(frequency_order_input: float,
                             frequency_base_input: float,
                             frequency_ref_input: float,
                             frequency_low_input: float,
                             frequency_sample_rate_input: float) -> Tuple[float, float, np.ndarray, float, float,
                                                                          np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Standard Logarithmic Interval Time Parameters: ALWAYS USE HZ

    :param frequency_order_input: Nth order
    :param frequency_base_input: G2 or G3
    :param frequency_ref_input: reference frequency
    :param frequency_low_input: lowest frequency of interest
    :param frequency_sample_rate_input: sample rate
    :return: scale_order (Band order N > 1, defaults to 1.),
        scale_base (positive reference Base G > 1, defaults to G3),
        scale_band_number (Band number n),
        frequency_ref (reference frequency value),
        frequency_center_algebraic (Algebraic center of frequencies),
        frequency_center_geometric (Geometric center of frequencies),
        frequency_start (first frequency),
        frequency_end (last frequency)

    """

    scale_ref_input = 1/frequency_ref_input
    scale_nyquist_input = 2/frequency_sample_rate_input
    scale_high_input = 1/frequency_low_input

    scale_order, scale_base, scale_band_number, \
    scale_ref, scale_center_algebraic, scale_center_geometric, \
    scale_start, scale_end = \
        band_intervals_periods(frequency_order_input, frequency_base_input,
                               scale_ref_input,
                               scale_nyquist_input, scale_high_input)
    frequency_ref = 1/scale_ref
    frequency_center_geometric = 1/scale_center_geometric
    frequency_end = 1/scale_start
    frequency_start = 1/scale_end
    frequency_center_algebraic = (frequency_end + frequency_start)/2.
    
    # Inherit the order, base, and band number
    return scale_order, scale_base, -scale_band_number, frequency_ref, frequency_center_algebraic, \
           frequency_center_geometric, frequency_start, frequency_end


def band_frequencies_low_high(frequency_order_input: float,
                              frequency_base_input: float,
                              frequency_ref_input: float,
                              frequency_low_input: float,
                              frequency_high_input: float,
                              frequency_sample_rate_input: float) -> Tuple[float, float, np.ndarray, float, np.ndarray,
                                                                           np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Standard Logarithmic Interval Time Parameters: ALWAYS USE HZ

    :param frequency_order_input: Nth order
    :param frequency_base_input: G2 or G3
    :param frequency_ref_input: reference frequency
    :param frequency_low_input: lowest frequency of interest
    :param frequency_high_input: highest frequency of interest
    :param frequency_sample_rate_input: sample rate
    :return:scale_order (Band order N > 1, defaults to 1.),
        scale_base (positive reference Base G > 1, defaults to G3),
        scale_band_number (Band number n),
        frequency_ref (reference frequency value),
        frequency_center_algebraic (Algebraic center of frequencies),
        frequency_center_geometric (Geometric center of frequencies),
        frequency_start (first frequency),
        frequency_end (last frequency)
    """

    scale_ref_input = 1/frequency_ref_input
    scale_nyquist_input = 2/frequency_sample_rate_input
    scale_low_input = 1/frequency_high_input
    if scale_low_input < scale_nyquist_input:
        scale_low_input = scale_nyquist_input
    scale_high_input = 1/frequency_low_input

    scale_order, scale_base, scale_band_number, \
    scale_ref, scale_center_algebraic, scale_center_geometric, \
    scale_start, scale_end = \
        band_intervals_periods(frequency_order_input, frequency_base_input,
                               scale_ref_input,
                               scale_low_input,
                               scale_high_input)
    frequency_ref = 1/scale_ref
    frequency_center_geometric = 1/scale_center_geometric
    frequency_end = 1/scale_start
    frequency_start = 1/scale_end
    frequency_center_algebraic = (frequency_end + frequency_start)/2.

    # Inherit the order, base, and band number
    return scale_order, scale_base, -scale_band_number,  frequency_ref, frequency_center_algebraic, \
           frequency_center_geometric, frequency_start, frequency_end


def band_intervals_periods(scale_order_input: float,
                           scale_base_input: float,
                           scale_ref_input: float,
                           scale_low_input: float,
                           scale_high_input: float) -> Tuple[float, float, np.ndarray, float, np.ndarray,
                                                             np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Standard Logarithmic Interval Scale Parameters using time scales in seconds
    If scales are provided as frequency, previous computations convert to time.
    Designed to take bappsband to just below Nyquist, within a band edge.
    ALWAYS CONVERT TO SECONDS
    Last updated: 20200905

    Parameters
    ----------
    scale_order_input: float
        Band order N, for ISO3 use N = 1.0 or 3.0 or 6.0 or 12.0 or 24.0
    scale_base_input: float
        reference base G; i.e. G3 = 10.**0.3 or G2 = 2.0
    scale_ref_input: float
        time reference: in seconds
    scale_low_input: float
        Lowest scale. If Nyquist scale, 2 * sample interval in seconds.
    scale_high_input: float
        Highest scale of interest in seconds

    Returns
    -------
    scale_order: float
        Band order N > 1, defaults to 1.
    scale_base: float
        positive reference Base G > 1, defaults to G3
    scale_ref: float
        positive reference scale
    scale_band_number: numpy ndarray
        Band number n
    scale_center_algebraic: numpy ndarray
        Algebraic center band scale
    scale_center_geometric: numpy ndarray
        Geometric center band scale
    scale_start: numpy ndarray
        Lower band edge scale
    scale_end: numpy ndarray
        Upper band edge scale
    """
    # Initiate error handling, all inputs should be numeric, positive, and real
    # Need error check for string inputs
    # If not real and positive, make them so
    [scale_ref, scale_low, scale_high, scale_base, scale_order] = np.absolute(
        [scale_ref_input, scale_low_input, scale_high_input, scale_base_input, scale_order_input])

    # print('\nConstructing standard band intervals')
    # Check for compliance with ISO3 and/or ANSI S1.11 and for scale_order = 1, 3, 6, 12, and 24
    if scale_base == Slice.G3:
        pass
    elif scale_base == Slice.G2:
        pass
    elif scale_base < 1.:
        print('\nWARNING: Base must be greater than unity. Overriding to G = 2')
        scale_base = Slice.G2
    else:
        print('\nWARNING: Base is not ISO3 or ANSI S1.11 compliant')
        print(('Continuing With Non-standard base = ' + repr(scale_base) + ' ...'))

    # Check for compliance with ISO3 for scale_order = 1, 3, 6, 12, and 24
    # and the two 'special' orders 0.75 and 1.5
    valid_scale_orders = (0.75, 1, 1.5, 3, 6, 12, 24, 48)
    if scale_order in valid_scale_orders:
        pass
    elif scale_order < 0.75:
        print('Order must be greater than 0.75. Overriding to Order 1')
        scale_order = 1
    else:
        print(('\nWARNING: Recommend Orders ' + str(valid_scale_orders)))
        print(('Continuing With Non-standard Order = ' + repr(scale_order) + ' ...'))

    # Compute scale edge and width parameters
    scale_edge = scale_base ** (1.0 / (2.0 * scale_order))
    scale_width = scale_edge - 1.0 / scale_edge

    if scale_low < Slice.T0S:
        scale_low = Slice.T0S/scale_edge
    if scale_high < scale_low:
        print('\nWARNING: Upper scale must be larger than the lowest scale')
        print('Overriding to min = max/G  \n')
        scale_low = scale_high / scale_base
    if scale_high == scale_low:
        print('\nWARNING: Upper scale = lowest scale, returning closest band edges')
        scale_high *= scale_edge
        scale_low /= scale_edge

    # Max and min bands are computed relative to the center scale
    n_max = np.round(scale_order * np.log(scale_high / scale_ref) / np.log(scale_base))
    n_min = np.floor(scale_order * np.log(scale_low / scale_ref) / np.log(scale_base))

    # Evaluate min, ensure it stays below Nyquist period
    scale_center_n_min = scale_ref * np.power(scale_base, n_min/scale_order)
    if (scale_center_n_min < scale_low) or (scale_center_n_min/scale_edge < scale_low-EPSILON):
        n_min += 1

    # Check for band number anomalies
    if n_max < n_min:
        print('\nSPECMOD: Insufficient bandwidth for Nth band specification')
        print(('Minimum scaled bandwidth (scale_high - scale_low)/scale_center = ' + repr(scale_width) + '\n'))
        print('Correct scale High/Low input parameters \n')
        print('Apply one order  \n')
        n_max = np.floor(np.log10(scale_high) / np.log10(scale_base))
        n_min = n_max - scale_order

    # Band number array for Nth octave
    scale_band_number = np.arange(n_min, n_max + 1)

    # Compute exact, evenly (log) distributed, constant Q,
    # Nth octave center and band edge frequencies
    scale_band_exponent = scale_band_number / scale_order

    scale_center_geometric = scale_ref * np.power(scale_base * np.ones(scale_band_number.shape), scale_band_exponent)
    scale_start = scale_center_geometric / scale_edge
    scale_end = scale_center_geometric * scale_edge
    # The spectrum is centered on the algebraic center scale
    scale_center_algebraic = (scale_start+scale_end)/2.

    return scale_order, scale_base, scale_band_number,  scale_ref, scale_center_algebraic, scale_center_geometric, \
           scale_start, scale_end


"""
Quantum wavelet specifications
"""


def wavelet_MQ_from_N(band_order_Nth: float) -> Tuple[float, float]:
    """
    Compute the quality factor Q and multiplier M for a specified band order N
    N is THE quantization parameter for the binary constant Q wavelet filters

    :param band_order_Nth: Band order, must be > 0.75 or reverts to N=3
    :return: float, float
    """
    if band_order_Nth < 0.7:
        print('N<0.7 specified, using N = ', 3)
        band_order_Nth = 3.
    order_bandedge = 2 ** (1. / 2. / band_order_Nth)  # kN in Garces 2013
    order_scaled_bandwidth = order_bandedge - 1. / order_bandedge
    quality_factor_Q = 1./order_scaled_bandwidth  # Exact for Nth octave bands
    cycles_M = quality_factor_Q*2*np.sqrt(np.log(2))  # Exact, from -3dB points
    return cycles_M, quality_factor_Q


def wavelet_NMQ_from_Q(quality_factor_Q: float) -> Tuple[float, float, float]:
    """
    For a specified Q, estimate order N and recompute exact Q and number of cycles M from N

    :param quality_factor_Q: number of oscillations with significant amplitude in the frame
    :return: band order Nth (float), number of cycles M (float), quality factor Q (float)
    """
    band_order_Nth = quality_factor_Q/np.sqrt(2)  # Approximate
    cycles_M, quality_factor_Q = wavelet_MQ_from_N(band_order_Nth)
    return band_order_Nth, cycles_M, quality_factor_Q


def wavelet_NMQ_from_M(cycles_M: float) -> Tuple[float, float, float]:
    """
    For a specified M, estimate order N and recompute exact Q and number of cycles M from N

    :param quality_factor_Q: number of oscillations with significant amplitude in the frame
    :return: band order Nth (float), number of cycles M (float), quality factor Q (float)
    """
    quality_factor_Q = cycles_M/(2*np.sqrt(np.log(2)))  # Exact, from -3dB/half bit points
    band_order_Nth, cycles_M, quality_factor_Q = wavelet_NMQ_from_Q(quality_factor_Q)
    return band_order_Nth, cycles_M, quality_factor_Q


def wavelet_support(band_order_Nth : float,
                    scale_frequency_center_hz: float,
                    frequency_sample_rate_hz: float,
                    is_power_2: bool = True) -> Tuple[int, float, float]:
    """
    Compact support for Gabor wavelet

    :param band_order_Nth: Nth order of constant Q bands
    :param scale_frequency_center_hz: scale frequency in hz
    :param is_power_2: power of two approximation
    :return: number of points of duration (int), scale frame in seconds (float), scale of atom (float)
    """
    cycles_M, _ = wavelet_MQ_from_N(band_order_Nth)
    scale_frame_T_s = cycles_M / scale_frequency_center_hz   # Lifetime Tn, in seconds, M/fc
    scale_atom = frequency_sample_rate_hz*scale_frame_T_s / (2. * np.pi)  # Canonical Gabor atom scale
    nominal_points = np.floor(scale_frame_T_s*frequency_sample_rate_hz)

    floor_switch = 0.8  # Theoretical foundation, empirical implementation
    if is_power_2:
        duration_points_floor = 2**(np.floor(np.log2(nominal_points)))
        duration_points_ceil = 2**(np.ceil(np.log2(nominal_points)))
        if duration_points_floor < floor_switch*nominal_points:
            duration_points = int(duration_points_ceil)
        else:
            duration_points = int(duration_points_floor)
    else:
        duration_points = int(nominal_points)
    return duration_points, scale_frame_T_s, scale_atom


def from_duration(band_order_Nth : float,
                  sig_duration_s: float) -> Tuple[float, float]:
    """
    Calculate scale factor for time and frequency from signal duration

    :param band_order_Nth: Nth order of constant Q bands
    :param sig_duration_s: total signal duration in seconds
    :return: scale time in seconds, scale frequency in Hz
    """
    cycles_M, _ = wavelet_MQ_from_N(band_order_Nth)
    scale_time_s = sig_duration_s/cycles_M
    scale_frequency_hz = 1/scale_time_s
    return scale_time_s, scale_frequency_hz


def frequency_bands_g2f1(scale_order_input: float,
                         frequency_low_input: float,
                         frequency_sample_rate_input: float) -> Tuple[float, float, float, np.ndarray, np.ndarray,
                                                                      np.ndarray]:
    """
    As with band intervals, takes it all the way to Nyquist

    :param scale_order_input: Nth order specification
    :param frequency_low_input: lowest frequency of interest
    :param frequency_sample_rate_input: sample rate
    :return: band order Nth, number of cycles M, quality factor, geometric center frequency, start frequency,
        end frequency
    """

    order_Nth, scale_base, scale_band_number, \
    frequency_ref, frequency_center_algebraic, frequency_center_geometric, \
    frequency_start, frequency_end = \
        band_frequencies_nyquist(frequency_order_input=scale_order_input,
                                 frequency_base_input=Slice.G2,
                                 frequency_ref_input=Slice.F1,
                                 frequency_low_input=frequency_low_input,
                                 frequency_sample_rate_input=frequency_sample_rate_input)
    cycles_M, quality_Q = wavelet_MQ_from_N(order_Nth)

    return order_Nth, cycles_M, quality_Q, frequency_center_geometric, frequency_start, frequency_end


def cqt_frequency_bands_g2f1(scale_order_input: float,
                             frequency_low_input: float,
                             frequency_sample_rate_input: float,
                             is_power_2: bool = True) -> Tuple[float, float, float, float, float]:
    """
    CQT frequency bands G2 - F1

    :param scale_order_input: Nth order specification
    :param frequency_low_input: lowest frequency of interest
    :param frequency_sample_rate_input: sample rate
    :param is_power_2: power of two approximation
    :return: CQT points hop minimum value, frequency center minimum value in Hz, number of bins for scale,
        band order Nth, CQT maximum points per segment
    """
    order_Nth, cycles_M, quality_Q, frequency_hz_center, frequency_hz_start, frequency_hz_end = \
        frequency_bands_g2f1(scale_order_input, frequency_low_input, frequency_sample_rate_input)

    scale_number_bins_0 = int(len(frequency_hz_center))

    hann_bandwidth = 1.50018310546875
    _, q_gabor = wavelet_MQ_from_N(order_Nth)
    threshold = frequency_hz_center * (1 + 0.5 * hann_bandwidth / q_gabor)  # > frequency_sample_rate_hz / 2.0:

    # Remember frequency order is inverted because solution is in periods.
    idn = np.argmax(threshold < 0.9*frequency_sample_rate_input/2.0)
    scale_number_bins = int(len(frequency_hz_center[idn:]))

    frequency_hz_center_min = np.min(frequency_hz_center)
    cqt_points_hop_min = int(2**(np.floor(scale_number_bins/order_Nth)-1.))
    cqt_points_per_seg_max, _, _ = wavelet_support(order_Nth,
                                                   frequency_hz_center_min,
                                                   frequency_sample_rate_input,
                                                   is_power_2)
    return cqt_points_hop_min, frequency_hz_center_min, scale_number_bins, order_Nth, cqt_points_per_seg_max


def wavelet_scale_morlet2(band_order_Nth: float,
                          scale_frequency_center_hz: float,
                          frequency_sample_rate_hz: float) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Nondimensional scale for canonical Morlet wavelet

    :param band_order_Nth: Nth order of constant Q bands
    :param scale_frequency_center_hz: scale frequency in hz
    :param frequency_sample_rate_hz: sample rate in hz
    :return: floats or np.ndarray
    """
    cycles_M, _ = wavelet_MQ_from_N(band_order_Nth)
    scale_atom = cycles_M*frequency_sample_rate_hz/scale_frequency_center_hz/(2. * np.pi)
    return scale_atom, cycles_M


def wavelet_inputs_morlet2(band_order_Nth: float,
                           time_s: np.ndarray,
                           offset_time_s: float,
                           scale_frequency_center_hz: float,
                           frequency_sample_rate_hz: float) -> Tuple[float, float, float]:
    """
    Adds scaled time-shifted time

    :param band_order_Nth: Nth order of constant Q bands
    :param time_s: array with timestamps of signal in seconds
    :param offset_time_s: offset time in seconds
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :return: time shifted value (float), molet2 scale (float), cycles of M (float)
    """

    xtime_shifted = frequency_sample_rate_hz*(time_s-offset_time_s)
    scale_morlet2, cycles_M = wavelet_scale_morlet2(band_order_Nth, scale_frequency_center_hz, frequency_sample_rate_hz)

    return xtime_shifted, scale_morlet2, cycles_M
