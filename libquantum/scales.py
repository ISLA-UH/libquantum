import numpy as np

"""
CREATED: 20200901
Construct standardized scales
MAG LAST UPDATED:
"""

""" Smallest number for 64-bit floats. Deploy to avoid division by zero or log zero singularities"""
EPSILON = np.finfo(np.float64).eps
MICROS_TO_S = 1E-6
MICROS_TO_MILLIS = 1E-3
KPA_TO_PA = 1E3
DEGREES_TO_M = 111.
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


def planck_scale_s(scale_order):
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
                                    frequency_sample_rate_input=16000)


def band_periods_nyquist(scale_order_input: float, scale_base_input: float,
                         scale_ref_input: float,
                         scale_sample_interval_input: float, scale_high_input: float) -> \
        (float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Evaluate Standard Logarithmic Interval Time Parameters: ALWAYS USE SECONDS
    scale = pseudo-period
    scale_sample_interval_input: float
        Sample interval in seconds, sets lowest frequency
    scale_high_input: float
        Highest center scale of interest
    """
    scale_nyquist_input = 2*scale_sample_interval_input
    return band_intervals_periods(scale_order_input, scale_base_input,
                                  scale_ref_input,
                                  scale_nyquist_input, scale_high_input)


def band_frequencies_nyquist(frequency_order_input: float, frequency_base_input: float,
                             frequency_ref_input: float,
                             frequency_low_input: float, frequency_sample_rate_input: float) -> \
        (float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Evaluate Standard Logarithmic Interval Time Parameters: ALWAYS USE HZ
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
    return scale_order, scale_base, -scale_band_number, \
           frequency_ref, frequency_center_algebraic, frequency_center_geometric, \
           frequency_start, frequency_end


def band_frequencies_low_high(frequency_order_input: float, frequency_base_input: float,
                              frequency_ref_input: float,
                              frequency_low_input: float,
                              frequency_high_input: float,
                              frequency_sample_rate_input: float) -> \
        (float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Evaluate Standard Logarithmic Interval Time Parameters: ALWAYS USE HZ
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
    return scale_order, scale_base, -scale_band_number, \
           frequency_ref, frequency_center_algebraic, frequency_center_geometric, \
           frequency_start, frequency_end


def band_intervals_periods(scale_order_input: float, scale_base_input: float,
                           scale_ref_input: float,
                           scale_low_input: float,
                           scale_high_input: float) -> \
        (float, float, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
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
    scale_center: numpy ndarray
        Center band scale
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
        # print('Specified Base G3 = 10.**(3./10.) Meets ISO3 Preferred Series')
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
        # print('Specified Order = ' + repr(scale_order) + ' Meets ISO3 Preferred Series')
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

    return scale_order, scale_base, scale_band_number, \
           scale_ref, scale_center_algebraic, scale_center_geometric, \
           scale_start, scale_end


"""
Quantum wavelet specifications
"""


def wavelet_MQ_from_N(band_order_Nth: float):
    """
    Compute the quality factor Q and multiplier M for a specified band order N
    N is THE quantization parameter for the binary constant Q wavelet filters
    :param band_order_Nth: Band order, must be > 0.75 or reverts to N=3
    :return: float, float
    """
    if band_order_Nth < 0.7:
        # raise TypeError('N<0.7 specified, using N = {}.'.format(str(3)))
        print('N<0.7 specified, using N = ', 3)
        band_order_Nth = 3.
    order_bandedge = 2 ** (1. / 2. / band_order_Nth)  # kN in Garces 2013
    order_scaled_bandwidth = order_bandedge - 1. / order_bandedge
    quality_factor_Q = 1./order_scaled_bandwidth  # Exact for Nth octave bands
    cycles_M = quality_factor_Q*2*np.sqrt(np.log(2))  # Exact, from -3dB points
    return cycles_M, quality_factor_Q


def wavelet_NMQ_from_Q(quality_factor_Q: float):
    """
    For a specified Q, estimate order N and recompute exact Q and number of cycles M from N
    :param quality_factor_Q: number of oscillations with significant amplitude in the frame
    :return: float, float, float
    """
    band_order_Nth = quality_factor_Q/np.sqrt(2)  # Approximate
    cycles_M, quality_factor_Q = wavelet_MQ_from_N(band_order_Nth)
    return band_order_Nth, cycles_M, quality_factor_Q


def wavelet_NMQ_from_M(cycles_M: float):
    """
    For a specified M, estimate order N and recompute exact Q and number of cycles M from N
    :param quality_factor_Q: number of oscillations with significant amplitude in the frame
    :return: float, float, float
    """
    quality_factor_Q = cycles_M/(2*np.sqrt(np.log(2)))  # Exact, from -3dB/half bit points
    band_order_Nth, cycles_M, quality_factor_Q = wavelet_NMQ_from_Q(quality_factor_Q)
    return band_order_Nth, cycles_M, quality_factor_Q


def wavelet_support(band_order_Nth : float,
                    scale_frequency_center_hz: float,
                    frequency_sample_rate_hz: float,
                    is_power_2: bool=True):
    """
    Compact support for Gabor wavelet
    :param band_order_Nth: band order
    :param scale_frequency_center_hz: scale frequency in hz
    :param is_power_2: power of two approximation
    :return: int, float, float
    """
    cycles_M, _ = wavelet_MQ_from_N(band_order_Nth)
    scale_frame_T_s = cycles_M / scale_frequency_center_hz   # Lifetime Tn, in seconds, M/fc
    # scale_lifetime_T_horn_s = scale_frame_T_s / (2. * np.pi)  # Lifetime per cycle, M/omega
    scale_atom = frequency_sample_rate_hz*scale_frame_T_s / (2. * np.pi)  # Canonical Gabor atom scale
    nominal_points = np.floor(scale_frame_T_s*frequency_sample_rate_hz)

    floor_switch = 0.8  # Theoretical foundation, empirical implementation
    # TODO: Will this work with vector inputs?
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
                  sig_duration_s: float):
    cycles_M, _ = wavelet_MQ_from_N(band_order_Nth)
    scale_time_s = sig_duration_s/cycles_M
    scale_frequency_hz = 1/scale_time_s
    return scale_time_s, scale_frequency_hz


def frequency_bands_g2f1(scale_order_input: float,
                         frequency_low_input: float,
                         frequency_sample_rate_input: float) -> \
        (float, float, float, np.ndarray, np.ndarray, np.ndarray):
    """
    As with band intervals, takes it all the way to Nyquist
    :param scale_order_input:
    :param frequency_low_input:
    :param frequency_sample_rate_input:
    :return:
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
                             is_power_2: bool = True) -> \
        (float, float, float, float, float):

    order_Nth, cycles_M, quality_Q, frequency_hz_center, frequency_hz_start, frequency_hz_end = \
        frequency_bands_g2f1(scale_order_input, frequency_low_input, frequency_sample_rate_input)

    scale_number_bins_0 = int(len(frequency_hz_center))

    # TODO: Document resolution of bandwidth discrepancy
    hann_bandwidth = 1.50018310546875
    _, q_gabor = wavelet_MQ_from_N(order_Nth)
    # From Librosa/filters: threshold =  freq[-1] * (1 + 0.5 * window_bandwidth(window) / Q)
    # > frequency_sample_rate_hz / 2.0:
    threshold = frequency_hz_center * (1 + 0.5 * hann_bandwidth / q_gabor)  # > frequency_sample_rate_hz / 2.0:
    # print('threshold:', threshold)
    # print(scale_number_bins_0)
    # Remember frequency order is inverted because solution is in periods.
    idn = np.argmax(threshold < 0.9*frequency_sample_rate_input/2.0)
    scale_number_bins = int(len(frequency_hz_center[idn:]))

    frequency_hz_center_min = np.min(frequency_hz_center)
    cqt_points_hop_min = int(2**(np.floor(scale_number_bins/order_Nth)-1.))
    # cqt_points_hop_min = int(2**(np.ceil(scale_number_bins/order_Nth)))
    cqt_points_per_seg_max, _, _ = wavelet_support(order_Nth,
                                                   frequency_hz_center_min,
                                                   frequency_sample_rate_input,
                                                   is_power_2)
    return cqt_points_hop_min, frequency_hz_center_min, scale_number_bins, order_Nth, cqt_points_per_seg_max


# def spect_scale(sample_rate_hz, order_base, order_Nth, scale_ref_input,
#                 soi_fundamental_frequency_hz, soi_fundamental_cycles_per_window):
#     # TODO: Where does this fit?
#     frequency_hz_minimum = soi_fundamental_frequency_hz/4.   # Octave below fundamental
#
#     time_window_duration = soi_fundamental_cycles_per_window/soi_fundamental_frequency_hz  # 1/spectral resolution for FFT
#
#     scale_ref, scale_base, scale_order, scale_band_number, scale_band_exponent, \
#     scale_center, scale_start, scale_end = band_intervals(scale_ref_input, order_base, order_Nth,
#                                                           frequency_hz_minimum, sample_rate_hz/2.)
#     scale_number_bins = int(len(scale_band_number))
#     scale_nth_octave = int(scale_order)
#
#     print('Scale number octaves:', np.floor(scale_number_bins/order_Nth))
#     print('Scale number bins:', scale_number_bins)
#     # print('Scale band:', scale_band_number)
#     # print('Scale center:', scale_center)
#
#     points_per_seg = time_window_duration*sample_rate_hz
#     points_per_seg_base2 = int(2**np.ceil(np.log2(points_per_seg)))  # 2**n points
#     time_per_seg_s_base2 = points_per_seg_base2/sample_rate_hz
#
#     print('NFFT segment, base2:', points_per_seg_base2)
#     print('NFFT segment base2 time, s:', time_per_seg_s_base2)
#     # TODO: find code where I fixed this
#     points_hop = int(points_per_seg_base2/8)
#     # TODO: Here it is
#     cqt_points_hop = int(2**(np.floor(scale_number_bins/order_Nth)-1.))
#     points_overlap = points_per_seg_base2 - points_hop
#     return scale_nth_octave, scale_number_bins, points_hop, points_overlap, points_per_seg_base2, frequency_hz_minimum


def wavelet_scale_morlet2(band_order_Nth: float,
                          scale_frequency_center_hz: float,
                          frequency_sample_rate_hz: float):
    """
    Nondimensional scale for canonical Morlet wavelet
    :param band_order_Nth: band order
    :param scale_frequency_center_hz: scale frequency in hz
    :param frequency_sample_rate_hz: sample rate in hz
    :return: floats or np.ndarray
    """
    cycles_M, _ = wavelet_MQ_from_N(band_order_Nth)
    scale_atom = cycles_M*frequency_sample_rate_hz/scale_frequency_center_hz/(2. * np.pi)
    return scale_atom, cycles_M


def wavelet_inputs_morlet2(band_order_Nth: float, time_s: np.ndarray, offset_time_s: float,
                           scale_frequency_center_hz: float, frequency_sample_rate_hz: float):
    """
    Adds scaled time-shifted time
    :param band_order_Nth:
    :param time_s:
    :param offset_time_s:
    :param scale_frequency_center_hz:
    :param frequency_sample_rate_hz:
    :return: floats or np.ndarrays
    """

    xtime_shifted = frequency_sample_rate_hz*(time_s-offset_time_s)
    scale_morlet2, cycles_M = wavelet_scale_morlet2(band_order_Nth, scale_frequency_center_hz, frequency_sample_rate_hz)

    return xtime_shifted, scale_morlet2, cycles_M


# if __name__ == "__main__":
#
#     print('First Planck band, with minimum pseudo-period bandwidth = ', Slice.T0S)
#     for scale_order in [0.75, 1, 1.5, 3, 6, 12]:
#         planck_center, planck_low, planck_high, quality_Q = planck_scale_s(scale_order)
#         print('\nOrder_N:', scale_order)
#         print('Number of Oscillations, Q:', quality_Q)
#         # print('%1s%12s%12s' % ('Scaled Center', 'Lower', 'Upper'))
#         # print('%12.4e%12.4e%12.4e' % (planck_center/Slice.T0S, planck_low/Slice.T0S, planck_high/Slice.T0S))
#         print('%1s%12s%12s' % ('Time_s Center', 'Lower', 'Upper'))
#         print('%12.4e%12.4e%12.4e' % (planck_center, planck_low, planck_high))
#         print('Scaled bandwidth:', (planck_high-planck_low)/Slice.T0S)
#
#
#     print('\n Scale Period up to Nyquist')
#     export.print_scales_to_screen(scale_order_input=1,
#                                   scale_base_input=Slice.G3,
#                                   scale_ref_input=Slice.T0S,
#                                   scale_sample_interval_input=1E-43,
#                                   scale_high_input=1E-41)
#
#     print('\n Scale Frequency up to Nyquist, G3')
#     export.print_frequencies_to_screen(frequency_order_input=1,
#                                        frequency_base_input=Slice.G3,
#                                        frequency_ref_input=Slice.F1,
#                                        frequency_low_input=1/1E-41,
#                                        frequency_sample_rate_input=1/1E-43)
#
#     print('\n Scale Frequency up to Nyquist, G2')
#     export.print_frequencies_to_screen(frequency_order_input=1,
#                                        frequency_base_input=Slice.G2,
#                                        frequency_ref_input=Slice.F1,
#                                        frequency_low_input=1/1E-41,
#                                        frequency_sample_rate_input=1/1E-43)
#
#     print('\n *** Equal tempered scale re A4 = 440 Hz')
#     export.print_frequencies_to_screen(frequency_order_input=12,
#                                        frequency_base_input=Slice.G2,
#                                        frequency_ref_input=440,
#                                        frequency_low_input=16.35,
#                                        frequency_sample_rate_input=16000)
