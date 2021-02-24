import numpy as np
import scipy.signal as signal
from libquantum import scales
from libquantum import utils

"""LAST UPDATED: 20201021 MAG
The purpose of this code is to construct quantized, standardized information packets
using binary metrics. Based on Garces (2020). 
Cleaned up and compartmentalized for debugging"""


def chirp_complex(band_order_Nth: float,
                  time_s: np.ndarray,
                  offset_time_s: float,
                  scale_frequency_center_hz: float,
                  frequency_sample_rate_hz: float,
                  index_shift: float = 0,
                  scale_base: float = scales.Slice.G2):
    """
    # TODO: This is the core function. Optimize this!!
    # TODO: Construct matrix output when frequency_center is a vector
    Quantum chirp for specified band_order_Nth and arbitrary time duration
    Unscaled, to be used by both Dictionary 1 and Dictionary 2
    :param band_order_Nth: Nth order of constant Q bands
    :param time_s: time in seconds, duration should be greater than or equal to M/fc
    :param offset_time_s: offset time in seconds, should be between min and max of time_s
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate on Hz
    :param index_shift: Redshift = -1, Blueshift = +1, None=0
    :param scale_base: G2 or G3
    :return: waveform_complex, time_shifted_s
    """

    xtime_shifted = chirp_time(time_s, offset_time_s, frequency_sample_rate_hz)
    time_shifted_s = xtime_shifted/frequency_sample_rate_hz

    # Fundamental chirp parameters
    # TODO: Accept range of frequencies, construct matrix
    cycles_M, quality_factor_Q, gamma = \
        chirp_MQG_from_N(band_order_Nth, index_shift, scale_base)
    scale_atom = chirp_scale(cycles_M, scale_frequency_center_hz, frequency_sample_rate_hz)
    p_complex = chirp_p_complex(scale_atom, gamma, index_shift)
    amp_dict_0, amp_dict_1 = chirp_amplitude(scale_atom, gamma, index_shift)

    # TODO: Write function to return uncertainty
    # time_std = scale_atom/np.sqr(2)
    # time_std_s = time_std/frequency_sample_rate_hz
    wavelet_gauss = np.exp(-p_complex * xtime_shifted**2)
    wavelet_gabor = wavelet_gauss * np.exp(1j * cycles_M*xtime_shifted/scale_atom)

    return wavelet_gabor, time_shifted_s, amp_dict_0, amp_dict_1


def chirp_spectrum(frequency_hz,
                   offset_time_s,
                   band_order_Nth,
                   scale_frequency_center_hz,
                   frequency_sample_rate_hz,
                   index_shift: float = 0,
                   scale_base: float = scales.Slice.G2):
    """
    Spectrum of quantum wavelet for specified band_order_Nth and arbitrary time duration
    :param frequency_hz: frequency range below Nyquist
    :param offset_time_s: time of wavelet centroid
    :param band_order_Nth: Nth order of constant Q bands
    :param scale_frequency_center_hz: band center frequency in Hz
    :param frequency_sample_rate_hz: sample rate on Hz
    :param index_shift: TODO
    :param scale_base: TODO
    :return: Fourier transform of the Gabor atom
    """
    # TODO: Accept range of frequencies, construct matrix
    cycles_M, quality_factor_Q, gamma = \
        chirp_MQG_from_N(band_order_Nth, index_shift, scale_base)
    scale_atom = chirp_scale(cycles_M, scale_frequency_center_hz, frequency_sample_rate_hz)
    p_complex = chirp_p_complex(scale_atom, gamma, index_shift)

    angular_frequency_center = 2 * np.pi * scale_frequency_center_hz/frequency_sample_rate_hz
    angular_frequency = 2 * np.pi * frequency_hz/frequency_sample_rate_hz
    offset_phase = angular_frequency * frequency_sample_rate_hz * offset_time_s
    angular_frequency_shifted = angular_frequency - angular_frequency_center
    frequency_shifted_hz = angular_frequency_shifted*frequency_sample_rate_hz/(2*np.pi)
    # TODO: Write function to return uncertainty
    # frequency_std_hz = frequency_sample_rate_hz/(2*np.pi*scale_atom*np.sqr(2))

    spectrum_amplitude = np.sqrt(np.pi/p_complex)
    gauss_arg = 1./(4*p_complex)
    spectrum_gauss = np.exp(-gauss_arg * (angular_frequency_shifted * scale_atom) ** 2)
    # Phase shift from time offset
    spectrum_gabor = spectrum_amplitude * spectrum_gauss * np.exp(-1j * offset_phase)

    return spectrum_gabor, frequency_shifted_hz


def chirp_MQG_from_N(band_order_Nth: float,
                     index_shift: float = 0,
                     scale_base: float = scales.Slice.G2):
    """
    Compute the quality factor Q and multiplier M for a specified band order N
    N is THE quantization parameter for the binary constant Q wavelet filters
    :param band_order_Nth: Band order, must be > 0.75 or reverts to N=3
    :param index_shift: TODO
    :param scale_base: TODO
    :return: float, float
    """
    if band_order_Nth < 0.7:
        # raise TypeError('N<0.7 specified, using N = {}.'.format(str(3)))
        print('N<0.7 specified, using N = ', 3)
        band_order_Nth = 3.
    order_bandedge = scale_base ** (1. / 2. / band_order_Nth)  # kN in Garces 2013
    order_scaled_bandwidth = order_bandedge - 1. / order_bandedge
    quality_factor_Q = 1./order_scaled_bandwidth  # Exact for Nth octave bands
    # Gamma is M/(2Q)
    gamma = np.sqrt(np.log(2))*(1-np.log(2)*(index_shift/np.pi)**2)**(-0.5)
    cycles_M = 2*quality_factor_Q*gamma  # Exact, from 1/2 power points

    return cycles_M, quality_factor_Q, gamma


def chirp_scale(cycles_M: float,
                scale_frequency_center_hz: float,
                frequency_sample_rate_hz: float):
    """
    Nondimensional scale for canonical Morlet wavelet
    :param cycles_M: number of cycles per band period
    :param scale_frequency_center_hz: scale frequency in hz
    :param frequency_sample_rate_hz: sample rate in hz
    :return: floats or np.ndarray
    """
    scale_atom = cycles_M*frequency_sample_rate_hz/scale_frequency_center_hz/(2. * np.pi)
    return scale_atom


def chirp_scale_from_order(band_order_Nth: float,
                           scale_frequency_center_hz: float,
                           frequency_sample_rate_hz: float,
                           index_shift: float = 0,
                           scale_base: float = scales.Slice.G2):
    """
    Nondimensional scale for canonical Morlet wavelet
    :param cycles_M: number of cycles per band period
    :param scale_frequency_center_hz: scale frequency in hz
    :param frequency_sample_rate_hz: sample rate in hz
    :param index_shift: TODO
    :param scale_base: TODO
    :return: floats or np.ndarray
    """
    cycles_M, _, _ = chirp_MQG_from_N(band_order_Nth,
                                      index_shift,
                                      scale_base)
    scale_atom = cycles_M*frequency_sample_rate_hz/scale_frequency_center_hz/(2. * np.pi)
    return scale_atom


def chirp_uncertainty(scale_atom: np.ndarray,
                      frequency_sample_rate_hz: float,
                      gamma, index_shift):
    """
    Uncertainty
    :param scale_atom: TODO
    :param frequency_sample_rate_hz: sample rate in hz
    :param gamma_exact: from index_shift  TODO which one is correct, this one or the declaration?
    :param index_shift: TODO
    :return: floats or np.ndarrays
    """

    time_std_s = scale_atom/np.sqr(2)/frequency_sample_rate_hz
    angular_frequency_std = np.sqrt(1+(index_shift*gamma)**2)/scale_atom/np.sqr(2)
    angular_frequency_std_hz = frequency_sample_rate_hz*angular_frequency_std
    frequency_std_hz = angular_frequency_std_hz/2/np.pi

    return time_std_s, frequency_std_hz, angular_frequency_std_hz


def chirp_p_complex(scale_atom, gamma, index_shift):
    """
    Fundamental chirp variable
    :param scale_atom:
    :param gamma:
    :param index_shift:
    :return: p_complex
    """
    p_complex = (1 - 1j*index_shift*gamma/np.pi)/(2*scale_atom**2)
    return p_complex


def chirp_amplitude(scale_atom, gamma, index_shift):
    """
    Fundamental chirp variable
    :param scale_atom:
    :param gamma:
    :param index_shift:
    :return: p_complex
    """
    p_complex = chirp_p_complex(scale_atom, gamma, index_shift)
    amp_dict_0 = 1/np.pi**0.25 * 1/np.sqrt(scale_atom)
    amp_dict_1 = np.sqrt(np.abs(p_complex)/np.pi)
    return amp_dict_0, amp_dict_1


def chirp_time(time_s: np.ndarray,
               offset_time_s: float,
               frequency_sample_rate_hz: float):
    """
    Scaled time-shifted time
    :param time_s:
    :param offset_time_s:
    :param frequency_sample_rate_hz:
    :return: floats or np.ndarrays
    """
    xtime_shifted = frequency_sample_rate_hz*(time_s-offset_time_s)
    return xtime_shifted


def chirp_scales_from_duration(band_order_Nth : float,
                               sig_duration_s: float,
                               index_shift: float = 0,
                               scale_base: float = scales.Slice.G2):
    """

    :param band_order_Nth:
    :param sig_duration_s:
    :param index_shift:
    :param scale_base:
    :return:
    """
    cycles_M, _, _ = chirp_MQG_from_N(band_order_Nth,
                                      index_shift,
                                      scale_base)
    scale_time_s = sig_duration_s/cycles_M
    scale_frequency_hz = 1/scale_time_s
    return scale_time_s, scale_frequency_hz


def chirp_frequency_bands(scale_order_input: float,
                          frequency_low_input: float,
                          frequency_sample_rate_input: float,
                          frequency_high_input: float,
                          index_shift: float = 0,
                          frequency_ref: float = scales.Slice.F1,
                          scale_base: float = scales.Slice.G2) -> \
        (float, float, float, float, np.ndarray, np.ndarray, np.ndarray):
    """

    :param scale_order_input:
    :param frequency_low_input:
    :param frequency_sample_rate_input:
    :param frequency_high_input:
    :param index_shift:
    :param frequency_ref:
    :param scale_base:
    :return:
    """

    order_Nth, scale_base, scale_band_number, \
    frequency_ref, frequency_center_algebraic, frequency_center_geometric, \
    frequency_start, frequency_end = \
        scales.band_frequencies_low_high(frequency_order_input=scale_order_input,
                                         frequency_base_input=scale_base,
                                         frequency_ref_input=frequency_ref,
                                         frequency_low_input=frequency_low_input,
                                         frequency_high_input=frequency_high_input,
                                         frequency_sample_rate_input=frequency_sample_rate_input)
    cycles_M, quality_Q, gamma = chirp_MQG_from_N(order_Nth, index_shift, scale_base)

    return order_Nth, cycles_M, quality_Q, gamma, frequency_center_geometric, frequency_start, frequency_end


def chirp_centered_4cwt(band_order_Nth: float,
                        sig_or_time: np.ndarray,
                        scale_frequency_center_hz: float,
                        frequency_sample_rate_hz: float,
                        index_shift: float = 0,
                        scale_base: float = scales.Slice.G2,
                        dictionary_type = "norm"):
    """
    Gabor atoms for CWT computation centered on the duration of signal
    :param sig_or_time: time or time series, wavelet matches this duration
    :param band_order_Nth: Nth order of constant Q bands
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate is Hz
    :param index_shift: TODO
    :param scale_base: G2 or G3
    :param dictionary_type: Canonical unit-norm ("norm") or unit spectrum ("spect")
    :return: waveform_complex, time_shifted_s
    """

    duration_points = len(sig_or_time)
    time_s = np.arange(duration_points)/frequency_sample_rate_hz
    offset_time_s = time_s[-1]/2.

    wavelet_gabor, time_centered_s, amp_dict_0, amp_dict_1 = \
        chirp_complex(band_order_Nth,
                      time_s, offset_time_s, scale_frequency_center_hz, frequency_sample_rate_hz,
                      index_shift, scale_base)

    if dictionary_type == "norm":
        wavelet_chirp = amp_dict_0*wavelet_gabor
    elif dictionary_type == "tone":
        wavelet_chirp = np.sqrt(2)*amp_dict_1*wavelet_gabor
    else:
        wavelet_chirp = amp_dict_1*wavelet_gabor

    return wavelet_chirp, time_centered_s


def cwt_chirp_complex(band_order_Nth: float,
                      sig_wf: np.ndarray,
                      frequency_low_hz: float,
                      frequency_sample_rate_hz: float,
                      frequency_high_hz: float = scales.Slice.F0,
                      cwt_type: str = "fft",
                      index_shift: float = 0,
                      frequency_ref: float = scales.Slice.F1,
                      scale_base: float = scales.Slice.G2,
                      dictionary_type: str = "norm"):
    """

    :param band_order_Nth:
    :param sig_wf:
    :param frequency_low_hz:
    :param frequency_sample_rate_hz:
    :param frequency_high_hz:
    :param cwt_type:
    :param index_shift:
    :param frequency_ref:
    :param scale_base:
    :param dictionary_type:
    :return:
    """

    wavelet_points = len(sig_wf)
    time_s = np.arange(wavelet_points)/frequency_sample_rate_hz

    if cwt_type == "morlet2":
        index_shift = 0

    # Planck frequency is absolute upper limit
    # print(frequency_high_hz, frequency_sample_rate_hz/2.)
    if frequency_high_hz > frequency_sample_rate_hz/2.:
        frequency_high_hz = frequency_sample_rate_hz/2.

    order_Nth, cycles_M, quality_Q, _,\
    frequency_cwt_hz_flipped, frequency_start_flipped, frequency_end_flipped = \
        chirp_frequency_bands(scale_order_input=band_order_Nth,
                              frequency_low_input=frequency_low_hz,
                              frequency_sample_rate_input=frequency_sample_rate_hz,
                              frequency_high_input=frequency_high_hz,
                              index_shift=index_shift,
                              frequency_ref=frequency_ref,
                              scale_base=scale_base)

    # frequency_cwt_hz_plot = np.append(frequency_start, frequency_end[-1])

    scale_points = len(frequency_cwt_hz_flipped)

    # TODO: cwt_flipped is not always set.  Flow of control needs to be updated to account for this.
    if cwt_type == "morlet2":
        scale_atom = chirp_scale(cycles_M, frequency_cwt_hz_flipped, frequency_sample_rate_hz)
        cwt = signal.cwt(sig_wf, signal.morlet2, scale_atom, w=cycles_M)
    # TODO: Optimize this as in signal.cwt. Conv can take matrices.

    elif cwt_type == "fft":
        # TODO: Can see "ghost" folding in fft, compared to "conv"
        sig_fft = np.fft.fft(sig_wf)
        cwt_flipped = np.empty((scale_points, wavelet_points), dtype=np.complex128)
        for ii in range(scale_points):
            atom, _ = chirp_centered_4cwt(band_order_Nth=order_Nth,
                                          sig_or_time=sig_wf,
                                          scale_frequency_center_hz=frequency_cwt_hz_flipped[ii],
                                          frequency_sample_rate_hz=frequency_sample_rate_hz,
                                          index_shift=index_shift,
                                          scale_base=scale_base,
                                          dictionary_type=dictionary_type)
            atom_fft = np.fft.fft(atom)
            cwt_raw = np.fft.ifft(sig_fft*np.conj(atom_fft))
            cwt_flipped[ii, :] = np.append(cwt_raw[wavelet_points//2:], cwt_raw[0:wavelet_points//2])

    elif cwt_type == "conv":
        cwt_flipped = np.empty((scale_points, wavelet_points), dtype=np.complex128)
        for ii in range(scale_points):
            atom, _ = chirp_centered_4cwt(band_order_Nth=order_Nth,
                                          sig_or_time=sig_wf,
                                          scale_frequency_center_hz=frequency_cwt_hz_flipped[ii],
                                          frequency_sample_rate_hz=frequency_sample_rate_hz,
                                          index_shift=index_shift,
                                          scale_base=scale_base,
                                          dictionary_type=dictionary_type)
            cwt_flipped[ii, :] = signal.convolve(sig_wf, np.conj(atom)[::-1], mode='same')
    else:
        print("Incorrect cwt_type specification in cwt_chirp_complex")

    # TODO: re above: cwt_flipped is not always set; must make sure this run if that's the case.
    # Time scales are increasing, which is the opposite of what is expected for the frequency. Flip.
    frequency_cwt_hz = np.flip(frequency_cwt_hz_flipped)
    cwt = np.flipud(cwt_flipped)
    cwt_bits = utils.log2epsilon(cwt)

    return cwt, cwt_bits, time_s, frequency_cwt_hz


def cwt_chirp_from_sig(sig_wf: np.ndarray,
                       frequency_sample_rate_hz: float,
                       band_order_Nth: float = 3,
                       cwt_type: str = "fft",
                       index_shift: float = 0,
                       frequency_ref: float = scales.Slice.F1,
                       scale_base: float = scales.Slice.G2,
                       dictionary_type: str = "norm"):
    """
    COMMENTS IN  PROGRESS
    :param band_order_Nth:
    :param sig_wf:
    :param frequency_sample_rate_hz:
    :param cwt_type:
    :param index_shift:
    :param frequency_ref:
    :param scale_base:
    :param dictionary_type:
    :return:
    """

    wavelet_points = len(sig_wf)
    duration_s = wavelet_points/frequency_sample_rate_hz
    _, min_frequency_hz = \
        chirp_scales_from_duration(band_order_Nth=band_order_Nth,
                                   sig_duration_s=duration_s,
                                   index_shift=index_shift,
                                   scale_base=scale_base)

    cwt, cwt_bits, time_s, frequency_cwt_hz = \
        cwt_chirp_complex(band_order_Nth=band_order_Nth,
                          sig_wf=sig_wf,
                          frequency_low_hz=min_frequency_hz,
                          frequency_sample_rate_hz=frequency_sample_rate_hz,
                          frequency_high_hz=frequency_sample_rate_hz/2.,
                          cwt_type=cwt_type,
                          index_shift=index_shift,
                          frequency_ref=frequency_ref,
                          scale_base=scale_base,
                          dictionary_type=dictionary_type)

    return cwt, cwt_bits, time_s, frequency_cwt_hz
