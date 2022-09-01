"""
This module contains functions to construct quantized, standardized information packets using binary metrics.
No-chirp/sweep (index_shift=0, variable removed), simplified for the base stockwell transform.
"""

import numpy as np
import scipy.signal as signal
from libquantum import scales
from libquantum import utils
from typing import Tuple, Union

"""
The purpose of this code is to construct quantized, standardized information packets
using binary metrics. Based on Garces (2020). 
Cleaned up and compartmentalized for debugging
"""


def cycles_from_order(band_order_Nth: float) -> float:
    """
    Compute the number of cycles M for a specified band order N
    N is the quantization parameter for the constant Q wavelet filters

    :param band_order_Nth: Band order, must be > 0.75 or reverts to N=3
    :return: cycles_M, number of cycled per normalized angular frequency
    """
    if band_order_Nth < 0.7:
        print('N<0.7 specified, using N = ', 3)
        band_order_Nth = 3.
    cycles_M = 3.*np.pi/4.*band_order_Nth  # Practical implementation, estimated from 1/2 power points
    return cycles_M


def scale_from_frequency_hz(band_order_Nth: float,
                            scale_frequency_center_hz: Union[np.ndarray, float],
                            frequency_sample_rate_hz: float) -> \
        Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Nondimensional scale and angular frequency for canonical Gabor/Morlet wavelet

    :param cycles_M: number of cycles per band period
    :param scale_frequency_center_hz: scale frequency in hz
    :param frequency_sample_rate_hz: sample rate in hz
    :return: scale_atom, scaled angular frequency
    """
    cycles_M = cycles_from_order(band_order_Nth)
    scale_angular_frequency = 2. * np.pi * scale_frequency_center_hz/frequency_sample_rate_hz
    scale_atom = cycles_M/scale_angular_frequency
    return scale_atom, scale_angular_frequency


def wavelet_amplitude(scale_atom: Union[np.ndarray, float]) -> \
        Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Return chirp amplitude
    amp_dict_canonical = return unit integrated power and spectral energy. Good for math, ref William et al. 1991.
    amp_dict_unit_spectrum = return unit peak spectrum; for practical implementation.
    amp_dict_unity = 1. Default (no scaling), for testing and validation against real and imaginary wavelets.

    :param scale_atom: atom/logon scale
    :return: amp_canonical, amp_unit_spectrum
    """

    amp_canonical = (np.pi * scale_atom ** 2) ** (-1/4)
    amp_unit_spectrum = (4*np.pi*scale_atom**2) ** (-1/4) * amp_canonical
    return amp_canonical, amp_unit_spectrum


def amplitude_convert_norm_to_spect(scale_atom: Union[np.ndarray, float]) -> \
        Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Return chirp amplitude
    amp_dict_canonical = return unit integrated power and spectral energy. Good for math, ref William et al. 1991.
    amp_dict_unit_spectrum = return unit peak spectrum; for practical implementation.
    amp_dict_unity = 1. Default (no scaling), for testing and validation against real and imaginary wavelets.

    :param scale_atom: atom/logon scale
    :return: amp_canonical, amp_unit_spectrum
    """

    amp_canonical = (np.pi * scale_atom ** 2) ** (-1/4)
    amp_unit_spectrum = (4*np.pi*scale_atom**2) ** (-1/4) * amp_canonical
    amp_norm2spect = amp_unit_spectrum/amp_canonical

    return amp_norm2spect


def wavelet_time(time_s: np.ndarray,
                 offset_time_s: float,
                 frequency_sample_rate_hz: float) -> np.ndarray:
    """
    Scaled time-shifted time

    :param time_s: array with time
    :param offset_time_s: offset time in seconds
    :param frequency_sample_rate_hz: sample rate in Hz
    :return: numpy array with time-shifted time
    """
    xtime_shifted = frequency_sample_rate_hz*(time_s-offset_time_s)
    return xtime_shifted


def wavelet_complex(band_order_Nth: float,
                    time_s: np.ndarray,
                    offset_time_s: float,
                    scale_frequency_center_hz: Union[np.ndarray, float],
                    frequency_sample_rate_hz: float) -> \
        Tuple[np.ndarray, np.ndarray, Union[np.ndarray, float], Union[np.ndarray, float],
              Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float]]:

    """
    Quantized atom for specified band_order_Nth and arbitrary time duration.
    Unscaled, to be modified by the dictionary type and use case.
    Returns a frequency x time dimension wavelet vector

    :param band_order_Nth: Nth order of constant Q bands
    :param time_s: time in seconds, duration should be greater than or equal to M/fc
    :param offset_time_s: offset time in seconds, should be between min and max of time_s
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate on Hz
    :param index_shift: Redshift = -1, Blueshift = +1, None=0
    :param scale_base: G2 or G3
    :return: waveform_complex, time_shifted_s
    """

    # Center and nondimensionalize time
    xtime_shifted = wavelet_time(time_s, offset_time_s, frequency_sample_rate_hz)

    # Nondimensional chirp parameters
    scale_atom, scale_angular_frequency = \
        scale_from_frequency_hz(band_order_Nth, scale_frequency_center_hz, frequency_sample_rate_hz)

    if np.isscalar(scale_atom):
        # Single frequency input
        xtime = xtime_shifted
        scale = scale_atom
        omega = scale_angular_frequency
    else:
        # Convert scale, frequency and time vectors to [frequency x time] matrices
        xtime = np.tile(xtime_shifted, (len(scale_atom), 1))
        scale = np.tile(scale_atom, (len(xtime_shifted), 1)).T
        omega = np.tile(scale_angular_frequency, (len(xtime_shifted), 1)).T

    # Base wavelet with unit absolute amplitude.
    # Note centered imaginary wavelet (sine) does not reach unity because of Gaussian envelope.
    wavelet_gabor = np.exp(-0.5*(xtime/scale)**2) * np.exp(1j*omega*xtime)
    amp_canonical, amp_unit_spectrum = wavelet_amplitude(scale)

    return wavelet_gabor, xtime_shifted, scale_angular_frequency, scale, omega, amp_canonical, amp_unit_spectrum


def wavelet_centered_4cwt(band_order_Nth: float,
                          duration_points: int,
                          scale_frequency_center_hz: Union[np.ndarray, float],
                          frequency_sample_rate_hz: float,
                          dictionary_type: str = "norm") -> \
        Tuple[np.ndarray, np.ndarray, Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float]]:

    """
    Gabor atoms for CWT computation centered on the duration of signal

    :param duration_points: number of points in the signal
    :param band_order_Nth: Nth order of constant Q bands
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate is Hz
    :param dictionary_type: Canonical unit-norm ("norm"), unit spectrum ("spect"), or unit modulus ("unit")
    :return: waveform_complex, time_shifted_s
    """

    time_s = np.arange(duration_points)/frequency_sample_rate_hz
    offset_time_s = time_s[-1]/2.

    wavelet_gabor, xtime_shifted, scale_angular_frequency, scale, omega, amp_canonical, amp_unit_spectrum = \
        wavelet_complex(band_order_Nth, time_s, offset_time_s, scale_frequency_center_hz, frequency_sample_rate_hz)

    if dictionary_type == "norm":
        amp = amp_canonical
    elif dictionary_type == "spect":
        amp = amp_unit_spectrum
    elif dictionary_type == "unit":
        if np.isscalar(scale):
            amp = 1.
        else:
            amp = np.ones(scale.shape)
    else:
        amp = amp_canonical

    wavelet_chirp = amp * wavelet_gabor
    time_centered_s = xtime_shifted/frequency_sample_rate_hz

    return wavelet_chirp, time_centered_s, scale, omega, amp


def scale_frequency_bands(scale_order_input: float,
                          frequency_low_input: float,
                          frequency_sample_rate_input: float,
                          frequency_high_input: float,
                          frequency_ref: float = scales.Slice.F1,
                          scale_base: float = scales.Slice.G2) -> Tuple[float, float,
                                                                        np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate frequency bands for chirp

    :param scale_order_input: Nth order specification
    :param frequency_low_input: lowest frequency of interest
    :param frequency_sample_rate_input: sample rate
    :param frequency_high_input: highest frequency of interest
    :param index_shift: index of shift
    :param frequency_ref: reference frequency
    :param scale_base: positive reference Base G > 1. Default is G2
    :return: Nth order, cycles M, geometric center of frequencies, start frequency,
        end frequency
    """

    # Order Nth may be overwitten is a silly request is made
    order_Nth, scale_base, scale_band_number, \
    frequency_ref, frequency_center_algebraic, frequency_center_geometric, \
    frequency_start, frequency_end = \
        scales.band_frequencies_low_high(frequency_order_input=scale_order_input,
                                         frequency_base_input=scale_base,
                                         frequency_ref_input=frequency_ref,
                                         frequency_low_input=frequency_low_input,
                                         frequency_high_input=frequency_high_input,
                                         frequency_sample_rate_input=frequency_sample_rate_input)
    cycles_M = cycles_from_order(scale_order_input)

    return order_Nth, cycles_M, frequency_center_geometric, frequency_start, frequency_end


def cwt_complex_inferno(band_order_Nth: float,
                        sig_wf: np.ndarray,
                        frequency_sample_rate_hz: float,
                        frequency_low_hz: float,
                        frequency_high_hz: float = None,
                        cwt_type: str = "fft",
                        frequency_ref: float = scales.Slice.F1,
                        scale_base: float = scales.Slice.G2,
                        dictionary_type: str = "norm") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate CWT for chirp

    :param band_order_Nth: Nth order of constant Q bands
    :param sig_wf: array with input signal
    :param frequency_low_hz: lowest frequency in Hz
    :param frequency_sample_rate_hz: sample rate in Hz
    :param frequency_high_hz: highest frequency in Hz
    :param cwt_type: one of "fft", or "morlet2". Default is "fft"
    :param index_shift: index of shift. Default is 0.0
    :param frequency_ref: reference frequency in Hz. Default is F1
    :param scale_base: G2 or G3. Default is G2
    :param dictionary_type: Canonical unit-norm ("norm") or unit spectrum ("spect"). Default is "norm"
    :return: cwt, cwt_bits, time_s, frequency_cwt_hz
    """

    wavelet_points = len(sig_wf)
    time_cwt_s = np.arange(wavelet_points)/frequency_sample_rate_hz

    if frequency_high_hz is None:
        frequency_high_hz = frequency_sample_rate_hz/2.

    order_Nth, cycles_M, frequency_cwt_hz_flipped, frequency_start_flipped, frequency_end_flipped = \
        scale_frequency_bands(scale_order_input=band_order_Nth,
                              frequency_low_input=frequency_low_hz,
                              frequency_sample_rate_input=frequency_sample_rate_hz,
                              frequency_high_input=frequency_high_hz,
                              frequency_ref=frequency_ref,
                              scale_base=scale_base)

    scale_points = len(frequency_cwt_hz_flipped)

    cw_complex, _, _, _, amp = \
        wavelet_centered_4cwt(band_order_Nth=order_Nth,
                              duration_points=wavelet_points,
                              scale_frequency_center_hz=frequency_cwt_hz_flipped,
                              frequency_sample_rate_hz=frequency_sample_rate_hz,
                              dictionary_type=dictionary_type)

    if cwt_type == "morlet2":
        # Use scipy morlet2
        scale_atom, _ = \
            scale_from_frequency_hz(band_order_Nth=order_Nth,
                                    frequency_sample_rate_hz=frequency_sample_rate_hz,
                                    scale_frequency_center_hz=frequency_cwt_hz_flipped)

        cwt_flipped = signal.cwt(data=sig_wf, wavelet=signal.morlet2,
                                 widths=scale_atom,
                                 w=cycles_M,
                                 dtype=np.complex128)
        if dictionary_type == 'spect':
            cwt_amp_norm2spec = amplitude_convert_norm_to_spect(scale_atom=scale_atom)
            # Convert to 2d matrix
            spec_scale = np.tile(cwt_amp_norm2spec, (wavelet_points, 1)).T
            cwt_flipped *= spec_scale

    else:
        # Convolution using the fft method
        # Convert to a 2d matrix
        sig_wf_2d = np.tile(sig_wf, (scale_points, 1))
        # Flip time
        cw_complex_fliplr = np.fliplr(cw_complex)
        cwt_flipped = signal.fftconvolve(sig_wf_2d, np.conj(cw_complex_fliplr), mode='same', axes=-1)

    # Time scales are increasing, which is the opposite of what is expected for the frequency. Flip.
    frequency_cwt_hz = np.flip(frequency_cwt_hz_flipped)
    cwt = np.flipud(cwt_flipped)

    return frequency_cwt_hz, time_cwt_s, cwt


def cwt_complex_any_scale_pow2(band_order_Nth: float,
                               sig_wf: np.ndarray,
                               frequency_sample_rate_hz: float,
                               frequency_cwt_hz: np.ndarray,
                               cwt_type: str = "fft",
                               dictionary_type: str = "norm") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate CWT for chirp

    :param band_order_Nth: Nth order of constant Q bands
    :param sig_wf: array with input signal
    :param frequency_sample_rate_hz: sample rate in Hz, ordered from low to high frequency
    :param frequency_cwt_hz: center frequency vector
    :param cwt_type: one of "fft", or "morlet2". Default is "fft"
    :param index_shift: index of shift. Default is 0.0
    :param frequency_ref: reference frequency in Hz. Default is F1
    :param scale_base: G2 or G3. Default is G2
    :param dictionary_type: Canonical unit-norm ("norm") or unit spectrum ("spect"). Default is "norm"
    :return: cwt, cwt_bits, time_s, frequency_cwt_hz
    """

    wavelet_points = len(sig_wf)
    time_cwt_s = np.arange(wavelet_points)/frequency_sample_rate_hz
    scale_points = len(frequency_cwt_hz)
    cycles_M = cycles_from_order(band_order_Nth=band_order_Nth)

    cw_complex, _, _, _, amp = \
        wavelet_centered_4cwt(band_order_Nth=band_order_Nth,
                              duration_points=wavelet_points,
                              scale_frequency_center_hz=frequency_cwt_hz,
                              frequency_sample_rate_hz=frequency_sample_rate_hz,
                              dictionary_type=dictionary_type)

    if cwt_type == "morlet2":
        scale_atom, _ = \
            scale_from_frequency_hz(band_order_Nth=band_order_Nth,
                                    frequency_sample_rate_hz=frequency_sample_rate_hz,
                                    scale_frequency_center_hz=frequency_cwt_hz)
        cwt = signal.cwt(data=sig_wf, wavelet=signal.morlet2,
                         widths=scale_atom,
                         w=cycles_M,
                         dtype=np.complex128)
        if dictionary_type == 'spect':
            cwt_amp_norm2spec = amplitude_convert_norm_to_spect(scale_atom=scale_atom)
            # Convert to 2d matrix
            spec_scale = np.tile(cwt_amp_norm2spec, (wavelet_points, 1)).T
            cwt *= spec_scale

    else:
        # Convolution using the fft method
        # Convert to a 2d matrix
        sig_wf_2d = np.tile(sig_wf, (scale_points, 1))
        # Flip time
        cw_complex_fliplr = np.fliplr(cw_complex)
        cwt = signal.fftconvolve(sig_wf_2d, np.conj(cw_complex_fliplr), mode='same', axes=-1)

    return frequency_cwt_hz, time_cwt_s, cwt


def power_and_information_shannon_cwt(cwt_complex):
    """
    Computes power and information metrics
    :param cwt_complex:
    :return:
    """
    power = 2 * np.abs(np.copy(cwt_complex)) ** 2
    power_per_band = np.sum(power, axis=-1)
    power_per_sample = np.sum(power, axis=0)
    power_total = np.sum(power) + scales.EPSILON
    power_scaled = power/power_total
    information_bits = -power_scaled*np.log2(power_scaled + scales.EPSILON)
    information_bits_per_band = np.sum(information_bits, axis=-1)
    information_bits_per_sample = np.sum(information_bits, axis=0)
    information_bits_total = np.sum(information_bits) + scales.EPSILON
    information_scaled = information_bits/information_bits_total
    return power, power_per_band, power_per_sample, power_total, power_scaled, \
           information_bits, information_bits_per_band, information_bits_per_sample, \
           information_bits_total, information_scaled




# TODO: WRITE METHODS FOR POWER AND INFORMATION/BITS
# def chirp_spectrum(frequency_hz: np.ndarray,
#                    offset_time_s: float,
#                    band_order_Nth: float,
#                    scale_frequency_center_hz: float,
#                    frequency_sample_rate_hz: float,
#                    index_shift: float = 0,
#                    scale_base: float = scales.Slice.G2) -> Tuple[Union[complex, float, np.ndarray], np.ndarray]:
#     """
#     Spectrum of quantum wavelet for specified band_order_Nth and arbitrary time duration
#
#     :param frequency_hz: frequency range below Nyquist
#     :param offset_time_s: time of wavelet centroid
#     :param band_order_Nth: Nth order of constant Q bands
#     :param scale_frequency_center_hz: band center frequency in Hz
#     :param frequency_sample_rate_hz: sample rate on Hz
#     :param index_shift: index of shift. Default is 0.0
#     :param scale_base: positive reference Base G > 1. Default is G2
#     :return: Fourier transform of the Gabor atom
#     """
#
#     cycles_M, quality_factor_Q, gamma = \
#         chirp_MQG_from_N(band_order_Nth, index_shift, scale_base)
#     scale_atom = chirp_scale(cycles_M, scale_frequency_center_hz, frequency_sample_rate_hz)
#     p_complex = chirp_p_complex(scale_atom, gamma, index_shift)
#
#     angular_frequency_center = 2 * np.pi * scale_frequency_center_hz/frequency_sample_rate_hz
#     angular_frequency = 2 * np.pi * frequency_hz/frequency_sample_rate_hz
#     offset_phase = angular_frequency * frequency_sample_rate_hz * offset_time_s
#     angular_frequency_shifted = angular_frequency - angular_frequency_center
#     frequency_shifted_hz = angular_frequency_shifted*frequency_sample_rate_hz/(2*np.pi)
#
#     # spectrum_amplitude = np.sqrt(np.pi/p_complex)
#     spectrum_amplitude = np.sqrt(p_complex/np.abs(p_complex))
#     gauss_arg = 1./(4*p_complex)
#     # spectrum_gauss = np.exp(-gauss_arg * (angular_frequency_shifted * scale_atom) ** 2)
#     spectrum_gauss = np.exp(-gauss_arg * (angular_frequency_shifted**2))
#     # Phase shift from time offset
#     spectrum_gabor = spectrum_amplitude * spectrum_gauss * np.exp(-1j * offset_phase)
#
#     return spectrum_gabor, frequency_shifted_hz
#
#
# def chirp_spectrum_centered(band_order_Nth: float,
#                             scale_frequency_center_hz: float,
#                             frequency_sample_rate_hz: float,
#                             index_shift: float = 0,
#                             scale_base: float = scales.Slice.G2) -> Tuple[Union[complex, float, np.ndarray], np.ndarray]:
#     """
#     Spectrum of quantum wavelet for specified band_order_Nth and arbitrary time duration
#
#     :param frequency_hz: frequency range below Nyquist
#     :param offset_time_s: time of wavelet centroid
#     :param band_order_Nth: Nth order of constant Q bands
#     :param scale_frequency_center_hz: band center frequency in Hz
#     :param frequency_sample_rate_hz: sample rate on Hz
#     :param index_shift: index of shift. Default is 0.0
#     :param scale_base: positive reference Base G > 1. Default is G2
#     :return: Fourier transform of the Gabor atom
#     """
#
#     # TODO: Generalize to two dictionaries
#     cycles_M, quality_factor_Q, gamma = \
#         chirp_MQG_from_N(band_order_Nth, index_shift, scale_base)
#     scale_atom = chirp_scale(cycles_M, scale_frequency_center_hz, frequency_sample_rate_hz)
#     p_complex = chirp_p_complex(scale_atom, gamma, index_shift)
#     angular_frequency_shifted = np.arange(-np.pi, np.pi, np.pi/2**7)
#     frequency_shifted_hz = angular_frequency_shifted*frequency_sample_rate_hz/(2*np.pi)
#
#     # spectrum_amplitude = np.sqrt(np.pi/p_complex)
#     spectrum_amplitude = np.sqrt(p_complex/np.abs(p_complex))
#     spectrum_gauss = np.exp(-(angular_frequency_shifted**2)/(4*p_complex))
#     spectrum_gabor = spectrum_amplitude * spectrum_gauss
#
#     return spectrum_gabor, frequency_shifted_hz
#
#
# def chirp_uncertainty(scale_atom: float,
#                       frequency_sample_rate_hz: float,
#                       gamma: float,
#                       index_shift: float) -> Tuple[float, float, float]:
#     """
#     Uncertainty of chirp
#
#     :param scale_atom: from chirp_scale or chirp_scale_from_order
#     :param frequency_sample_rate_hz: sample rate in hz
#     :param gamma: from index_shift, M/(2Q)
#     :param index_shift: index of shift
#     :return: time std in seconds, frequency std in Hz, angular frequency std in Hz
#     """
#
#     time_std_s = scale_atom/np.sqr(2)/frequency_sample_rate_hz
#     angular_frequency_std = np.sqrt(1+(index_shift*gamma)**2)/scale_atom/np.sqr(2)
#     angular_frequency_std_hz = frequency_sample_rate_hz*angular_frequency_std
#     frequency_std_hz = angular_frequency_std_hz/2/np.pi
#
#     return time_std_s, frequency_std_hz, angular_frequency_std_hz
#
#
# def chirp_p_complex(scale_atom: float,
#                     gamma: float,
#                     index_shift: float) -> complex:
#     """
#     Fundamental chirp variable
#
#     :param scale_atom: from chirp_scale or chirp_scale_from_order
#     :param gamma: from index_shift, M/(2Q)
#     :param index_shift: index of shift
#     :return: p_complex
#     """
#     p_complex = (1 - 1j*index_shift*gamma/np.pi)/(2*scale_atom**2)
#     return p_complex
#
#
# def chirp_amplitude(scale_atom: float,
#                     gamma: float,
#                     index_shift: float) -> Tuple[float, float]:
#     """
#     Return chirp amplitude
#
#     :param scale_atom: from chirp_scale or chirp_scale_from_order
#     :param gamma: from index_shift, M/(2Q)
#     :param index_shift: index of shift
#     :return: amp_dict_0, amp_dict_1
#     """
#     p_complex = chirp_p_complex(scale_atom, gamma, index_shift)
#     amp_dict_0 = 1/np.pi**0.25 * 1/np.sqrt(scale_atom)
#     amp_dict_1 = np.sqrt(np.abs(p_complex)/np.pi)
#     return amp_dict_0, amp_dict_1
#
#
# def chirp_time(time_s: np.ndarray,
#                offset_time_s: float,
#                frequency_sample_rate_hz: float) -> np.ndarray:
#     """
#     Scaled time-shifted time
#
#     :param time_s: array with time
#     :param offset_time_s: offset time in seconds
#     :param frequency_sample_rate_hz: sample rate in Hz
#     :return: numpy array with time-shifted time
#     """
#     xtime_shifted = frequency_sample_rate_hz*(time_s-offset_time_s)
#     return xtime_shifted
#
#
# def chirp_scales_from_duration(band_order_Nth : float,
#                                sig_duration_s: float,
#                                index_shift: float = 0.,
#                                scale_base: float = scales.Slice.G2) -> Tuple[float, float]:
#     """
#     Calculate scale factor for time and frequency from chirp duration
#
#     :param band_order_Nth: Band order
#     :param sig_duration_s: signal duration in seconds
#     :param index_shift: index fo shift. Default is 0.0
#     :param scale_base: positive reference Base G > 1. Default is G2
#     :return: time in seconds and frequency in Hz scale factors
#     """
#     cycles_M, _, _ = chirp_MQG_from_N(band_order_Nth,
#                                       index_shift,
#                                       scale_base)
#     scale_time_s = sig_duration_s/cycles_M
#     scale_frequency_hz = 1/scale_time_s
#     return scale_time_s, scale_frequency_hz
#
#
# def chirp_frequency_bands(scale_order_input: float,
#                           frequency_low_input: float,
#                           frequency_sample_rate_input: float,
#                           frequency_high_input: float,
#                           index_shift: float = 0,
#                           frequency_ref: float = scales.Slice.F1,
#                           scale_base: float = scales.Slice.G2) -> Tuple[float, float, float, float,
#                                                                         np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Calculate frequency bands for chirp
#
#     :param scale_order_input: Nth order specification
#     :param frequency_low_input: lowest frequency of interest
#     :param frequency_sample_rate_input: sample rate
#     :param frequency_high_input: highest frequency of interest
#     :param index_shift: index of shift
#     :param frequency_ref: reference frequency
#     :param scale_base: positive reference Base G > 1. Default is G2
#     :return: Nth order, cycles M, quality factor Q, gamma, geometric center of frequencies, start frequency,
#         end frequency
#     """
#
#     order_Nth, scale_base, scale_band_number, \
#     frequency_ref, frequency_center_algebraic, frequency_center_geometric, \
#     frequency_start, frequency_end = \
#         scales.band_frequencies_low_high(frequency_order_input=scale_order_input,
#                                          frequency_base_input=scale_base,
#                                          frequency_ref_input=frequency_ref,
#                                          frequency_low_input=frequency_low_input,
#                                          frequency_high_input=frequency_high_input,
#                                          frequency_sample_rate_input=frequency_sample_rate_input)
#     cycles_M, quality_Q, gamma = chirp_MQG_from_N(order_Nth, index_shift, scale_base)
#
#     return order_Nth, cycles_M, quality_Q, gamma, frequency_center_geometric, frequency_start, frequency_end
#
#

#
#
# def cwt_chirp_complex(band_order_Nth: float,
#                       sig_wf: np.ndarray,
#                       frequency_low_hz: float,
#                       frequency_sample_rate_hz: float,
#                       frequency_high_hz: float = scales.Slice.F0,
#                       cwt_type: str = "fft",
#                       index_shift: float = 0,
#                       frequency_ref: float = scales.Slice.F1,
#                       scale_base: float = scales.Slice.G2,
#                       dictionary_type: str = "norm") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Calculate CWT for chirp
#
#     :param band_order_Nth: Nth order of constant Q bands
#     :param sig_wf: array with input signal
#     :param frequency_low_hz: lowest frequency in Hz
#     :param frequency_sample_rate_hz: sample rate in Hz
#     :param frequency_high_hz: highest frequency in Hz
#     :param cwt_type: one of "conv", "fft", or "morlet2". Default is "fft"
#            Address ghost folding in "fft", compared to "conv"
#     :param index_shift: index of shift. Default is 0.0
#     :param frequency_ref: reference frequency in Hz. Default is F1
#     :param scale_base: G2 or G3. Default is G2
#     :param dictionary_type: Canonical unit-norm ("norm") or unit spectrum ("spect"). Default is "norm"
#     :return: cwt, cwt_bits, time_s, frequency_cwt_hz
#     """
#
#     wavelet_points = len(sig_wf)
#     time_s = np.arange(wavelet_points)/frequency_sample_rate_hz
#
#     if cwt_type == "morlet2":
#         index_shift = 0
#
#     # Planck frequency is absolute upper limit
#     if frequency_high_hz > frequency_sample_rate_hz/2.:
#         frequency_high_hz = frequency_sample_rate_hz/2.
#
#     order_Nth, cycles_M, quality_Q, _, \
#     frequency_cwt_hz_flipped, frequency_start_flipped, frequency_end_flipped = \
#         chirp_frequency_bands(scale_order_input=band_order_Nth,
#                               frequency_low_input=frequency_low_hz,
#                               frequency_sample_rate_input=frequency_sample_rate_hz,
#                               frequency_high_input=frequency_high_hz,
#                               index_shift=index_shift,
#                               frequency_ref=frequency_ref,
#                               scale_base=scale_base)
#
#     scale_points = len(frequency_cwt_hz_flipped)
#
#     if cwt_type == "morlet2":
#         scale_atom = chirp_scale(cycles_M, frequency_cwt_hz_flipped, frequency_sample_rate_hz)
#         cwt_flipped = signal.cwt(data=sig_wf, wavelet=signal.morlet2,
#                                  widths=scale_atom,
#                                  w=cycles_M,
#                                  dtype=np.complex128)
#     elif cwt_type == "fft":
#         sig_fft = np.fft.fft(sig_wf)
#         cwt_flipped = np.empty((scale_points, wavelet_points), dtype=np.complex128)
#         for ii in range(scale_points):
#             atom, _ = chirp_centered_4cwt(band_order_Nth=order_Nth,
#                                           sig_or_time=sig_wf,
#                                           scale_frequency_center_hz=frequency_cwt_hz_flipped[ii],
#                                           frequency_sample_rate_hz=frequency_sample_rate_hz,
#                                           index_shift=index_shift,
#                                           scale_base=scale_base,
#                                           dictionary_type=dictionary_type)
#             atom_fft = np.fft.fft(atom)
#             cwt_raw = np.fft.ifft(sig_fft*np.conj(atom_fft))
#             cwt_flipped[ii, :] = np.append(cwt_raw[wavelet_points//2:], cwt_raw[0:wavelet_points//2])
#
#     elif cwt_type == "conv":
#         cwt_flipped = np.empty((scale_points, wavelet_points), dtype=np.complex128)
#         for ii in range(scale_points):
#             atom, _ = chirp_centered_4cwt(band_order_Nth=order_Nth,
#                                           sig_or_time=sig_wf,
#                                           scale_frequency_center_hz=frequency_cwt_hz_flipped[ii],
#                                           frequency_sample_rate_hz=frequency_sample_rate_hz,
#                                           index_shift=index_shift,
#                                           scale_base=scale_base,
#                                           dictionary_type=dictionary_type)
#             cwt_flipped[ii, :] = signal.convolve(sig_wf, np.conj(atom)[::-1], mode='same')
#     else:
#         print("Incorrect cwt_type specification in cwt_chirp_complex")
#
#     # Time scales are increasing, which is the opposite of what is expected for the frequency. Flip.
#     frequency_cwt_hz = np.flip(frequency_cwt_hz_flipped)
#     cwt = np.flipud(cwt_flipped)
#     cwt_bits = utils.log2epsilon(cwt)
#
#     return cwt, cwt_bits, time_s, frequency_cwt_hz
#
#
# def cwt_chirp_from_sig(sig_wf: np.ndarray,
#                        frequency_sample_rate_hz: float,
#                        band_order_Nth: float = 3,
#                        cwt_type: str = "fft",
#                        index_shift: float = 0,
#                        frequency_ref: float = scales.Slice.F1,
#                        scale_base: float = scales.Slice.G2,
#                        dictionary_type: str = "norm"):
#     """
#     Calculate CWT for chirp
#
#     :param sig_wf: array with input signal
#     :param frequency_sample_rate_hz: sample rate in Hz
#     :param band_order_Nth: Nth order of constant Q bands
#     :param cwt_type: one of "conv", "fft", or "morlet2". Default is "fft"
#     :param index_shift: index of shift. Default is 0.0
#     :param frequency_ref: reference frequency in Hz. Default is F1
#     :param scale_base: G2 or G3. Default is G2
#     :param dictionary_type: Canonical unit-norm ("norm") or unit spectrum ("spect"). Default is "norm"
#     :return: cwt, cwt_bits, time_s, frequency_cwt_hz
#     """
#
#     wavelet_points = len(sig_wf)
#     duration_s = wavelet_points/frequency_sample_rate_hz
#     _, min_frequency_hz = \
#         chirp_scales_from_duration(band_order_Nth=band_order_Nth,
#                                    sig_duration_s=duration_s,
#                                    index_shift=index_shift,
#                                    scale_base=scale_base)
#
#     cwt, cwt_bits, time_s, frequency_cwt_hz = \
#         cwt_chirp_complex(band_order_Nth=band_order_Nth,
#                           sig_wf=sig_wf,
#                           frequency_low_hz=min_frequency_hz,
#                           frequency_sample_rate_hz=frequency_sample_rate_hz,
#                           frequency_high_hz=frequency_sample_rate_hz/2.,
#                           cwt_type=cwt_type,
#                           index_shift=index_shift,
#                           frequency_ref=frequency_ref,
#                           scale_base=scale_base,
#                           dictionary_type=dictionary_type)
#
#     return cwt, cwt_bits, time_s, frequency_cwt_hz
