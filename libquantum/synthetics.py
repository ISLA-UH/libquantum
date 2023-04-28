"""
This module constructs synthetic signals
"""

import numpy as np
import scipy.signal as signal
from scipy.integrate import cumulative_trapezoid
from typing import Optional, Tuple, Union
from libquantum import utils, scales, atoms


def gabor_loose_grain(band_order_Nth: float,
                      number_points: int,
                      scale_frequency_center_hz: float,
                      frequency_sample_rate_hz: float,
                      index_shift: float = 0,
                      frequency_base_input: float = scales.Slice.G2) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Loose grain with tight Tukey wrap to ensure zero at edges

    :param band_order_Nth: Nth order of constant Q bands
    :param number_points: Number of points in the signal
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :param index_shift: index of shift for the Gabor chirp, default of zero
    :param frequency_base_input: G2 or G3. Default is G2
    :return: numpy array with Tukey grain
    """

    # Fundamental chirp parameters
    cycles_M, quality_factor_Q, gamma = \
        atoms.chirp_MQG_from_N(band_order_Nth, index_shift, frequency_base_input)
    scale_atom = atoms.chirp_scale(cycles_M, scale_frequency_center_hz, frequency_sample_rate_hz)
    p_complex = atoms.chirp_p_complex(scale_atom, gamma, index_shift)

    # # Time from nominal duration
    # grain_duration_s = cycles_M/scale_frequency_center_hz
    # Time from number of points
    time_s = np.arange(number_points)/frequency_sample_rate_hz
    offset_time_s = np.max(time_s)/2.

    xtime_shifted = atoms.chirp_time(time_s, offset_time_s, frequency_sample_rate_hz)
    wavelet_gauss = np.exp(-p_complex * xtime_shifted**2)
    wavelet_gabor = wavelet_gauss * np.exp(1j * cycles_M*xtime_shifted/scale_atom)
    loose_grain_taper = utils.taper_tukey(wavelet_gabor, 0.1)
    loose_grain = np.copy(wavelet_gabor) * loose_grain_taper

    return loose_grain, time_s, scale_atom


def gabor_tight_grain(band_order_Nth: float,
                      scale_frequency_center_hz: float,
                      frequency_sample_rate_hz: float,
                      index_shift: float = 0,
                      frequency_base_input: float = scales.Slice.G2) -> np.ndarray:
    """
    Gabor grain with tight Tukey wrap to ensure zero at edges

    :param band_order_Nth: Nth order of constant Q bands
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :param index_shift: index of shift
    :param frequency_base_input: G2 or G3. Default is G2
    :return: numpy array with Tukey grain
    """

    # Fundamental chirp parameters
    cycles_M, quality_factor_Q, gamma = \
        atoms.chirp_MQG_from_N(band_order_Nth, index_shift, frequency_base_input)
    scale_atom = atoms.chirp_scale(cycles_M, scale_frequency_center_hz, frequency_sample_rate_hz)
    p_complex = atoms.chirp_p_complex(scale_atom, gamma, index_shift)

    # Time from nominal duration
    grain_duration_s = cycles_M/scale_frequency_center_hz
    time_s = np.arange(int(np.round(grain_duration_s*frequency_sample_rate_hz)))/frequency_sample_rate_hz
    offset_time_s = np.max(time_s)/2.

    xtime_shifted = atoms.chirp_time(time_s, offset_time_s, frequency_sample_rate_hz)
    wavelet_gauss = np.exp(-p_complex * xtime_shifted**2)
    wavelet_gabor = wavelet_gauss * np.exp(1j * cycles_M*xtime_shifted/scale_atom)
    tight_grain_taper = utils.taper_tukey(wavelet_gabor, 0.1)
    tight_grain = np.copy(wavelet_gabor)*tight_grain_taper

    return tight_grain


def tukey_tight_grain(band_order_Nth: float,
                      scale_frequency_center_hz: float,
                      frequency_sample_rate_hz: float,
                      fraction_cosine: float = 0.5,
                      index_shift: float = 0,
                      frequency_base_input: float = scales.Slice.G2) -> np.ndarray:
    """
    Tukey grain with same support as Gabor atom

    :param band_order_Nth: Nth order of constant Q bands
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :param fraction_cosine: fraction of the window inside the cosine tapered window, shared between the head and tail.
        Default is 0.5
    :param index_shift: index of shift
    :param frequency_base_input: G2 or G3. Default is G2
    :return: numpy array with Tukey grain
    """

    # Fundamental chirp parameters
    cycles_M, quality_factor_Q, gamma = \
        atoms.chirp_MQG_from_N(band_order_Nth, index_shift, frequency_base_input)
    scale_atom = atoms.chirp_scale(cycles_M, scale_frequency_center_hz, frequency_sample_rate_hz)
    p_complex = atoms.chirp_p_complex(scale_atom, gamma, index_shift)

    # Time from nominal duration
    grain_duration_s = cycles_M/scale_frequency_center_hz
    time_s = np.arange(int(np.round(grain_duration_s*frequency_sample_rate_hz)))/frequency_sample_rate_hz
    offset_time_s = np.max(time_s)/2.

    xtime_shifted = atoms.chirp_time(time_s, offset_time_s, frequency_sample_rate_hz)
    # Pull out phase component from gaussian envelope
    wavelet_gauss_phase = np.imag(-p_complex * xtime_shifted**2)
    wavelet_gabor = np.exp(1j * cycles_M*xtime_shifted/scale_atom + 1j * wavelet_gauss_phase)
    tight_grain_taper = utils.taper_tukey(wavelet_gabor, fraction_cosine)
    tight_grain = np.copy(wavelet_gabor)*tight_grain_taper

    return tight_grain


def gabor_grain_frequencies(frequency_order_input: float,
                            frequency_low_input: float,
                            frequency_high_input: float,
                            frequency_sample_rate_input: float,
                            frequency_base_input: float = scales.Slice.G2,
                            frequency_ref_input: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Frequencies for g-chirps

    :param frequency_order_input: Nth order
    :param frequency_low_input: lowest frequency of interest
    :param frequency_high_input: highest frequency of interest
    :param frequency_sample_rate_input: sample rate
    :param frequency_base_input: G2 or G3. Default is G2
    :param frequency_ref_input: reference frequency. Default is 1.0
    :return: three numpy arrays with center frequency, start frequency and end frequency
    """

    scale_order, scale_base, _, frequency_ref, frequency_center_algebraic, \
    frequency_center, frequency_start, frequency_end = \
        scales.band_frequencies_low_high(frequency_order_input, frequency_base_input,
                                         frequency_ref_input,
                                         frequency_low_input,
                                         frequency_high_input,
                                         frequency_sample_rate_input)

    return frequency_center, frequency_start, frequency_end


def chirp_rdvxm_noise_16bit(duration_points: int = 2**12,
                            sample_rate_hz: float = 80.,
                            noise_std_loss_bits: float = 4.,
                            frequency_center_hz: Optional[float] = None):
    """
    Construct chirp with linear frequency sweep, white noise added, anti-aliased filter applied

    :param duration_points: number of points, length of signal. Default is 2 ** 12
    :param sample_rate_hz: sample rate in Hz. Default is 80.0
    :param noise_std_loss_bits: number of bits below signal standard deviation. Default is 4.0
    :param frequency_center_hz: center frequency fc in Hz. Optional
    :return: numpy ndarray with anti-aliased chirp with white noise
    """

    duration_s = duration_points/sample_rate_hz
    if frequency_center_hz:
        frequency_start_hz = 0.5*frequency_center_hz
        frequency_end_hz = sample_rate_hz/4.
    else:
        frequency_center_hz = 8./duration_s
        frequency_start_hz = 0.5*frequency_center_hz
        frequency_end_hz = sample_rate_hz/4.

    sig_time_s = np.arange(int(duration_points))/sample_rate_hz
    chirp_wf = signal.chirp(sig_time_s, frequency_start_hz, sig_time_s[-1],
                            frequency_end_hz, method='linear', phi=0, vertex_zero=True)
    chirp_wf *= taper_tukey(chirp_wf, 0.25)
    noise_wf = white_noise_fbits(sig=chirp_wf, std_bit_loss=noise_std_loss_bits)
    chirp_white = chirp_wf + noise_wf
    chirp_white_aa = antialias_halfNyquist(chirp_white)
    chirp_white_aa.astype(np.float16)

    return chirp_white_aa


def sawtooth_rdvxm_noise_16bit(duration_points: int = 2**12,
                               sample_rate_hz: float = 80.,
                               noise_std_loss_bits: float = 4.,
                               frequency_center_hz: Optional[float] = None) -> np.ndarray:
    """
    Construct a anti-aliased sawtooth waveform with white noise

    :param duration_points: number of points, length of signal. Default is 2 ** 12
    :param sample_rate_hz: sample rate in Hz. Default is 80.0
    :param noise_std_loss_bits: number of bits below signal standard deviation. Default is 4.0
    :param frequency_center_hz: center frequency fc in Hz. Optional
    :return: numpy ndarray with anti-aliased sawtooth signal with white noise
    """

    duration_s = duration_points/sample_rate_hz
    if frequency_center_hz:
        frequency_center_angular = 2*np.pi*frequency_center_hz
    else:
        frequency_center_hz = 8./duration_s
        frequency_center_angular = 2*np.pi*frequency_center_hz

    sig_time_s = np.arange(int(duration_points))/sample_rate_hz
    saw_wf = signal.sawtooth(frequency_center_angular*sig_time_s, width=0)
    saw_wf *= taper_tukey(saw_wf, 0.25)
    noise_wf = white_noise_fbits(sig=saw_wf, std_bit_loss=noise_std_loss_bits)
    saw_white = saw_wf + noise_wf
    saw_white_aa = antialias_halfNyquist(saw_white)
    saw_white_aa.astype(np.float16)

    return saw_white_aa


def chirp_linear_in_noise(snr_bits: float,
                          sample_rate_hz: float,
                          duration_s: float,
                          frequency_start_hz: float,
                          frequency_end_hz: float,
                          intro_s: Union[int, float],
                          outro_s: Union[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct chirp with linear frequency sweep, white noise added.

    :param snr_bits: number of bits below signal standard deviation
    :param sample_rate_hz: sample rate in Hz
    :param duration_s: duration of chirp in seconds
    :param frequency_start_hz: start frequency in Hz
    :param frequency_end_hz: end frequency in Hz
    :param intro_s: number of seconds before chirp
    :param outro_s: number of seconds after chirp
    :return: numpy ndarray with waveform, numpy ndarray with time in seconds
    """

    sig_time_s = np.arange(int(sample_rate_hz*duration_s))/sample_rate_hz
    chirp_wf = signal.chirp(sig_time_s, frequency_start_hz, sig_time_s[-1],
                            frequency_end_hz, method='linear', phi=0, vertex_zero=True)
    chirp_wf *= taper_tukey(chirp_wf, 0.25)
    sig_wf = np.concatenate((np.zeros(int(intro_s*sample_rate_hz)),
                             chirp_wf,
                             np.zeros(int(outro_s*sample_rate_hz))))
    noise_wf = white_noise_fbits(sig=sig_wf, std_bit_loss=snr_bits)
    synth_wf = sig_wf+noise_wf
    synth_time_s = np.arange(len(synth_wf))/sample_rate_hz
    return synth_wf, synth_time_s


def white_noise_fbits(sig: np.ndarray,
                      std_bit_loss: float) -> np.ndarray:
    """
    Compute white noise with zero mean and standard deviation that is snr_bits below the input signal

    :param sig: input signal, detrended
    :param std_bit_loss: number of bits below signal standard deviation
    :return: gaussian noise with zero mean
    """
    sig_std = np.std(sig)
    # This is in power, or variance
    noise_loss = 2.**std_bit_loss
    std_from_fbits = sig_std/noise_loss
    # White noise, zero mean
    sig_noise = np.random.normal(0, std_from_fbits, size=sig.size)
    return sig_noise


def taper_tukey(sig_or_time: np.ndarray,
                fraction_cosine: float) -> np.ndarray:
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


def antialias_halfNyquist(synth: np.ndarray) -> np.ndarray:
    """
    Anti-aliasing filter with -3dB at 1/4 of sample rate, 1/2 of Nyquist

    :param synth: array with signal data
    :return: numpy array with anti-aliased signal
    """
    # Anti-aliasing filter with -3dB at 1/4 of sample rate, 1/2 of Nyquist
    # Signal frequencies are scaled by Nyquist
    filter_order = 2
    edge_high = 0.5
    [b, a] = signal.butter(filter_order, edge_high, btype='lowpass')
    synth_anti_aliased = signal.filtfilt(b, a, np.copy(synth))
    return synth_anti_aliased


def frequency_algebraic_Nth(frequency_geometric: np.ndarray,
                            band_order_Nth: float) -> np.ndarray:
    """
    Compute algebraic frequencies in band order

    :param frequency_geometric: geometric frequencies
    :param band_order_Nth:  Nth order of constant Q bands
    :return:
    """
    frequency_algebra = frequency_geometric*(np.sqrt(1+1/(8*band_order_Nth**2)))
    return frequency_algebra


def integrate_cumtrapz(timestamps_s: np.ndarray,
                       sensor_wf: np.ndarray,
                       initial_value: float = 0) -> np.ndarray:
    """
    cumulative trapezoid integration using scipy.integrate.cumulative_trapezoid

    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :param initial_value: the value to add in the initial of the integrated data to match length of input (default is 0)
    :return: integrated data with the same length as the input
    """
    integrated_data = cumulative_trapezoid(x=timestamps_s,
                                           y=sensor_wf,
                                           initial=initial_value)
    return integrated_data
