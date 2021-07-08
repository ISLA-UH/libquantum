"""
This module constructs synthetic signals
Last updated: 8 July 2021
"""

import numpy as np
import scipy.signal as signal
from scipy.integrate import cumulative_trapezoid
from typing import Optional
from libquantum import utils, scales, atoms


# Synthetics for rdvxm testing
def gabor_tight_grain(band_order_Nth: float,
                      scale_frequency_center_hz: float,
                      frequency_sample_rate_hz: float,
                      index_shift: float = 0,
                      frequency_base_input: float = scales.Slice.G2):
    """

    :param band_order_Nth:
    :param scale_frequency_center_hz:
    :param frequency_sample_rate_hz:
    :param index_shift:
    :param frequency_base_input:
    :return:
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
                      fraction_cosine: float=0.5,
                      index_shift: float = 0,
                      frequency_base_input: float = scales.Slice.G2):
    """

    :param band_order_Nth:
    :param scale_frequency_center_hz:
    :param frequency_sample_rate_hz:
    :param index_shift:
    :param frequency_base_input:
    :return:
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
                            frequency_ref_input: float = 1.0) -> (np.ndarray, np.ndarray):
    """


    :param frequency_order_input:
    :param frequency_low_input:
    :param frequency_high_input:
    :param frequency_sample_rate_input:
    :param frequency_base_input:
    :param frequency_ref_input:
    :return:
    """

    scale_order, scale_base, _, frequency_ref, frequency_center_algebraic, \
    frequency_center, frequency_start, frequency_end = \
        scales.band_frequencies_low_high(frequency_order_input, frequency_base_input,
                                         frequency_ref_input,
                                         frequency_low_input,
                                         frequency_high_input,
                                         frequency_sample_rate_input)

    return frequency_center, frequency_start, frequency_end


def gt_rdvxm_center_noise_16bit(duration_points: int = 2**12, sample_rate_hz: float = 80,
                                noise_std_loss_bits: float = 2, frequency_center_hz: Optional[float] = None):
    """
    Construct the GT explosion pulse of Garces (2019) in Gaussion noise with SNR in bits re signal STD

    :param duration_points:
    :param sample_rate_hz:
    :param noise_std_loss_bits:
    :param frequency_center_hz:
    :return:
    """

    if frequency_center_hz:
        pseudo_period_s = 1/frequency_center_hz
    else:
        pseudo_period_s = duration_points/sample_rate_hz/4.

    time_center_s = np.arange(duration_points)/sample_rate_hz
    time_center_s -= time_center_s[-1]/2.
    sig_gt = gt_blast_period_center(time_center_s, pseudo_period_s)
    sig_noise = white_noise_fbits(sig_gt, noise_std_loss_bits)
    gt_white = sig_gt + sig_noise
    # AA filter
    gt_white_aa = antialias_halfNyquist(gt_white)
    gt_white_aa.astype(np.float16)

    return gt_white_aa


def gt_rdvxm_center_noise_uneven(sensor_epoch_micros: np.array, start_epoch_micros: float,
                                 duration_points: int = 2**12, sample_rate_hz: float = 80,
                                 noise_std_loss_bits: float = 2, frequency_center_hz: Optional[float] = None):
    """
    Construct the GT explosion pulse of Garces (2019) for uneven sensor time
    in Gaussion noise with SNR in bits re signal STD

    :param sensor_epoch_micros:
    :param start_epoch_micros:
    :param duration_points:
    :param sample_rate_hz:
    :param noise_std_loss_bits:
    :param frequency_center_hz:
    :return:
    """

    if frequency_center_hz:
        pseudo_period_s = 1/frequency_center_hz
    else:
        pseudo_period_s = duration_points/sample_rate_hz/4.

    # Convert to seconds
    time_duration_s = duration_points/sample_rate_hz
    time_center_s = (sensor_epoch_micros - start_epoch_micros)/1E6 - time_duration_s/2.
    sig_gt = gt_blast_period_center(time_center_s, pseudo_period_s)
    sig_noise = white_noise_fbits(sig_gt, noise_std_loss_bits)
    gt_white = sig_gt + sig_noise
    # AA filter
    gt_white_aa = antialias_halfNyquist(gt_white)

    return gt_white_aa


def chirp_rdvxm_noise_16bit(duration_points: int = 2**12, sample_rate_hz: float = 80,
                            noise_std_loss_bits: float = 4, frequency_center_hz: Optional[float] = None):
    """


    :param duration_points:
    :param sample_rate_hz:
    :param noise_std_loss_bits:
    :param frequency_center_hz:
    :return:
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


def sawtooth_rdvxm_noise_16bit(duration_points: int = 2**12, sample_rate_hz: float = 80,
                               noise_std_loss_bits: float = 4, frequency_center_hz: Optional[float] = None):
    """


    :param duration_points:
    :param sample_rate_hz:
    :param noise_std_loss_bits:
    :param frequency_center_hz:
    :return:
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


# GT Test Pulse
def gt_blast_center_integral_and_derivative(frequency_peak_hz, sample_rate_hz):
    """
    Integral and derivative relative to tau (NOT time_s)

    :param frequency_peak_hz:
    :param sample_rate_hz:
    :return:
    """

    duration_s = 2/frequency_peak_hz       # 16 cycles for 6th octave (M = 14)
    pseudo_period_s = 1/frequency_peak_hz
    time_pos_s = pseudo_period_s/4.
    duration_points = int(duration_s*sample_rate_hz)
    time_center_s = np.arange(duration_points)/sample_rate_hz
    time_center_s -= time_center_s[-1]/2.
    tau_center = time_center_s/time_pos_s
    tau_interval = np.mean(np.diff(tau_center))

    sig_gt = gt_blast_period_center(time_center_s, pseudo_period_s)
    sig_gt_i = gt_blast_integral_period_center(time_center_s, pseudo_period_s)
    sig_gt_d = gt_blast_derivative_period_center(time_center_s, pseudo_period_s)
    sig_gt_d[np.argmax(sig_gt)-1] = np.max(np.diff(sig_gt))/tau_interval

    return tau_center, sig_gt, sig_gt_i, sig_gt_d


def gt_blast_center_fast(frequency_peak_hz,
                         sample_rate_hz):
    """
    TODO: Set default of 16 bits for std_loss_bits

    :param frequency_peak_hz:
    :param sample_rate_hz:
    :return:
    """

    duration_s = 16/frequency_peak_hz       # 16 cycles for 6th octave (M = 14)
    noise_std_loss_bits = 16                # 16 bit record, system noise
    pseudo_period_s = 1/frequency_peak_hz
    duration_points = int(duration_s*sample_rate_hz)
    time_center_s = np.arange(duration_points)/sample_rate_hz
    time_center_s -= time_center_s[-1]/2.
    sig_gt = gt_blast_period_center(time_center_s, pseudo_period_s)
    sig_noise = white_noise_fbits(sig_gt, noise_std_loss_bits)
    gt_white = sig_gt + sig_noise
    # AA filter
    gt_white_aa = antialias_halfNyquist(gt_white)
    return time_center_s, gt_white_aa


# This is a very flexible variation
def gt_blast_center_noise_uneven(sensor_epoch_s: np.array,
                                 noise_std_loss_bits: float = 2,
                                 frequency_center_hz: Optional[float] = None):
    """
    Construct the GT explosion pulse of Garces (2019) for even or uneven sensor time
    in Gaussion noise with SNR in bits re signal STD

    :param sensor_epoch_s:
    :param noise_std_loss_bits:
    :param frequency_center_hz:
    :return:
    """

    time_duration_s = sensor_epoch_s[-1]-sensor_epoch_s[0]

    if frequency_center_hz:
        pseudo_period_s = 1/frequency_center_hz
    else:
        pseudo_period_s = time_duration_s/4.

    # Convert to seconds
    time_center_s = sensor_epoch_s - sensor_epoch_s[0] - time_duration_s/2.
    sig_gt = gt_blast_period_center(time_center_s, pseudo_period_s)
    sig_noise = white_noise_fbits(np.copy(sig_gt), noise_std_loss_bits)
    gt_white = sig_gt + sig_noise
    # AA filter
    gt_white_aa = antialias_halfNyquist(gt_white)

    return gt_white_aa


def gt_blast_center_noise(duration_s, frequency_peak_hz, sample_rate_hz, noise_std_loss_bits):
    """
    TODO: Set default of 16 bits for std_loss_bits

    :param duration_s:
    :param frequency_peak_hz:
    :param sample_rate_hz:
    :param noise_std_loss_bits:
    :return:
    """
    pseudo_period_s = 1/frequency_peak_hz
    duration_points = int(duration_s*sample_rate_hz)
    time_center_s = np.arange(duration_points)/sample_rate_hz
    time_center_s -= time_center_s[-1]/2.
    sig_gt = gt_blast_period_center(time_center_s, pseudo_period_s)
    sig_noise = white_noise_fbits(sig_gt, noise_std_loss_bits)
    gt_white = sig_gt + sig_noise
    # AA filter
    gt_white_aa = antialias_halfNyquist(gt_white)
    return time_center_s, gt_white_aa


def gt_blast_period_center(time_center_s, pseudo_period_s):
    """


    :param time_center_s:
    :param pseudo_period_s:
    :return:
    """
    # Garces (2019) ground truth GT blast pulse
    # with the +1, tau is the zero crossing time - time_start renamed to time_zero for first zero crossing.
    # time_start = time_zero - time_pos
    time_pos_s = pseudo_period_s/4.
    tau = time_center_s/time_pos_s + 1.
    # Initialize GT
    p_GT = np.zeros(tau.size)  # Granstrom-Triangular (GT), 2019
    # Initialize time ranges
    sigint1 = np.where((0.0 <= tau) & (tau <= 1.))  # ONLY positive pulse
    sigintG17 = np.where((1. < tau) & (tau <= 1 + np.sqrt(6.)))  # GT balanced pulse
    p_GT[sigint1] = (1. - tau[sigint1])
    p_GT[sigintG17] = 1./6. * (1. - tau[sigintG17]) * (1. + np.sqrt(6) - tau[sigintG17]) ** 2.

    return p_GT


def gt_blast_derivative_period_center(time_center_s, pseudo_period_s):
    """


    :param time_center_s:
    :param pseudo_period_s:
    :return:
    """
    # Garces (2019) ground truth GT blast pulse
    # with the +1, tau is the zero crossing time - time_start renamed to time_zero for first zero crossing.
    # time_start = time_zero - time_pos
    time_pos_s = pseudo_period_s/4.
    tau = time_center_s/time_pos_s + 1.
    # Initialize GT
    p_GTd = np.zeros(tau.size)  # Granstrom-Triangular (GT), 2019
    # Initialize time ranges
    sigint1 = np.where((0.0 <= tau) & (tau <= 1.))  # ONLY positive pulse
    sigintG17 = np.where((1. < tau) & (tau <= 1 + np.sqrt(6.)))  # GT balanced pulse
    p_GTd[sigint1] = -1.
    p_GTd[sigintG17] = -1./6. * (3. + np.sqrt(6) - 3*tau[sigintG17]) * (1. + np.sqrt(6) - tau[sigintG17])

    return p_GTd


def gt_blast_integral_period_center(time_center_s, pseudo_period_s):
    """


    :param time_center_s:
    :param pseudo_period_s:
    :return:
    """
    # Garces (2019) ground truth GT blast pulse
    # with the +1, tau is the zero crossing time - time_start renamed to time_zero for first zero crossing.
    # time_start = time_zero - time_pos
    time_pos_s = pseudo_period_s/4.
    tau = time_center_s/time_pos_s + 1.
    # Initialize GT
    p_GTi = np.zeros(tau.size)  # Granstrom-Triangular (GT), 2019
    # Initialize time ranges
    sigint1 = np.where((0.0 <= tau) & (tau <= 1.))  # ONLY positive pulse
    sigintG17 = np.where((1. < tau) & (tau <= 1 + np.sqrt(6.)))  # GT balanced pulse
    p_GTi[sigint1] = (1. - tau[sigint1]/2.)*tau[sigint1]

    p_GTi[sigintG17] = -tau[sigintG17]/72. * (
            3 * tau[sigintG17]**3 - 4 * (3 + 2 * np.sqrt(6)) * tau[sigintG17]**2 +
            6 * (9 + 4 * np.sqrt(6)) * tau[sigintG17] - 12 * (7 + 2 * np.sqrt(6)))

    integration_constant = p_GTi[sigint1][-1] - p_GTi[sigintG17][0]
    p_GTi[sigintG17] += integration_constant

    return p_GTi


def gt_blast_ft(frequency_peak_hz, frequency_hz):
    """


    :param frequency_peak_hz:
    :param frequency_hz:
    :return:
    """
    w_scaled = 0.5*np.pi*frequency_hz/frequency_peak_hz
    ft_G17_positive = (1. - 1j*w_scaled - np.exp(-1j*w_scaled))/w_scaled**2.
    ft_G17_negative = np.exp(-1j*w_scaled*(1+np.sqrt(6.)))/(3.*w_scaled**4.) * \
                      (1j*w_scaled*np.sqrt(6.) + 3. +
                       np.exp(1j*w_scaled*np.sqrt(6.))*(3.*w_scaled**2. + 1j*w_scaled*2.*np.sqrt(6.)-3.))
    ft_G17 = (ft_G17_positive + ft_G17_negative)*np.pi/(2*np.pi*frequency_peak_hz)
    return ft_G17


def gt_blast_spectral_density(frequency_peak_hz, frequency_hz):
    """


    :param frequency_peak_hz:
    :param frequency_hz:
    :return:
    """
    fourier_tx = gt_blast_ft(frequency_peak_hz, frequency_hz)
    spectral_density = 2*np.abs(fourier_tx*np.conj(fourier_tx))
    spectral_density_peak = np.max(spectral_density)
    # spectral_density_peak = np.pi/(2*np.pi*frequency_center_hz)
    return spectral_density, spectral_density_peak


def chirp_linear_in_noise(snr_bits, sample_rate_hz, duration_s,
                          frequency_start_hz, frequency_end_hz,
                          intro_s, outro_s,):
    """


    :param snr_bits:
    :param sample_rate_hz:
    :param duration_s:
    :param frequency_start_hz:
    :param frequency_end_hz:
    :param intro_s:
    :param outro_s:
    :return:
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


def white_noise_fbits(sig: np.ndarray, std_bit_loss: float) -> np.ndarray:
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


def taper_tukey(sig_or_time: np.ndarray, fraction_cosine: float) -> np.ndarray:
    """
    Constructs a symmetric Tukey window with the same dimensions as a time or signal numpy array.
    fraction_cosine = 0 is a rectangular window, 1 is a Hann window

    :param sig_or_time: input signal or time
    :param fraction_cosine: fraction of the window inside the cosine tapered window, shared between the head and tail
    :return: tukey taper window amplitude
    """
    number_points = np.size(sig_or_time)
    amplitude = signal.tukey(M=number_points, alpha=fraction_cosine, sym=True)
    return amplitude


def antialias_halfNyquist(synth):
    """


    :param synth:
    :return:
    """
    # Anti-aliasing filter with -3dB at 1/4 of sample rate, 1/2 of Nyquist
    # Signal frequencies are scaled by Nyquist
    filter_order = 2
    edge_high = 0.5
    [b, a] = signal.butter(filter_order, edge_high, btype='lowpass')
    synth_anti_aliased = signal.filtfilt(b, a, np.copy(synth))
    return synth_anti_aliased


def frequency_algebraic_Nth(frequency_geometric, band_order_Nth):
    frequency_algebra = frequency_geometric*(np.sqrt(1+1/(8*band_order_Nth**2)))
    return frequency_algebra


def integrate_cumtrapz(timestamps_s: np.ndarray,
                       sensor_wf: np.ndarray,
                       initial_value: float = 0) -> np.ndarray:
    """
    cumulative trapazoid integration using scipy.integrate.cumulative_trapezoid

    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :param initial_value: the value to add in the initial of the integrated data to match length of input (default is 0)
    :return: integrated data with the same length as the input
    """
    integrated_data = cumulative_trapezoid(x=timestamps_s,
                                           y=sensor_wf,
                                           initial=initial_value)
    return integrated_data


