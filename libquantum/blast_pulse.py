"""
This module facilitates the rapid construction of the GT blast pulse synthetic,
its integral and derivatives, and its spectrum
References:
- Garcés, M. A. (2019). Explosion Source Models,
Chapter in Infrasound Monitoring for Atmospheric Studies,
Second Edition, Springer, Switzerland, DOI 10.1007/978-3-319-75140_5, p. 273-345.
- Schnurr, J. M., K. Kim, M. A. Garcés, A. Rodgers (2020).
Improved parametric models for explosion pressure signals derived from large datasets,
Seism. Res. Lett.
- Kim, K, A. R. Rodgers, M. A. Garces, and S. C. Myers (2021).
Empirical Acoustic Source Model for Chemical Explosions in Air.
Bulletin of the Seismological Society of America
"""

import numpy as np
from typing import Optional, Tuple, Union
from libquantum.synthetics import white_noise_fbits, antialias_halfNyquist
from libquantum.scales import EPSILON


def gt_blast_period_center(time_center_s: np.ndarray,
                           pseudo_period_s: float) -> np.ndarray:
    """
    GT blast pulse

    :param time_center_s: array with time
    :param pseudo_period_s: period in seconds
    :return: numpy array with GT blast pulse
    """
    # With the +1, tau is the zero crossing time - time_start renamed to time_zero for first zero crossing.
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


def gt_hilbert_blast_period_center(time_center_s: np.ndarray,
                                   pseudo_period_s: float) -> np.ndarray:
    """
    Hilbert transform of the GT blast pulse

    :param time_center_s: array with time
    :param pseudo_period_s: period in seconds
    :return: numpy array with Hilbert transform of the GT blast pulse
    """
    # With the +1, tau is the zero crossing time - time_start renamed to time_zero for first zero crossing.
    time_pos_s = pseudo_period_s/4.
    tau = time_center_s/time_pos_s + 1.
    a = 1 + np.sqrt(6)
    # Initialize GT
    p_GT_H = np.zeros(tau.size)  # Hilbert of Granstrom-Triangular (GT), 2019
    # Initialize time ranges
    sigint1 = np.where((0.0 <= tau) & (tau <= 1.))  # ONLY positive pulse
    sigint2 = np.where((1. < tau) & (tau <= 1 + np.sqrt(6.)))  # GT balanced pulse
    tau1 = tau[sigint1]
    tau2 = tau[sigint2]

    p_GT_H[sigint1] = 1. + (1-tau1)*np.log(tau1+EPSILON) - (1-tau1)*np.log(1-tau1+EPSILON)
    p_GT_H21 = (a-1)/6. * (a*(2*a+5) - 1 + 6*tau2**2 - 3*tau2*(1+3*a))
    p_GT_H22 = (tau2-1)*(a-tau2)**2 * (np.log(a-tau2+EPSILON) - np.log(tau2-1+EPSILON))
    p_GT_H[sigint2] = 1./6. * (p_GT_H21 + p_GT_H22)
    p_GT_H /= np.pi

    return p_GT_H


def gt_blast_center_fast(frequency_peak_hz: float = 6.3,
                         sample_rate_hz: float = 100.,
                         noise_std_loss_bits: float = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast computation of GT pulse with noise

    :param frequency_peak_hz: peak frequency, nominal 6.3 Hz for 1 tonne TNT
    :param sample_rate_hz: sample rate, nominal 100 Hz
    :param noise_std_loss_bits: noise loss relative to signal variance
    :return: centered time in seconds, GT pulse with white noise
    """

    duration_s = 16/frequency_peak_hz       # 16 cycles for 6th octave (M = 14)
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


def gt_blast_center_noise(duration_s: float = 16,
                          frequency_peak_hz: float = 6.3,
                          sample_rate_hz: float = 100,
                          noise_std_loss_bits: float = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast computation of GT pulse with noise for a specified duration in seconds

    :param duration_s: signal duration in seconds
    :param frequency_peak_hz: peak frequency, nominal 6.3 Hz for 1 tonne TNT
    :param sample_rate_hz: sample rate, nominal 100 Hz
    :param noise_std_loss_bits: noise loss relative to signal variance
    :return: centered time in seconds, GT pulse with white noise
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


def gt_blast_center_noise_uneven(sensor_epoch_s: np.array,
                                 noise_std_loss_bits: float = 2,
                                 frequency_center_hz: Optional[float] = None) -> np.ndarray:
    """
    Construct the GT explosion pulse of Garces (2019) for even or uneven sensor time
    in Gaussion noise with SNR in bits re signal STD.
    This is a very flexible variation.

    :param sensor_epoch_s: array with timestamps for signal in epoch seconds
    :param noise_std_loss_bits: number of bits below signal standard deviation. Default is 2
    :param frequency_center_hz: center frequency in Hz. Optional
    :return: numpy array with anti-aliased GT explosion pulse with Gaussian noise
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


# Integrals and derivatives, with delta function estimate and discontinuity boundary conditions
def gt_blast_derivative_period_center(time_center_s: np.ndarray,
                                      pseudo_period_s: float) -> np.ndarray:
    """
    Derivative of the GT blast with delta function approximation

    :param time_center_s: array with time
    :param pseudo_period_s: period in seconds
    :return: numpy ndarray with derivative of the GT blast with delta function approximation
    """
    # Garces (2019) ground truth GT blast pulse
    # with the +1, tau is the zero crossing time - time_start renamed to time_zero for first zero crossing.
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


def gt_blast_integral_period_center(time_center_s: np.ndarray,
                                    pseudo_period_s: float) -> np.ndarray:
    """
    Integral of the GT blast with initial condition at zero crossing

    :param time_center_s: array with time
    :param pseudo_period_s: period in seconds
    :return: numpy ndarray with integral of GT blast pulse
    """
    # Garces (2019) ground truth GT blast pulse
    # with the +1, tau is the zero crossing time - time_start renamed to time_zero for first zero crossing.
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


def gt_blast_center_integral_and_derivative(frequency_peak_hz,
                                            sample_rate_hz: float) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integral and derivative of GT pulse relative to tau (NOT time_s)

    :param frequency_peak_hz: peak frequency in Hz
    :param sample_rate_hz: sample rate in Hz
    :return: tau center, numpy ndarray with GT blast pulse, numpy ndarray with integral of GT blast pulse, numpy array
        with derivative of GT blast pulse
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


def gt_blast_ft(frequency_peak_hz: float,
                frequency_hz: Union[float, np.ndarray]) -> Union[float, complex, np.ndarray]:
    """
    Fourier transform of the GT blast pulse

    :param frequency_peak_hz: peak frequency in Hz
    :param frequency_hz: frequency in Hz, float or np.ndarray
    :return: Fourier transform of the GT blast pulse
    """
    w_scaled = 0.5*np.pi*frequency_hz/frequency_peak_hz
    ft_G17_positive = (1. - 1j*w_scaled - np.exp(-1j*w_scaled))/w_scaled**2.
    ft_G17_negative = np.exp(-1j*w_scaled*(1+np.sqrt(6.)))/(3.*w_scaled**4.) * \
                      (1j*w_scaled*np.sqrt(6.) + 3. +
                       np.exp(1j*w_scaled*np.sqrt(6.))*(3.*w_scaled**2. + 1j*w_scaled*2.*np.sqrt(6.)-3.))
    ft_G17 = (ft_G17_positive + ft_G17_negative)*np.pi/(2*np.pi*frequency_peak_hz)
    return ft_G17


def gt_blast_spectral_density(frequency_peak_hz: float,
                              frequency_hz: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], float]:
    """
    Spectral density of the GT blast pulse

    :param frequency_peak_hz: peak frequency in Hz
    :param frequency_hz: frequency in Hz, float or np.ndarray
    :return: spectral_density, spectral_density_peak
    """
    fourier_tx = gt_blast_ft(frequency_peak_hz, frequency_hz)
    spectral_density = 2*np.abs(fourier_tx*np.conj(fourier_tx))
    spectral_density_peak = np.max(spectral_density)
    return spectral_density, spectral_density_peak
