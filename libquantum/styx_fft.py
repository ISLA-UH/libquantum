"""
This module calculates spectra: STFT, FFT
"""

import numpy as np
import scipy.signal as signal
from libquantum import atoms, scales, utils
from libquantum.scales import EPSILON
from typing import Union, Tuple


# TODO: Standardize inputs
def butter_bandpass(sig_wf: np.ndarray,
                    frequency_sample_rate_hz: float,
                    frequency_cut_low_hz,
                    frequency_cut_high_hz,
                    filter_order: int = 4,
                    tukey_alpha: float = 0.5):
    """
    Buterworth bandpass filter
    :param sig_wf:
    :param frequency_sample_rate_hz:
    :param frequency_cut_low_hz:
    :param frequency_cut_high_hz:
    :param filter_order:
    :param tukey_alpha:
    :return:
    """

    nyquist = 0.5 * frequency_sample_rate_hz
    edge_low = frequency_cut_low_hz / nyquist
    edge_high = frequency_cut_high_hz / nyquist
    if edge_high >= 1:
        edge_high = 0.5  # Half of nyquist
    [b, a] = signal.butter(N=filter_order,
                           Wn=[edge_low, edge_high],
                           btype='bandpass')
    sig_taper = np.copy(sig_wf)
    sig_taper = sig_taper * signal.windows.tukey(M=len(sig_taper), alpha=tukey_alpha)
    sig_bandpass = signal.filtfilt(b, a, sig_taper)

    return sig_bandpass


def butter_highpass(sig_wf: np.ndarray,
                    frequency_sample_rate_hz: float,
                    frequency_cut_low_hz,
                    filter_order: int = 4,
                    tukey_alpha: float = 0.5):
    """
    Buterworth bandpass filter
    :param sig_wf:
    :param frequency_sample_rate_hz:
    :param frequency_cut_low_hz:
    :param filter_order:
    :param tukey_alpha:
    :return:
    """

    nyquist = 0.5 * frequency_sample_rate_hz
    edge_low = frequency_cut_low_hz / nyquist
    if edge_low >= 1:
        print('Cutoff greater than Nyquist')
        exit()
    [b, a] = signal.butter(N=filter_order,
                           Wn=[edge_low],
                           btype='highpass')
    sig_taper = np.copy(sig_wf)
    sig_taper = sig_taper * signal.windows.tukey(M=len(sig_taper), alpha=tukey_alpha)
    sig_highpass = signal.filtfilt(b, a, sig_taper)

    return sig_highpass


def stft_complex_pow2(sig_wf: np.ndarray,
                      frequency_sample_rate_hz: float,
                      nfft_points: int,
                      alpha: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Simplest, with 50% overlap and build-in defaults """
    frequency_stft_hz, time_stft_s, stft_complex = \
        signal.stft(x=sig_wf,
                    fs=frequency_sample_rate_hz,
                    window=('tukey', alpha),
                    nperseg=nfft_points,
                    noverlap=nfft_points // 2,
                    nfft=nfft_points,
                    detrend='constant',
                    return_onesided=True,
                    axis=-1,
                    boundary='zeros',
                    padded=True)

    return frequency_stft_hz, time_stft_s, stft_complex


def welch_power_pow2(sig_wf: np.ndarray,
                     frequency_sample_rate_hz: float,
                     nfft_points: int,
                     alpha: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:

    # Compute the Welch PSD; averaged spectrum over sliding windows
    frequency_welch_hz, welch_power = \
        signal.welch(x=sig_wf,
                     fs=frequency_sample_rate_hz,
                     window=('tukey', alpha),
                     nperseg=nfft_points,
                     noverlap=nfft_points // 2,
                     nfft=nfft_points,
                     detrend='constant',
                     return_onesided=True,
                     axis=-1,
                     scaling='spectrum',
                     average='mean')
    return frequency_welch_hz, welch_power


def power_and_information_shannon_stft(stft_complex):
    """
    Computes power and information metrics
    :param stft_complex:
    :return:
    """
    power = 2 * np.abs(np.copy(stft_complex)) ** 2
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


def power_and_information_shannon_welch(welch_power):
    """
    Computes power and information metrics
    :param welch_power:
    :return:
    """
    power = np.copy(welch_power)
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

# def stft_from_sig(sig_wf: np.ndarray,
#                   frequency_sample_rate_hz: float,
#                   frequency_averaging_hz: float = 1.,
#                   hop_fraction: float = 0.5,
#                   tukey_alpha: float = 0.5,
#                   band_order_Nth: float = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Fast STFT estimate using Scipy Spectrogram. See examples s01-s04
#
#     :param sig_wf: array with input signal
#     :param frequency_sample_rate_hz: sample rate of frequency in Hz
#     :param frequency_averaging_hz: baseline lowest frequency, sets spectral resolution in FFT
#     :param hop_fraction: hop length as a fraction of the window, default of 50%
#     :param tukey_alpha: percent of the Tukey window with a cosine envelope, default of 50%
#     :param band_order_Nth: Nth order of constant Q bands, default of 3
#     :return: four numpy ndarrays with STFT, STFT_bits, time_stft_s, frequency_stft_hz
#     """
#
#     # Check
#     if hop_fraction >= 1. or hop_fraction <= 0.:
#         hop_fraction = 0.5
#     if tukey_alpha >= 1. or tukey_alpha <= 0:
#         tukey_alpha = 0.5
#
#     # NaNs, masks, etc. (if any) should be taken care of by this stage
#     sig_duration_s = len(sig_wf)/frequency_sample_rate_hz
#     _, min_frequency_hz = scales.from_duration(band_order_Nth, sig_duration_s)
#
#     if min_frequency_hz < frequency_averaging_hz:
#         # Override
#         min_frequency_hz = frequency_averaging_hz
#
#     # Compute constant Q frequency bands for the specified order for later comparison
#     order_Nth, cycles_M, quality_Q, \
#     frequency_center, frequency_start, frequency_end = \
#         scales.frequency_bands_g2f1(scale_order_input=band_order_Nth,
#                                     frequency_low_input=min_frequency_hz,
#                                     frequency_sample_rate_input=frequency_sample_rate_hz)
#
#     # Choose the spectral resolution as the key parameter
#     frequency_resolution_min_hz = np.min(frequency_end - frequency_start)
#     frequency_resolution_max_hz = np.max(frequency_end - frequency_start)
#     frequency_resolution_hz_geo = np.sqrt(frequency_resolution_min_hz*frequency_resolution_max_hz)
#     stft_time_duration_s = 1/frequency_resolution_hz_geo
#     stft_points_per_seg = int(frequency_sample_rate_hz*stft_time_duration_s)
#     # From hop fraction
#     stft_points_hop = int(hop_fraction*stft_points_per_seg)
#
#     print('STFT Duration, NFFT, HOP:', len(sig_wf), stft_points_per_seg, stft_points_hop)
#
#     # Compute the spectrogram with the spectrum/psd options
#     frequency_stft_hz, time_stft_s, sig_stft_psd_spec = \
#         signal.spectrogram(x=sig_wf,
#                            fs=frequency_sample_rate_hz,
#                            window=('tukey', tukey_alpha),
#                            nperseg=stft_points_per_seg,
#                            noverlap=stft_points_hop,
#                            nfft=stft_points_per_seg,
#                            detrend='constant',
#                            return_onesided=True,
#                            axis=-1,
#                            scaling='spectrum',
#                            mode='psd')
#
#     # Must be scaled to match scipy psd
#     stft_abs = np.sqrt(np.abs(sig_stft_psd_spec))
#     # TODO: Reconsider
#     # Unit tone amplitude -> 1/2 bit in log2 STFT amplitude (not power)
#     stft_bits = utils.log2epsilon(stft_abs)
#
#     return stft_abs, stft_bits, time_stft_s, frequency_stft_hz
#
#
# def stft_core(sig_wf: np.ndarray,
#               frequency_sample_rate_hz: float,
#               frequency_averaging_hz: float = 1.,
#               hop_fraction: float = 0.5,
#               tukey_alpha: float = 0.5,
#               band_order_Nth: float = 3,
#               ave_q_resolution = 'False') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Fast STFT estimate using Scipy stft. Baseline algorithms; potentially TMI but provides max
#     parameter control and clear documentation.
#
#     :param sig_wf: array with input signal
#     :param frequency_sample_rate_hz: sample rate of frequency in Hz
#     :param frequency_averaging_hz: baseline lowest frequency, sets spectral resolution in FFT
#     :param hop_fraction: hop length as a fraction of the window, default of 50%
#     :param tukey_alpha: percent of the Tukey window with a cosine envelope, default of 50%
#     :param band_order_Nth: Nth order of constant Q bands, default of 3
#     :return: four numpy ndarrays with STFT, STFT_bits, time_stft_s, frequency_stft_hz
#     """
#
#     # Check
#     if hop_fraction >= 1. or hop_fraction <= 0.:
#         hop_fraction = 0.5
#     if tukey_alpha >= 1. or tukey_alpha <= 0:
#         tukey_alpha = 0.5
#
#     # NaNs, masks, etc. (if any) should be taken care of by this stage
#     sig_duration_points = len(sig_wf)
#     sig_duration_s = sig_duration_points/frequency_sample_rate_hz
#     _, min_frequency_hz = scales.from_duration(band_order_Nth, sig_duration_s)
#
#     if min_frequency_hz < frequency_averaging_hz:
#         # Override
#         min_frequency_hz = frequency_averaging_hz
#
#     # Default
#     stft_time_duration_s = 1/min_frequency_hz
#
#     if ave_q_resolution:
#         # Compute constant Q frequency bands for the specified order for later comparison.
#         # Override time duration
#         order_Nth, cycles_M, quality_Q, \
#         frequency_center, frequency_start, frequency_end = \
#             scales.frequency_bands_g2f1(scale_order_input=band_order_Nth,
#                                         frequency_low_input=min_frequency_hz,
#                                         frequency_sample_rate_input=frequency_sample_rate_hz)
#
#         # Choose the spectral resolution as the key parameter
#         frequency_resolution_min_hz = np.min(frequency_end - frequency_start)
#         frequency_resolution_max_hz = np.max(frequency_end - frequency_start)
#         frequency_resolution_hz_geo = np.sqrt(frequency_resolution_min_hz*frequency_resolution_max_hz)
#         stft_time_duration_s = 1/frequency_resolution_hz_geo
#
#     stft_points_per_seg = int(frequency_sample_rate_hz*stft_time_duration_s)
#     # From hop fraction
#     stft_points_hop = int(hop_fraction*stft_points_per_seg)
#     nfft_points_per_seg = stft_points_per_seg
#     if not utils.is_power_of_two(stft_points_per_seg):
#         nfft_points_per_seg = 2**int(np.ceil(np.log2(stft_points_per_seg)))
#
#     print('STFT Duration, NPPS, HOP, NFFT:', len(sig_wf), stft_points_per_seg, stft_points_hop, nfft_points_per_seg)
#     # Compute the spectrogram with the spectrum/psd options
#     frequency_stft_hz, time_stft_s, sig_stft = \
#         signal.stft(x=sig_wf,
#                     fs=frequency_sample_rate_hz,
#                     window=('tukey', tukey_alpha),
#                     nperseg=stft_points_per_seg,
#                     noverlap=stft_points_hop,
#                     nfft=nfft_points_per_seg,
#                     detrend='constant',
#                     return_onesided=True,
#                     boundary='zeros',
#                     padded=True,
#                     axis=-1)
#
#     # Must be scaled to match scipy psd
#     stft_abs = np.abs(sig_stft)
#     # TODO: Reconsider
#     # Unit tone amplitude -> 1/2 bit in log2 STFT amplitude (not power)
#     stft_bits = utils.log2epsilon(stft_abs)
#
#     return sig_stft, stft_abs, stft_bits, time_stft_s, frequency_stft_hz
#
#
# # GENERAL FFT TOOLS
# def fft_real_dB(sig_wf: np.ndarray,
#                 sample_interval_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     FFT, real frequencies only, magnitude in dB. Revalidated 220620, MAG.
#
#     :param sig_wf: array with input signal
#     :param sample_interval_s: sample interval in seconds
#     :return: four numpy ndarrays with fft_frequency_pos, fft_sig_pos, fft_spectral_power_pos_dB,
#         fft_spectral_phase_radians
#     """
#     fft_points = len(sig_wf)
#     fft_sig_pos = np.fft.rfft(sig_wf)
#     # returns correct RMS power level sqrt(2) -> 1
#     fft_sig_pos /= fft_points
#     fft_frequency_pos = np.fft.rfftfreq(fft_points, d=sample_interval_s)
#     fft_spectral_power_pos_dB = 10.*np.log10(2.*(np.abs(fft_sig_pos))**2. + EPSILON)
#     fft_spectral_phase_radians = np.angle(fft_sig_pos)
#     return fft_frequency_pos, fft_sig_pos, fft_spectral_power_pos_dB, fft_spectral_phase_radians
#
#
# def fft_real_bits(sig_wf: np.ndarray,
#                   sample_interval_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     FFT, real frequencies only, magnitude in bits. Revalidated 220620, MAG.
#     :param sig_wf: array with input signal
#     :param sample_interval_s: sample interval in seconds
#     :return: four numpy ndarrays with fft_frequency_pos, fft_sig_pos, fft_spectral_power_pos_bits,
#         fft_spectral_phase_radians
#     """
#     # FFT for sigetic, by the book
#     fft_points = len(sig_wf)
#     fft_sig_pos = np.fft.rfft(sig_wf)
#     # returns correct RMS power level sqrt(2) -> 1
#     fft_sig_pos /= fft_points
#     fft_frequency_pos = np.fft.rfftfreq(fft_points, d=sample_interval_s)
#     fft_power = 2.*(np.abs(fft_sig_pos))**2.
#     # Unit tone amplitude -> 1/2 bit in log2 STFT amplitude (not power)
#     fft_spectral_power_pos_bits = utils.log2epsilon(np.sqrt(fft_power))
#     fft_spectral_phase_radians = np.angle(fft_sig_pos)
#     return fft_frequency_pos, fft_sig_pos, fft_spectral_power_pos_bits, fft_spectral_phase_radians
#
#
# def ifft_real(fft_sig_pos) -> np.ndarray:
#     """
#     Calculate the inverse of the one-dimensional discrete Fourier Transform for real input.
#
#     :param fft_sig_pos: input array
#     :return: the truncated or zero-padded input, transformed along the axis
#     """
#     ifft_sig = np.fft.irfft(fft_sig_pos).real
#     fft_points = len(ifft_sig)
#     ifft_sig *= fft_points
#     return ifft_sig
#
#
# def fft_complex_bits(sig_wf: np.ndarray,
#                      sample_interval_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute the one-dimensional discrete Fourier Transform in bits
#
#     :param sig_wf: array with input signal
#     :param sample_interval_s: sample interval in seconds
#     :return: four numpy arrays with fft_frequency, fft_sig, fft_spectral_bits, fft_spectral_phase_radians
#     """
#     # FFT for sigetic, by the book
#     fft_points = len(sig_wf)
#     fft_sig = np.fft.fft(sig_wf)
#     # returns correct RMS power level (not RFFT)
#     fft_sig /= fft_points
#     fft_frequency = np.fft.fftfreq(fft_points, d=sample_interval_s)
#     fft_spectral_bits = utils.log2epsilon(fft_sig)
#     fft_spectral_phase_radians = np.angle(fft_sig)
#     return fft_frequency, fft_sig, fft_spectral_bits, fft_spectral_phase_radians
#
#
# # Inverse Fourier Transform FFT of real function
# def ifft_complex(fft_sig_complex) -> np.ndarray:
#     """
#     Compute the one-dimensional inverse discrete Fourier Transform.
#
#     :param fft_sig_complex: input array, can be complex.
#     :return: the truncated or zero-padded input, transformed along the axis
#     """
#     ifft_sig = np.fft.ifft(fft_sig_complex)
#     fft_points = len(ifft_sig)
#     ifft_sig *= fft_points
#     return ifft_sig
#
#
# # Time shifted the FFT before inversion
# def fft_time_shift(fft_sig: np.ndarray,
#                    fft_frequency: np.ndarray,
#                    time_lead: Union[int, float]) -> np.ndarray:
#     """
#     Time shift an FFT. Frequency and the time shift time_lead must have consistent units and be within window
#
#     :param fft_sig: FFT signal
#     :param fft_frequency: FFT frequencies
#     :param time_lead: time shift
#     :return: numpy ndarray with time shifted FFT signal
#     """
#     # frequency and the time shift time_lead must have consistent units and be within window
#     fft_phase_time_shift = np.exp(-1j*2*np.pi*fft_frequency*time_lead)
#     fft_sig *= fft_phase_time_shift
#     return fft_sig
#
#
# # Compute Welch  from spectrogram
# def fft_welch_from_Sxx_bits(frequency_center: np.ndarray,
#                             Sxx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Compute Welch periodogram from spectrogram
#
#     :param frequency_center: array of sample frequencies.
#     :param Sxx: numpy array with spectrogram
#     :return: numpy array with frequencies, numpy array with Welch periodogram
#     """
#     # Estimate Welch periodogram by adding Sxx and dividing by the number of windows
#     # Removes zero frequency
#     Welch_Sxx = np.average(Sxx, axis=1)
#     Welch_Sxx_bits = 0.5*utils.log2epsilon(Welch_Sxx[1:])
#     f_center_nozero = frequency_center[1:]
#     return f_center_nozero, Welch_Sxx_bits
#
#
# def fft_welch_snr_power(f_center: np.ndarray,
#                         Sxx: np.ndarray,
#                         Sxx2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Calculate SNR power from specogram
#
#     :param f_center: array of sample frequencies.
#     :param Sxx: numpy array with spectogram
#     :param Sxx2: numpy array with spectogram 2. Must have the same length as Sxx
#     :return: numpy ndarray with snr frequency, numpy ndarray with snr power
#     """
#     # Estimate Welch periodogram by adding Sxx and dividing by the number of windows
#     # Removes zero frequency
#     Welch_Sxx = np.average(Sxx[1:], axis=1)
#     Welch_Sxx /= np.max(Sxx[1:])
#     Welch_Sxx2 = np.average(Sxx2[1:], axis=1)
#     Welch_Sxx2 /= np.max(Sxx2[1:])
#     snr_power = Welch_Sxx2/Welch_Sxx
#     snr_frequency = f_center[1:]
#     return snr_frequency, snr_power
#
#
