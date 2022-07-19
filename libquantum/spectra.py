"""
This module calculates spectra: STFT, FFT, CQT
"""

import numpy as np
import scipy.signal as signal
import librosa
from libquantum import atoms, scales, utils
from libquantum.scales import EPSILON
from typing import Union, Tuple


def quantum_gauss_stdev(points_number: int) -> float:
    """
    Calculate Gauss Standard Deviation based on the input number of points

    :param points_number: number of points
    :return: float with standard deviation
    """
    gauss_stdev = 0.5*points_number/np.pi
    return gauss_stdev


def q_gauss(points_number: int) -> np.ndarray:
    """
    Calculate Gauss envelope

    :param points_number: number of points
    :return: numpy array with envelope values
    """
    gauss_stdev = quantum_gauss_stdev(points_number)
    gauss_amp = np.pi**(-0.25)
    gauss_envelope = gauss_amp*signal.get_window(('gaussian', gauss_stdev), points_number)
    return gauss_envelope


def cqt_scaling(band_order_Nth: float,
                scale_frequency_center_hz: float,
                frequency_sample_rate_hz: float,
                tfr_shape: tuple,
                dictionary_type: str = "norm") -> Union[np.ndarray, float]:
    """
    Calculate scale for CQT

    :param band_order_Nth: Nth order of constant Q bands
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :param tfr_shape: shape of TFR, Tuple
    :param dictionary_type: "tone" or "norm". Default is 'norm'
    :return: numpy array with scale values if dictionary_type
    """
    atom_scales = atoms.chirp_scale_from_order(band_order_Nth,
                                               scale_frequency_center_hz,
                                               frequency_sample_rate_hz)
    if dictionary_type == "tone":
        # The sqrt(2) factor reconciled the rms tone amplitude.
        # That way a 15 bit input amplitude returns 15 bit log2 Sxx
        atom_amp = np.sqrt(2)*(4*np.pi*atom_scales**2)**(-0.25)  # See Eq. A15 of Garces 2020
        cqt_multipier = utils.just_tile(atom_amp, tfr_shape)
    else:
        cqt_multipier = 1.
    return cqt_multipier


def cqt_from_sig(sig_wf: np.ndarray,
                 frequency_sample_rate_hz: float,
                 band_order_Nth: float,
                 cqt_window: str = 'hann',
                 dictionary_type: str = "norm") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the constant-Q transform of a signal.

    :param sig_wf: array with input signal
    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :param band_order_Nth: Nth order of constant Q bands
    :param cqt_window: string, "cqt_gauss" or librosa window specification for the basis filter. Default is 'hann'
    :param dictionary_type: "tone" or "norm". Default is 'norm'
    :return: four numpy ndarrays with CQT, CQT_bits, time_cqt_s, frequency_cqt_hz
    """
    sig_duration_s = len(sig_wf)/frequency_sample_rate_hz
    min_scale_s, min_frequency_hz = scales.from_duration(band_order_Nth, sig_duration_s)

    # Match default cwt
    cqt_points_hop_min, frequency_hz_center_min, scale_number_bins, order_Nth, cqt_points_per_seg_max = \
        scales.cqt_frequency_bands_g2f1(band_order_Nth,
                                        min_frequency_hz,
                                        frequency_sample_rate_hz,
                                        is_power_2=False)

    print('CQT Duration, NFFT, HOP:', len(sig_wf), cqt_points_per_seg_max, cqt_points_hop_min)
    int_order_N = int(band_order_Nth)
    # CQT is not power
    if cqt_window == "cqt_gauss":
        CQT = librosa.core.cqt(sig_wf, sr=frequency_sample_rate_hz, hop_length=cqt_points_hop_min,
                               fmin=frequency_hz_center_min, n_bins=scale_number_bins,
                               bins_per_octave=int_order_N, tuning=0.0,
                               filter_scale=1, norm=1, sparsity=0.0, window=q_gauss,
                               scale=True, pad_mode='reflect')
    else:
        CQT = librosa.core.cqt(sig_wf, sr=frequency_sample_rate_hz, hop_length=cqt_points_hop_min,
                               fmin=frequency_hz_center_min, n_bins=scale_number_bins,
                               bins_per_octave=int_order_N, tuning=0.0,
                               filter_scale=1, norm=1, sparsity=0.0, window=cqt_window,
                               scale=True, pad_mode='reflect')

    time_cqt_s = librosa.times_like(CQT, sr=frequency_sample_rate_hz, hop_length=cqt_points_hop_min)
    frequency_cqt_hz = librosa.core.cqt_frequencies(n_bins=scale_number_bins, fmin=frequency_hz_center_min,
                                                    bins_per_octave=int_order_N, tuning=0.0)
    cqt_multiplier = cqt_scaling(band_order_Nth,
                                 frequency_cqt_hz,
                                 frequency_sample_rate_hz,
                                 CQT.shape,
                                 dictionary_type)
    CQT *= cqt_multiplier
    CQT_bits = utils.log2epsilon(CQT)

    return CQT, CQT_bits, time_cqt_s, frequency_cqt_hz


def stft_from_sig(sig_wf: np.ndarray,
                  frequency_sample_rate_hz: float,
                  band_order_Nth: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Librosa STFT is complex FFT grid, not power

    :param sig_wf: array with input signal
    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :param band_order_Nth: Nth order of constant Q bands
    :return: four numpy ndarrays with STFT, STFT_bits, time_stft_s, frequency_stft_hz
    """

    sig_duration_s = len(sig_wf)/frequency_sample_rate_hz
    _, min_frequency_hz = scales.from_duration(band_order_Nth, sig_duration_s)

    order_Nth, cycles_M, quality_Q, \
    frequency_center, frequency_start, frequency_end = \
        scales.frequency_bands_g2f1(scale_order_input=band_order_Nth,
                                    frequency_low_input=min_frequency_hz,
                                    frequency_sample_rate_input=frequency_sample_rate_hz)

    # Choose the spectral resolution as the key parameter
    frequency_resolution_min_hz = np.min(frequency_end - frequency_start)
    frequency_resolution_max_hz = np.max(frequency_end - frequency_start)
    frequency_resolution_hz_geo = np.sqrt(frequency_resolution_min_hz*frequency_resolution_max_hz)
    stft_time_duration_s = 1/frequency_resolution_hz_geo
    stft_points_per_seg = int(frequency_sample_rate_hz*stft_time_duration_s)

    # From CQT
    stft_points_hop, _, _, _, _ = \
        scales.cqt_frequency_bands_g2f1(band_order_Nth,
                                        min_frequency_hz,
                                        frequency_sample_rate_hz,
                                        is_power_2=False)

    print('STFT Duration, NFFT, HOP:', len(sig_wf), stft_points_per_seg, stft_points_hop)

    STFT_Scaling = 2*np.sqrt(np.pi)/stft_points_per_seg
    STFT = librosa.core.stft(sig_wf, n_fft=stft_points_per_seg,
                             hop_length=stft_points_hop, win_length=None,
                             window='hann', center=True, pad_mode='reflect')

    # Must be scaled to match scipy psd
    STFT *= STFT_Scaling
    STFT_bits = utils.log2epsilon(STFT)

    time_stft_s = librosa.times_like(STFT, sr=frequency_sample_rate_hz,
                                     hop_length=stft_points_hop)
    frequency_stft_hz = librosa.core.fft_frequencies(sr=frequency_sample_rate_hz,
                                                     n_fft=stft_points_per_seg)

    return STFT, STFT_bits, time_stft_s, frequency_stft_hz


def stft_reassign_from_sig(sig_wf: np.ndarray,
                           frequency_sample_rate_hz: float,
                           band_order_Nth: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                           np.ndarray]:
    """
    Librosa STFT is complex FFT grid, not power
    Reassigned frequencies are not the same as the standard mesh frequencies

    :param sig_wf: array with input signal
    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :param band_order_Nth: Nth order of constant Q bands
    :return: six numpy ndarrays with STFT, STFT_bits, time_stft_s, frequency_stft_hz, time_stft_rsg_s,
        frequency_stft_rsg_hz
    """

    sig_duration_s = len(sig_wf)/frequency_sample_rate_hz
    _, min_frequency_hz = scales.from_duration(band_order_Nth, sig_duration_s)

    order_Nth, cycles_M, quality_Q, \
    frequency_center, frequency_start, frequency_end = \
        scales.frequency_bands_g2f1(scale_order_input=band_order_Nth,
                                    frequency_low_input=min_frequency_hz,
                                    frequency_sample_rate_input=frequency_sample_rate_hz)

    # Choose the spectral resolution as the key parameter
    frequency_resolution_min_hz = np.min(frequency_end - frequency_start)
    frequency_resolution_max_hz = np.max(frequency_end - frequency_start)
    frequency_resolution_hz_geo = np.sqrt(frequency_resolution_min_hz*frequency_resolution_max_hz)
    stft_time_duration_s = 1/frequency_resolution_hz_geo
    stft_points_per_seg = int(frequency_sample_rate_hz*stft_time_duration_s)

    # From CQT
    stft_points_hop, _, _, _, _ = \
        scales.cqt_frequency_bands_g2f1(band_order_Nth,
                                        min_frequency_hz,
                                        frequency_sample_rate_hz,
                                        is_power_2=False)

    print('Reassigned STFT Duration, NFFT, HOP:', len(sig_wf), stft_points_per_seg, stft_points_hop)

    STFT_Scaling = 2*np.sqrt(np.pi)/stft_points_per_seg

    # Reassigned frequencies require a 'best fit' solution.
    frequency_stft_rsg_hz, time_stft_rsg_s, STFT_mag = \
        librosa.reassigned_spectrogram(sig_wf, sr=frequency_sample_rate_hz,
                                       n_fft=stft_points_per_seg,
                                       hop_length=stft_points_hop, win_length=None,
                                       window='hann', center=False, pad_mode='reflect')

    # Must be scaled to match scipy psd
    STFT_mag *= STFT_Scaling
    STFT_bits = utils.log2epsilon(STFT_mag)

    # Standard mesh times and frequencies for plotting - nice to have both
    time_stft_s = librosa.times_like(STFT_mag, sr=frequency_sample_rate_hz,
                                     hop_length=stft_points_hop)
    frequency_stft_hz = librosa.core.fft_frequencies(sr=frequency_sample_rate_hz,
                                                     n_fft=stft_points_per_seg)

    # Reassigned frequencies are not the same as the standard mesh frequencies
    return STFT_mag, STFT_bits, time_stft_s, frequency_stft_hz, time_stft_rsg_s, frequency_stft_rsg_hz


# GENERAL FFT TOOLS
def fft_real_dB(sig: np.ndarray,
                sample_interval_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    FFT, real frequencies only, magnitude in dB

    :param sig: array with input signal
    :param sample_interval_s: sample interval in seconds
    :return: four numpy ndarrays with fft_frequency_pos, fft_sig_pos, fft_spectral_power_pos_dB,
        fft_spectral_phase_radians
    """
    fft_points = len(sig)
    fft_sig_pos = np.fft.rfft(sig)
    # returns correct RMS power level sqrt(2) -> 1
    fft_sig_pos /= fft_points
    fft_frequency_pos = np.fft.rfftfreq(fft_points, d=sample_interval_s)
    fft_spectral_power_pos_dB = 10.*np.log10(2.*(np.abs(fft_sig_pos))**2. + EPSILON)
    fft_spectral_phase_radians = np.angle(fft_sig_pos)
    return fft_frequency_pos, fft_sig_pos, fft_spectral_power_pos_dB, fft_spectral_phase_radians


def fft_real_bits(sig: np.ndarray,
                  sample_interval_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    FFT, real frequencies only, magnitude in bits
    :param sig: array with input signal
    :param sample_interval_s: sample interval in seconds
    :return: four numpy ndarrays with fft_frequency_pos, fft_sig_pos, fft_spectral_power_pos_bits,
        fft_spectral_phase_radians
    """
    # FFT for sigetic, by the book
    fft_points = len(sig)
    fft_sig_pos = np.fft.rfft(sig)
    # returns correct RMS power level sqrt(2) -> 1
    fft_sig_pos /= fft_points
    fft_frequency_pos = np.fft.rfftfreq(fft_points, d=sample_interval_s)
    fft_spectral_power_pos_bits = utils.log2epsilon(2.*np.abs(fft_sig_pos))
    fft_spectral_phase_radians = np.angle(fft_sig_pos)
    return fft_frequency_pos, fft_sig_pos, fft_spectral_power_pos_bits, fft_spectral_phase_radians


def ifft_real(fft_sig_pos) -> np.ndarray:
    """
    Calculate the inverse of the one-dimensional discrete Fourier Transform for real input.

    :param fft_sig_pos: input array
    :return: the truncated or zero-padded input, transformed along the axis
    """
    ifft_sig = np.fft.irfft(fft_sig_pos).real
    fft_points = len(ifft_sig)
    ifft_sig *= fft_points
    return ifft_sig


def fft_complex_bits(sig: np.ndarray,
                     sample_interval_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the one-dimensional discrete Fourier Transform in bits

    :param sig: array with input signal
    :param sample_interval_s: sample interval in seconds
    :return: four numpy arrays with fft_frequency, fft_sig, fft_spectral_bits, fft_spectral_phase_radians
    """
    # FFT for sigetic, by the book
    fft_points = len(sig)
    fft_sig = np.fft.fft(sig)
    # returns correct RMS power level
    fft_sig /= fft_points
    fft_frequency = np.fft.fftfreq(fft_points, d=sample_interval_s)
    fft_spectral_bits = utils.log2epsilon(fft_sig)
    fft_spectral_phase_radians = np.angle(fft_sig)
    return fft_frequency, fft_sig, fft_spectral_bits, fft_spectral_phase_radians


# Inverse Fourier Transform FFT of real function
def ifft_complex(fft_sig_complex) -> np.ndarray:
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    :param fft_sig_complex: input array, can be complex.
    :return: the truncated or zero-padded input, transformed along the axis
    """
    ifft_sig = np.fft.ifft(fft_sig_complex)
    fft_points = len(ifft_sig)
    ifft_sig *= fft_points
    return ifft_sig


# Time shifted the FFT before inversion
def fft_time_shift(fft_sig: np.ndarray,
                   fft_frequency: np.ndarray,
                   time_lead: Union[int, float]) -> np.ndarray:
    """
    Time shift an FFT. Frequency and the time shift time_lead must have consistent units and be within window

    :param fft_sig: FFT signal
    :param fft_frequency: FFT frequencies
    :param time_lead: time shift
    :return: numpy ndarray with time shifted FFT signal
    """
    # frequency and the time shift time_lead must have consistent units and be within window
    fft_phase_time_shift = np.exp(-1j*2*np.pi*fft_frequency*time_lead)
    fft_sig *= fft_phase_time_shift
    return fft_sig


# Compute Welch  from spectrogram
def fft_welch_from_Sxx_bits(f_center: np.ndarray,
                            Sxx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch periodogram from spectrogram

    :param f_center: array of sample frequencies.
    :param Sxx: numpy array with spectogram
    :return: numpy array with frequencies, numpy array with Welch periodogram
    """
    # Estimate Welch periodogram by adding Sxx and dividing by the number of windows
    # Removes zero frequency
    Welch_Sxx = np.average(Sxx, axis=1)
    Welch_Sxx_bits = 0.5*utils.log2epsilon(Welch_Sxx[1:])
    f_center_nozero = f_center[1:]
    return f_center_nozero, Welch_Sxx_bits


def fft_welch_snr_power(f_center: np.ndarray,
                        Sxx: np.ndarray,
                        Sxx2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate SNR power from specogram

    :param f_center: array of sample frequencies.
    :param Sxx: numpy array with spectogram
    :param Sxx2: numpy array with spectogram 2. Must have the same length as Sxx
    :return: numpy ndarray with snr frequency, numpy ndarray with snr power
    """
    # Estimate Welch periodogram by adding Sxx and dividing by the number of windows
    # Removes zero frequency
    Welch_Sxx = np.average(Sxx[1:], axis=1)
    Welch_Sxx /= np.max(Sxx[1:])
    Welch_Sxx2 = np.average(Sxx2[1:], axis=1)
    Welch_Sxx2 /= np.max(Sxx2[1:])
    snr_power = Welch_Sxx2/Welch_Sxx
    snr_frequency = f_center[1:]
    return snr_frequency, snr_power
