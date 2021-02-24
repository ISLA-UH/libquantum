import numpy as np
import scipy.signal as signal
import librosa
from libquantum import atoms, scales, utils
from libquantum.scales import EPSILON


# def mesh_peaks_from_passband(mesh_2d, frequency_1d, frequency_min, frequency_max):
#     """DEPRECATED, SEE TFR_GATE_PICK
#     """
#     index_frequency_hz_min_band = np.argmin(np.abs(frequency_1d - frequency_min))
#     index_frequency_hz_max_band = np.argmin(np.abs(frequency_1d - frequency_max))
#     mesh_max = np.max(mesh_2d[index_frequency_hz_min_band:index_frequency_hz_max_band, :],
#                       axis=0)
#     index_argmax = np.argmax(mesh_2d[index_frequency_hz_min_band:index_frequency_hz_max_band, :],
#                              axis=0)
#     frequency_peak = np.empty(index_argmax.shape)
#
#     for j in range(len(index_argmax)):
#         frequency_peak[j] = frequency_1d[index_frequency_hz_min_band + index_argmax[j]]
#
#     return mesh_max, frequency_peak


def quantum_gauss_stdev(points_number):
    # TODO: Revisit and nail it
    gauss_stdev = 0.5*points_number/np.pi
    return gauss_stdev


def q_gauss(points_number):
    gauss_stdev = quantum_gauss_stdev(points_number)
    gauss_amp = np.pi**(-0.25)
    gauss_envelope = gauss_amp*signal.get_window(('gaussian', gauss_stdev), points_number)
    return gauss_envelope


def cqt_scaling(band_order_Nth: float,
                scale_frequency_center_hz: float,
                frequency_sample_rate_hz: float,
                tfr_shape: tuple,
                dictionary_type: str = "norm"):

    # TODO: Clear up scaling. Tidy up math, then code.
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


def cqt_from_sig(sig_wf,
                 frequency_sample_rate_hz,
                 band_order_Nth,
                 cqt_window: str = 'hann',
                 dictionary_type: str = "norm"):
    sig_duration_s = len(sig_wf)/frequency_sample_rate_hz
    min_scale_s, min_frequency_hz = scales.from_duration(band_order_Nth, sig_duration_s)

    # Match default cwt
    cqt_points_hop_min, frequency_hz_center_min, scale_number_bins, order_Nth, cqt_points_per_seg_max = \
        scales.cqt_frequency_bands_g2f1(band_order_Nth,
                                        min_frequency_hz,
                                        frequency_sample_rate_hz,
                                        is_power_2=False)

    print('CQT Duration, NFFT, HOP:', len(sig_wf), cqt_points_per_seg_max, cqt_points_hop_min)
    # TODO: This will bomb for non-integer N
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
    frequency_cqt_hz = librosa.core.cqt_frequencies(scale_number_bins, frequency_hz_center_min,
                                                    bins_per_octave=int_order_N, tuning=0.0)
    cqt_multiplier = cqt_scaling(band_order_Nth,
                                 frequency_cqt_hz,
                                 frequency_sample_rate_hz,
                                 CQT.shape,
                                 dictionary_type)
    CQT *= cqt_multiplier
    # print('Max Hann CQT:', np.max(np.abs(CQT)))
    CQT_bits = utils.log2epsilon(CQT)

    return CQT, CQT_bits, time_cqt_s, frequency_cqt_hz


def stft_from_sig(sig_wf,
                  frequency_sample_rate_hz,
                  band_order_Nth):
    """Librosa STFT is complex FFT grid, not power"""

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
    frequency_resolution_hz_alg = np.mean(frequency_end - frequency_start)
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


# GENERAL FFT TOOLS
def fft_real_bits(sig, sample_interval_s):
    # FFT for sigetic, by the book
    fft_points = len(sig)
    fft_sig_pos = np.fft.rfft(sig)
    # returns correct RMS power level sqrt(2) -> 1
    fft_sig_pos /= fft_points
    fft_frequency_pos = np.fft.rfftfreq(fft_points, d=sample_interval_s)
    fft_spectral_power_pos_bits = utils.log2epsilon(2.*np.abs(fft_sig_pos))
    fft_spectral_phase_radians = np.angle(fft_sig_pos)
    return fft_frequency_pos, fft_sig_pos, \
           fft_spectral_power_pos_bits, fft_spectral_phase_radians


# Inverse Fourier Transform from positive FFT of a real function
def ifft_real(fft_sig_pos):
    ifft_sig = np.fft.irfft(fft_sig_pos).real
    fft_points = len(ifft_sig)
    ifft_sig *= fft_points
    return ifft_sig


def fft_complex_bits(sig, sample_interval_s):
    # FFT for sigetic, by the book
    fft_points = len(sig)
    fft_sig = np.fft.fft(sig)
    # returns correct RMS power level
    fft_sig /= fft_points
    fft_frequency = np.fft.fftfreq(fft_points, d=sample_interval_s)
    fft_spectral_bits = np.log2((np.abs(fft_sig) + EPSILON))
    fft_spectral_phase_radians = np.angle(fft_sig)
    return fft_frequency, fft_sig, fft_spectral_bits, fft_spectral_phase_radians


# Inverse Fourier Transform FFT of real function
def ifft_complex(fft_sig_complex):
    ifft_sig = np.fft.ifft(fft_sig_complex)
    fft_points = len(ifft_sig)
    ifft_sig *= fft_points
    return ifft_sig


# Time shifted the FFT before inversion
def fft_time_shift(fft_sig, fft_frequency, time_lead):
    # frequency and the time shift time_lead must have consistent units and be within window
    fft_phase_time_shift = np.exp(-1j*2*np.pi*fft_frequency*time_lead)
    fft_sig *= fft_phase_time_shift
    return fft_sig


# Compute Welch periodogram from spectrogram
def fft_welch_from_Sxx_bits(f_center, Sxx):
    # Estimate Welch periodogram by adding Sxx and dividing by the number of windows
    # Removes zero frequency
    Welch_Sxx = np.average(Sxx, axis=1)
    Welch_Sxx_bits = 0.5*np.log2(np.abs(Welch_Sxx[1:]))
    f_center_nozero = f_center[1:]
    return f_center_nozero, Welch_Sxx_bits


def fft_welch_snr_power(f_center, Sxx, Sxx2):
    # Estimate Welch periodogram by adding Sxx and dividing by the number of windows
    # Removes zero frequency
    Welch_Sxx = np.average(Sxx[1:], axis=1)
    Welch_Sxx /= np.max(Sxx[1:])
    Welch_Sxx2 = np.average(Sxx2[1:], axis=1)
    Welch_Sxx2 /= np.max(Sxx2[1:])
    snr_power = Welch_Sxx2/Welch_Sxx
    snr_frequency = f_center[1:]
    return snr_frequency, snr_power


def resample8K(sig: np.ndarray, frequency_sample_hz: float) -> np.ndarray:
    """
    Resample to 8k sample rate
    :param sig: time series
    :param frequency_sample_hz:
    :return: ndarray
    """
    frequency_resample_hz = 8000.
    sig_points = len(sig)
    resample_points = int(sig_points*frequency_resample_hz/frequency_sample_hz)
    sig_resample = signal.resample(sig, resample_points)
    return sig_resample
