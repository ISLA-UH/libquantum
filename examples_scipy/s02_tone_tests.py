"""
libquantum example 2: 02_tone_amplitude_check.py
Illustrate TFRs on tones and compare amplitudes
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import atoms, entropy, scales, synthetics, utils  # spectra,
# import libquantum.plot_templates.plot_time_frequency_reps as pltq  # white background
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq
from typing import Tuple



def duration_points(sample_rate_hz: float, time_s: float) -> Tuple[int, int, int]:
    """
    Compute number of points
    :param sample_rate_hz: sample rate in Hz
    :param time_s: time scale, period or duration
    :return: number of points, floor and ceiling of log2 of number of points
    """
    points_float: float = sample_rate_hz * time_s
    points_int: int = int(points_float)
    points_floor_log2: int = int(np.floor(np.log2(points_float)))
    points_ceil_log2: int = int(np.ceil(np.log2(points_float)))

    return points_int, points_floor_log2, points_ceil_log2


def duration_ceil(sample_rate_hz: float, time_s: float) -> Tuple[int, int, float]:
    """
    Compute ceiling of the number of points, and convert to seconds
    :param sample_rate_hz: sample rate in Hz
    :param time_s: time scale, period or duration
    :return: ceil of log 2 of number of points, power of two number of points, corresponding time in s
    """
    points_float: float = sample_rate_hz * time_s
    points_ceil_log2: int = int(np.ceil(np.log2(points_float)))
    points_ceil_pow2: int = 2**points_ceil_log2
    time_ceil_pow2_s: float = points_ceil_pow2 / sample_rate_hz
    
    return points_ceil_log2, points_ceil_pow2, time_ceil_pow2_s


def duration_floor(sample_rate_hz: float, time_s: float) -> Tuple[int, int, float]:
    """
    Compute floor of the number of points, and convert to seconds
    :param sample_rate_hz: sample rate in Hz
    :param time_s: time scale, period or duration
    :return: floor of log 2 of number of points, power of two number of points, corresponding time in s
    """
    points_float: float = sample_rate_hz * time_s
    points_floor_log2: int = int(np.floor(np.log2(points_float)))
    points_floor_pow2: int = 2**points_floor_log2
    time_floor_pow2_s: float = points_floor_pow2 / sample_rate_hz

    return points_floor_log2, points_floor_pow2, time_floor_pow2_s


if __name__ == "__main__":
    """
    # The primary goal of standardization is to permit multimodal sensor analysis for different sample rates
    # For a specified signal duration, there is only one key parameter: Order
    """

    print('Tone synthetic')
    order_number_input = 12
    EVENT_NAME = "Tone Test"
    station_id_str = 'Dyad_Sig'
    sig_is_pow2 = True

    sig_frequency_hz = 50.
    sig_sample_rate_hz = 800.
    sig_duration_nominal_s = 5.

    ref_frequency_hz = sig_frequency_hz/4.
    ref_period_nominal_s = 1./ref_frequency_hz

    ref_period_points_ceil_log2, ref_period_points_ceil_pow2, ref_period_ceil_pow2_s = \
        duration_ceil(sample_rate_hz=sig_sample_rate_hz, time_s=ref_period_nominal_s)

    # Convert to function calls
    # Gabor Multiplier
    sig_cycles_M = 2*np.sqrt(2*np.log(2))*order_number_input
    sig_duration_min_s = sig_cycles_M/sig_frequency_hz

    # Verify request
    if sig_duration_nominal_s < sig_duration_min_s:
        sig_duration_nominal_s = 2*sig_duration_min_s

    sig_duration_points_log2, sig_duration_points_pow2, sig_duration_pow2_s = \
        duration_ceil(sample_rate_hz=sig_sample_rate_hz, time_s=sig_duration_nominal_s)

    # Construct synthetic tone with max unit amplitude
    mic_sig_epoch_s = np.arange(sig_duration_points_pow2) / sig_sample_rate_hz
    mic_sig = np.sin(2*np.pi*sig_frequency_hz*mic_sig_epoch_s)
    mic_sig += synthetics.white_noise_fbits(sig=mic_sig, std_bit_loss=4.)

    mic_sig *= utils.taper_tukey(mic_sig_epoch_s, fraction_cosine=0.1)  # add taper
    synthetics.antialias_halfNyquist(mic_sig)  # Antialias filter synthetic

    # Q atom lowest scale
    _, min_frequency_hz = scales.from_duration(band_order_Nth=order_number_input,
                                               sig_duration_s=sig_duration_pow2_s)

    print('\nRequest Order N =', order_number_input)
    print('Reference frequency, hz:', 1/ref_period_ceil_pow2_s)
    print('Lowest frequency in hz that can support this order for this signal duration:', min_frequency_hz)
    print('Nyquist frequency:', sig_sample_rate_hz/2)
    print('Scale with signal duration and to Nyquist, default G2 base re F1')

    # Select plot frequencies
    fmin = np.ceil(min_frequency_hz)
    fmax = sig_sample_rate_hz/2  # Nyquist

    # TFR SECTION
    # Compute complex wavelet transform (cwt) from signal duration
    mic_cwt, mic_cwt_bits, mic_cwt_time_s, mic_cwt_frequency_hz = \
        atoms.cwt_chirp_from_sig(sig_wf=mic_sig,
                                 frequency_sample_rate_hz=sig_sample_rate_hz,
                                 band_order_Nth=order_number_input,
                                 dictionary_type="tone")

    mic_cwt_snr, mic_cwt_snr_bits, mic_cwt_snr_entropy = entropy.snr_mean_max(tfr_coeff_complex=mic_cwt)

    # Scipy TFR
    # compute raw spectrogram with scipy package
    mic_stft_frequency_hz, mic_stft_time_s, mic_stft = \
        signal.spectrogram(x=mic_sig,
                           fs=sig_sample_rate_hz,
                           window='hann',
                           nperseg=ref_period_points_ceil_pow2,
                           noverlap=ref_period_points_ceil_pow2//2,
                           nfft=ref_period_points_ceil_pow2,
                           detrend='constant',
                           return_onesided=True,
                           scaling='density',
                           axis=-1,
                           mode='psd')

    mic_stft_bits = np.log2(np.abs(mic_stft))
    mic_stft_snr, mic_stft_snr_bits, mic_stft_snr_entropy = entropy.snr_mean_max(tfr_coeff_complex=mic_stft)

    pltq.plot_wf_mesh_mesh_vert(redvox_id=station_id_str,
                                wf_panel_2_sig=mic_sig,
                                wf_panel_2_time=mic_sig_epoch_s,
                                mesh_time=mic_cwt_time_s,
                                mesh_frequency=mic_cwt_frequency_hz,
                                mesh_panel_1_trf=mic_cwt_bits,
                                mesh_panel_1_colormap_scaling="range",
                                mesh_panel_0_tfr=mic_cwt_snr_entropy,
                                wf_panel_2_units="Norm",
                                mesh_panel_1_cbar_units="bits",
                                mesh_panel_0_cbar_units="eSNR bits",
                                start_time_epoch=0,
                                figure_title="CWT for " + EVENT_NAME,
                                frequency_hz_ymin=fmin,
                                frequency_hz_ymax=fmax)

    pltq.plot_wf_mesh_mesh_vert(redvox_id=station_id_str,
                                wf_panel_2_sig=mic_sig,
                                wf_panel_2_time=mic_sig_epoch_s,
                                mesh_time=mic_stft_time_s,
                                mesh_frequency=mic_stft_frequency_hz,
                                mesh_panel_1_trf=mic_stft_bits,
                                mesh_panel_1_colormap_scaling="range",
                                mesh_panel_0_tfr=mic_stft_snr_entropy,
                                wf_panel_2_units="Norm",
                                mesh_panel_1_cbar_units="bits",
                                mesh_panel_0_cbar_units="eSNR bits",
                                start_time_epoch=0,
                                figure_title="stft for " + EVENT_NAME,
                                frequency_hz_ymin=fmin,
                                frequency_hz_ymax=fmax)

    plt.show()
