"""
libquantum example 2: 02_tone_amplitude_check.py
Illustrate TFRs on tones and compare amplitudes
"""
import numpy as np
import matplotlib.pyplot as plt
from libquantum import atoms, entropy, scales, synthetics, utils  # spectra,
import libquantum.plot_templates.plot_time_frequency_reps as pltq
# import libwwz

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
    ref_period_points_log2: int = int(np.floor(np.log2(sig_sample_rate_hz / ref_frequency_hz)))
    ref_period_points_pow2: int = 2**ref_period_points_log2

    # Convert to function calls
    # Gabor Multiplier
    sig_cycles_M = 2*np.sqrt(2*np.log(2))*order_number_input
    sig_duration_min_s = sig_cycles_M/sig_frequency_hz

    if sig_duration_nominal_s < sig_duration_min_s:
        sig_duration_nominal_s = 2*sig_duration_min_s

    sig_duration_points_down: int = int(sig_sample_rate_hz * sig_duration_nominal_s)
    sig_duration_points_log2: int = int(np.ceil(np.log2(sig_sample_rate_hz * sig_duration_nominal_s)))
    sig_duration_points_pow2: int = 2**sig_duration_points_log2

    print(ref_period_points_log2, sig_duration_points_log2)

    # exit()

    if sig_is_pow2:
        sig_duration_points = sig_duration_points_pow2
        sig_duration_s = sig_duration_points_pow2/sig_frequency_hz
    else:
        sig_duration_points = sig_duration_points_down
        sig_duration_s = sig_duration_points_down/sig_frequency_hz

    # Construct synthetic tone with max unit amplitude
    mic_sig_epoch_s = np.arange(sig_duration_points) / sig_sample_rate_hz
    mic_sig = np.sin(2*np.pi*sig_frequency_hz*mic_sig_epoch_s)
    mic_sig += synthetics.white_noise_fbits(sig=mic_sig, std_bit_loss=4.)

    mic_sig *= utils.taper_tukey(mic_sig_epoch_s, fraction_cosine=0.1)  # add taper
    synthetics.antialias_halfNyquist(mic_sig)  # Antialias filter synthetic

    # Frame to mic start and end and plot
    event_reference_time_epoch_s = mic_sig_epoch_s[0]

    max_time_s, min_frequency_hz = scales.from_duration(band_order_Nth=order_number_input,
                                                        sig_duration_s=sig_duration_s)
    print('\nRequest Order N =', order_number_input)
    print('Lowest frequency in hz that can support this order for this signal duration is ', min_frequency_hz)
    print('Scale with signal duration and to Nyquist, default G2 base re F1')

    # Select plot frequencies
    fmin = np.ceil(min_frequency_hz)
    fmax = 400

    # TFR SECTION
    # Compute complex wavelet transform (cwt) from signal duration
    mic_cwt, mic_cwt_bits, mic_cwt_time_s, mic_cwt_frequency_hz = \
        atoms.cwt_chirp_from_sig(sig_wf=mic_sig,
                                 frequency_sample_rate_hz=sig_sample_rate_hz,
                                 band_order_Nth=order_number_input,
                                 dictionary_type="tone")

    mic_cwt_snr, mic_cwt_snr_bits, mic_cwt_snr_entropy = entropy.snr_mean_max(tfr_coeff_complex=mic_cwt)
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

    plt.show()
