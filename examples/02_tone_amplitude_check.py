"""
libquantum example 2: 02_tone_amplitude_check.py
Illustrate TFRs on tones and compare amplitudes
"""
import numpy as np
import matplotlib.pyplot as plt
from libquantum import atoms, entropy, scales, spectra, utils, synthetics
import libquantum.plot_templates.plot_time_frequency_reps as pltq
import libwwz

if __name__ == "__main__":
    """
    # The primary goal of standardization is to permit multimodal sensor analysis for different sample rates
    # For a specified signal duration, there is only one key parameter: Order
    """

    print('Tone synthetic')
    order_number_input = 12
    EVENT_NAME = "Tone Test"
    station_id_str = 'Synthya'
    run_time_epoch_s = utils.datetime_now_epoch_s()

    mic_sig_sample_rate_hz = 800.
    sig_frequency_hz = 50.
    sig_duration_s = 5.

    # Construct synthetic tone with max unit amplitude
    mic_sig_epoch_s = np.arange(int(mic_sig_sample_rate_hz * sig_duration_s)) / mic_sig_sample_rate_hz + run_time_epoch_s
    mic_sig = np.sin(2*np.pi*sig_frequency_hz*mic_sig_epoch_s)
    mic_sig += synthetics.white_noise_fbits(sig=mic_sig, std_bit_loss=4.)

    mic_sig *= utils.taper_tukey(mic_sig_epoch_s, fraction_cosine=0.1)  # add taper
    synthetics.antialias_halfNyquist(mic_sig)  # Antialias filter synthetic

    # Frame to mic start and end and plot
    event_reference_time_epoch_s = mic_sig_epoch_s[0]

    max_time_s, min_frequency_hz = scales.from_duration(band_order_Nth=order_number_input,
                                                        sig_duration_s=sig_duration_s)
    print('\nRequest Order N=', order_number_input)
    print('Lowest frequency in hz that can support this order for this signal duration is ', min_frequency_hz)
    print('Scale with signal duration and to Nyquist, default G2 base re F1')

    # Select plot frequencies
    fmin = np.ceil(min_frequency_hz)
    fmax = 400

    # TFR SECTION
    # Compute complex wavelet transform (cwt) from signal duration
    mic_cwt, mic_cwt_bits, mic_cwt_time_s, mic_cwt_frequency_hz = \
        atoms.cwt_chirp_from_sig(sig_wf=mic_sig,
                                 frequency_sample_rate_hz=mic_sig_sample_rate_hz,
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
                                start_time_epoch=event_reference_time_epoch_s,
                                figure_title="CWT for " + EVENT_NAME,
                                frequency_hz_ymin=fmin,
                                frequency_hz_ymax=fmax)

    # Compute constant Q transform (CQT) from segmented signal duration
    mic_cqt, mic_cqt_bits, mic_cqt_time_s, mic_cqt_frequency_hz = \
        spectra.cqt_from_sig(sig_wf=mic_sig,
                             frequency_sample_rate_hz=mic_sig_sample_rate_hz,
                             band_order_Nth=order_number_input,
                             dictionary_type="tone")

    mic_cqt_snr, mic_cqt_snr_bits, mic_cqt_snr_entropy = entropy.snr_mean_max(tfr_coeff_complex=mic_cqt)
    pltq.plot_wf_mesh_mesh_vert(redvox_id=station_id_str,
                                wf_panel_2_sig=mic_sig,
                                wf_panel_2_time=mic_sig_epoch_s,
                                mesh_time=mic_cqt_time_s,
                                mesh_frequency=mic_cqt_frequency_hz,
                                mesh_panel_1_trf=mic_cqt_bits,
                                mesh_panel_1_colormap_scaling="range",
                                mesh_panel_0_tfr=mic_cqt_snr_entropy,
                                wf_panel_2_units="Norm",
                                mesh_panel_1_cbar_units="bits",
                                mesh_panel_0_cbar_units="eSNR bits",
                                figure_title="CQT Hann for " + EVENT_NAME,
                                frequency_hz_ymin=fmin,
                                frequency_hz_ymax=fmax)

    # Compute constant Q transform (CQT) from segmented signal duration using Gaussian window
    mic_cqtg, mic_cqtg_bits, mic_cqt_time_s, mic_cqt_frequency_hz = \
        spectra.cqt_from_sig(sig_wf=mic_sig,
                             frequency_sample_rate_hz=mic_sig_sample_rate_hz,
                             band_order_Nth=order_number_input,
                             cqt_window="cqt_gauss",
                             dictionary_type="tone")

    mic_cqtg_snr, mic_cqtg_snr_bits, mic_cqtg_snr_entropy = entropy.snr_mean_max(tfr_coeff_complex=mic_cqt)
    pltq.plot_wf_mesh_mesh_vert(redvox_id=station_id_str,
                                wf_panel_2_sig=mic_sig,
                                wf_panel_2_time=mic_sig_epoch_s,
                                mesh_time=mic_cqt_time_s,
                                mesh_frequency=mic_cqt_frequency_hz,
                                mesh_panel_1_trf=mic_cqtg_bits,
                                mesh_panel_1_colormap_scaling="range",
                                mesh_panel_0_tfr=mic_cqtg_snr_entropy,
                                wf_panel_2_units="Norm",
                                mesh_panel_1_cbar_units="bits",
                                mesh_panel_0_cbar_units="eSNR bits",
                                figure_title="CQT Gauss for " + EVENT_NAME,
                                frequency_hz_ymin=fmin,
                                frequency_hz_ymax=fmax)

    # Compute short term Fourier transform (STFT) from segmented signal duration
    mic_stft, mic_stft_bits, mic_stft_time_s, mic_stft_frequency_hz = \
        spectra.stft_from_sig(sig_wf=mic_sig,
                              frequency_sample_rate_hz=mic_sig_sample_rate_hz,
                              band_order_Nth=order_number_input)

    mic_stft_snr, mic_stft_snr_bits, mic_stft_snr_entropy = entropy.snr_mean_max(tfr_coeff_complex=mic_stft)

    # Log frequency is the default
    pltq.plot_wf_mesh_mesh_vert(frequency_scaling="log",
                                redvox_id=station_id_str,
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
                                figure_title="STFT for " + EVENT_NAME,
                                frequency_hz_ymin=fmin,
                                frequency_hz_ymax=fmax)

    # Linear frequency scale must be specified
    pltq.plot_wf_mesh_mesh_vert(frequency_scaling="linear",
                                redvox_id=station_id_str,
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
                                figure_title="STFT for " + EVENT_NAME,
                                frequency_hz_ymin=fmin,
                                frequency_hz_ymax=fmax)

    # Compute the WWZ
    freq_target = scales.Slice.F1
    freq_low = mic_cqt_frequency_hz.min()
    freq_high = mic_cqt_frequency_hz.max()
    band_order = order_number_input
    log_scale_base = scales.Slice.G2
    override = True
    freq_params = [freq_target, freq_low, freq_high, band_order, log_scale_base, override]
    print(freq_params)

    w_target = 2 * np.pi * freq_target
    decay_constant = 1 / (2 * w_target ** 2)

    wwz = libwwz.wwt(magnitudes=mic_sig,
                     timestamps=mic_sig_epoch_s-mic_sig_epoch_s[0],
                     time_divisions=len(mic_cqt_time_s),
                     freq_params=freq_params,
                     decay_constant=decay_constant,
                     method='octave')

    mic_wwz = wwz[3].T
    mic_wwz_bits = utils.log2epsilon(mic_wwz)
    mic_wwz_snr, mic_wwz_snr_bits, mic_wwz_snr_entropy = entropy.snr_mean_max(tfr_coeff_complex=mic_wwz)
    pltq.plot_wf_mesh_mesh_vert(redvox_id=station_id_str,
                                wf_panel_2_sig=mic_sig,
                                wf_panel_2_time=mic_sig_epoch_s,
                                mesh_time=mic_cqt_time_s,
                                mesh_frequency=mic_cqt_frequency_hz,
                                mesh_panel_1_trf=mic_wwz_bits,
                                mesh_panel_1_colormap_scaling="range",
                                mesh_panel_0_tfr=mic_wwz_snr_entropy,
                                wf_panel_2_units="Norm",
                                mesh_panel_1_cbar_units="bits",
                                mesh_panel_0_cbar_units="eSNR bits",
                                figure_title="WWZ for " + EVENT_NAME,
                                frequency_hz_ymin=fmin,
                                frequency_hz_ymax=fmax)
    plt.show()
