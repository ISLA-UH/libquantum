"""
libquantum example 3: 03_sweep_linear.py
Construct classic linear chirp and illustrate CWT and STFT TRFs.
"""

import os
from pathlib import Path
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
from libquantum import atoms, entropy, scales, spectra, utils, synthetics
import libquantum.plot_templates.plot_time_frequency_reps as pltq


if __name__ == "__main__":
    """
    Exercises with classic linear sweep
    Option of exporting to wav
    """

    # Do you want to export a wav file? True or False
    do_save_wave = False
    # If True, saves to home directory
    home_dir: str = str(Path.home())
    # Or can specify a preferred wav file directory
    # home_dir: str = "/Users/mgarces/Documents/DATA_API_M/synthetics"
    output_wav_directory = os.path.join(home_dir, "wav")

    EVENT_NAME = "redshift_linear_sweep"
    print("Event Name: " + EVENT_NAME)
    wav_filename = EVENT_NAME
    order_number_input = 3

    station_id_str = 'Synth'
    run_time_epoch_s = utils.datetime_now_epoch_s()

    # Chirp type
    is_redshift = True

    sig_wf_sample_rate_hz = 8000.
    sig_frequency_hz_start = 40.
    sig_frequency_hz_end = 400.
    sig_duration_s = 13.19675
    head_s = 0.5

    # sig_wf_sample_rate_hz = 8000.
    # sig_frequency_hz_start = 40.
    # sig_frequency_hz_end = 400.
    # sig_duration_s = 13.19675
    # head_s = 0.5


    # Blueshift sweep
    sig_wf_blu, sig_wf_epoch_s = synthetics.chirp_linear_in_noise(snr_bits=12.,
                                                                  sample_rate_hz=sig_wf_sample_rate_hz,
                                                                  duration_s=sig_duration_s,
                                                                  frequency_start_hz=sig_frequency_hz_start,
                                                                  frequency_end_hz=sig_frequency_hz_end,
                                                                  intro_s=head_s,
                                                                  outro_s=head_s)
    sig_wf_red = np.flipud(sig_wf_blu)

    # Choose origin and red/blue shift
    sig_wf_epoch_s += run_time_epoch_s
    sig_wf = np.copy(sig_wf_red)

    # Antialias filter synthetic
    synthetics.antialias_halfNyquist(synth=sig_wf)

    # Export to wav directory
    if do_save_wave:
        wav_sample_rate_hz = 8000.
        export_filename = os.path.join(output_wav_directory, wav_filename + "_8kz.wav")
        synth_wav = 0.9 * np.real(sig_wf) / np.max(np.abs((np.real(sig_wf))))
        scipy.io.wavfile.write(export_filename, int(wav_sample_rate_hz), synth_wav)

    # Frame to mic start and end and plot
    event_reference_time_epoch_s = sig_wf_epoch_s[0]

    max_time_s, min_frequency_hz = scales.from_duration(band_order_Nth=order_number_input,
                                                        sig_duration_s=sig_duration_s)
    print('\nRequest Order N=', order_number_input)
    print('Lowest frequency in hz that can support this order for this signal duration is ', min_frequency_hz)
    print('Scale with signal duration and to Nyquist, default G2 base re F1')

    # Select plot frequencies
    fmin = np.ceil(min_frequency_hz)
    fmax = sig_wf_sample_rate_hz/2.

    # TFR SECTION
    # Compute complex wavelet transform (cwt) from signal duration
    if is_redshift:
        mic_cwt, mic_cwt_bits, mic_cwt_time_s, mic_cwt_frequency_hz = \
            atoms.cwt_chirp_from_sig(sig_wf=sig_wf,
                                     frequency_sample_rate_hz=sig_wf_sample_rate_hz,
                                     band_order_Nth=order_number_input,
                                     dictionary_type="tone",
                                     index_shift=-1)

    else:
        mic_cwt, mic_cwt_bits, mic_cwt_time_s, mic_cwt_frequency_hz = \
            atoms.cwt_chirp_from_sig(sig_wf=sig_wf,
                                     frequency_sample_rate_hz=sig_wf_sample_rate_hz,
                                     band_order_Nth=order_number_input,
                                     dictionary_type="tone")

    mic_cwt_snr, mic_cwt_snr_bits, mic_cwt_snr_entropy = entropy.snr_mean_max(tfr_coeff_complex=mic_cwt)

    pltq.plot_wf_mesh_mesh_vert(redvox_id=station_id_str,
                                wf_panel_2_sig=sig_wf,
                                wf_panel_2_time=sig_wf_epoch_s,
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

    # Compute short term Fourier transform (STFT) from segmented signal duration
    mic_stft, mic_stft_bits, mic_stft_time_s, mic_stft_frequency_hz = \
        spectra.stft_from_sig(sig_wf=sig_wf,
                              frequency_sample_rate_hz=sig_wf_sample_rate_hz,
                              band_order_Nth=order_number_input)

    mic_stft_snr, mic_stft_snr_bits, mic_stft_snr_entropy = entropy.snr_mean_max(tfr_coeff_complex=mic_stft)

    # Log frequency is the default, for linear use frequency_scaling="linear",
    pltq.plot_wf_mesh_mesh_vert(frequency_scaling="log",
                                redvox_id=station_id_str,
                                wf_panel_2_sig=sig_wf,
                                wf_panel_2_time=sig_wf_epoch_s,
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
    plt.show()

