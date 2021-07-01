import os
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
from libquantum import atoms, entropy, scales, spectra, utils, synthetics
import libquantum.plot_templates.plot_time_frequency_reps as pltq
import libquantum.plot_templates.plot_time_frequency_picks as pltpk

if __name__ == "__main__":
    """
    # The primary goal of standardization is to permit multimodal sensor analysis for different sample rates
    # For a specified signal duration, there is only one key parameter: Order
    """

    # "Ideal" wavelet chirp type
    order_number_input = 12
    scale_base = scales.Slice.G3  # recurrent on decades
    scale_edge = scale_base ** (1.0 / (2.0 * order_number_input))

    overlap_fraction = 0.
    # overlap_fraction = 0.25
    glide_direction = -1
    # glide_direction = -1.5

    #grain_type_str = 'tukey'
    grain_type_str = 'gauss'

    EVENT_NAME = "q-glide_aud800_N" + str(order_number_input)
    print("Event Name: " + EVENT_NAME)
    wav_filename = EVENT_NAME

    do_save_wave = False
    do_reassignment = False
    input_directory = "/Users/mgarces/Documents/DATA_API_M/synthetics"
    output_wav_directory = os.path.join(input_directory, "wav")
    station_id_str = 'synth'
    run_time_epoch_s = utils.datetime_now_epoch_s()

    # sig_wf_sample_rate_hz = 8000.
    # sig_frequency_low_hz = 100.
    # sig_frequency_high_hz = 410.
    # head_points = 4000

    # These are the design (input/request) parameters
    sig_wf_sample_rate_hz = 8000.
    sig_frequency_low_hz = 40.
    sig_frequency_high_hz = 400.
    head_points = 4000

    # These are the standardized frequencies
    frequency_center_hz, frequency_start_hz, frequency_end_hz = \
        synthetics.gabor_grain_frequencies(frequency_order_input=order_number_input,
                                           frequency_low_input=sig_frequency_low_hz,
                                           frequency_high_input=sig_frequency_high_hz,
                                           frequency_sample_rate_input=sig_wf_sample_rate_hz,
                                           frequency_base_input = scale_base)

    print('\nSweep specifications')
    print('Highest Nth edge frequency, Hz:', np.max(frequency_end_hz))
    print('Highest Nth center frequency, Hz:', np.max(frequency_center_hz))
    print('Lowest Nth center frequency, Hz:', np.min(frequency_center_hz))
    print('Lowest Nth edge frequency, Hz:', np.min(frequency_start_hz))

    sig_wf_red = np.zeros(0)
    for sig_frequency_hz in frequency_center_hz:
        # Redshift sweep
        if grain_type_str == 'tukey':
            sig_step = \
                synthetics.tukey_tight_grain(band_order_Nth=order_number_input,
                                             scale_frequency_center_hz=sig_frequency_hz,
                                             frequency_sample_rate_hz=sig_wf_sample_rate_hz,
                                             fraction_cosine=.67,
                                             index_shift=glide_direction,
                                             frequency_base_input=scale_base)
        else:  # Gauss envelope
            sig_step = \
                synthetics.gabor_tight_grain(band_order_Nth=order_number_input,
                                             scale_frequency_center_hz=sig_frequency_hz,
                                             frequency_sample_rate_hz=sig_wf_sample_rate_hz,
                                             index_shift=glide_direction,
                                             frequency_base_input=scale_base)


        # Logarithmic shortening, not obvious
        # Explicit; sequence order is key, leave as is for debugging
        sig_step_len = len(sig_step)
        sig_red_len = len(sig_wf_red)
        overlap_points = int(overlap_fraction * sig_step_len)

        # TODO: there must be a tuning between the overlap_fraction and fraction_cosine. Find it.
        if sig_red_len > 0:
            sig_step = np.concatenate([np.zeros(sig_red_len-overlap_points), sig_step])
            sig_wf_red = np.concatenate([sig_wf_red, np.zeros(sig_step_len-overlap_points)])
            sig_wf_red = sig_wf_red + sig_step
        else:
            sig_wf_red = sig_step

    # Add head and tail
    sig_wf_red = np.concatenate([np.zeros(head_points), sig_wf_red, np.zeros(head_points)])
    #
    # # Add noise
    # noise_wf = synthetics.white_noise_fbits(sig=sig_wf_red, std_bit_loss=8)
    # sig_wf_red = sig_wf_red + noise_wf

    sig_time_s = np.arange(len(sig_wf_red))/sig_wf_sample_rate_hz
    sig_duration_s = np.max(sig_time_s)

    # Blueshift
    sig_wf_blue = np.flipud(sig_wf_red)

    # Choose origin and red/blue shift
    sig_wf_epoch_s = sig_time_s + run_time_epoch_s
    sig_wf = np.copy(np.imag(sig_wf_red))

    # plt.figure()
    # plt.plot(sig_time_s, sig_wf)
    # plt.grid(True)
    # plt.title('Redshift')
    # plt.show()

    # Antialias filter synthetic
    synthetics.antialias_halfNyquist(sig_wf)

    # Export to wav directory
    if do_save_wave:
        wav_sample_rate_hz = 8000.
        export_filename = os.path.join(output_wav_directory, wav_filename + "_8kz.wav")
        synth_wav = 0.9 * np.real(sig_wf) / np.max(np.abs((np.real(sig_wf))))
        scipy.io.wavfile.write(export_filename, int(wav_sample_rate_hz), synth_wav)

    # Frame to mic start and end and plot
    event_reference_time_epoch_s = sig_wf_epoch_s[0]
    # print('\nExtraction start time for mic: ', event_reference_time_epoch_s)

    # The min_frequency_hz is needed for STFT
    max_time_s, min_frequency_hz = scales.from_duration(order_number_input, sig_duration_s)
    print('\nRequest Order N=', order_number_input)
    print('Sweep duration, s:', sig_duration_s)
    print('Lowest frequency in hz that can support this order for this signal duration is ', min_frequency_hz)
    print('Scale with signal duration and to Nyquist, default G2 base re F1')


    # TFR SECTION
    # This could be placed in a loop
    # Compute complex wavelet transform (cwt) by specifying the start and end center frequencies
    # and getting the n-1 band below it.

    cwt_frequency_high_hz = np.max(frequency_end_hz)*scale_edge**2
    cwt_frequency_low_hz = np.min(frequency_start_hz)/scale_edge
    print('Highest requested CWT frequency, Hz:', cwt_frequency_high_hz)
    print('Lowest  requested CWT frequency, Hz:', cwt_frequency_low_hz)

    print("\nRedshift CWT")
    mic_cwt, mic_cwt_bits, mic_cwt_time_s, mic_cwt_frequency_hz = \
        atoms.cwt_chirp_complex(sig_wf=sig_wf,
                                frequency_low_hz=cwt_frequency_low_hz,
                                frequency_high_hz=cwt_frequency_high_hz,
                                frequency_sample_rate_hz=sig_wf_sample_rate_hz,
                                scale_base=scale_base,
                                band_order_Nth=order_number_input,
                                cwt_type="conv",
                                dictionary_type="tone",
                                index_shift=glide_direction)

    mic_cwt_snr, mic_cwt_snr_bits, mic_cwt_snr_entropy = entropy.snr_mean_max(mic_cwt)

    print('Highest computed CWT center frequency, Hz:', np.max(mic_cwt_frequency_hz))
    print('Lowest  computed CWT center frequency, Hz:', np.min(mic_cwt_frequency_hz))

    print("Classic CWT")
    mic_cwt0, mic_cwt_bits0, mic_cwt_time_s0, mic_cwt_frequency_hz0 = \
        atoms.cwt_chirp_complex(sig_wf=sig_wf,
                                frequency_low_hz=cwt_frequency_low_hz,
                                frequency_high_hz=cwt_frequency_high_hz,
                                frequency_sample_rate_hz=sig_wf_sample_rate_hz,
                                scale_base=scale_base,
                                band_order_Nth=order_number_input,
                                cwt_type="conv",
                                dictionary_type="tone",
                                index_shift=0)

    mic_cwt_snr0, mic_cwt_snr_bits0, mic_cwt_snr_entropy0 = entropy.snr_mean_max(mic_cwt0)

    print("Blueshift CWT")
    mic_cwt1, mic_cwt_bits1, mic_cwt_time_s1, mic_cwt_frequency_hz1 = \
        atoms.cwt_chirp_complex(sig_wf=sig_wf,
                                frequency_low_hz=cwt_frequency_low_hz,
                                frequency_high_hz=cwt_frequency_high_hz,
                                frequency_sample_rate_hz=sig_wf_sample_rate_hz,
                                scale_base=scale_base,
                                band_order_Nth=order_number_input,
                                cwt_type="conv",
                                dictionary_type="tone",
                                index_shift=-glide_direction)

    mic_cwt_snr1, mic_cwt_snr_bits1, mic_cwt_snr_entropy1 = entropy.snr_mean_max(mic_cwt1)

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
                                figure_title="Redshift CWT for " + EVENT_NAME)

    pltq.plot_wf_mesh_mesh_vert(redvox_id=station_id_str,
                                wf_panel_2_sig=sig_wf,
                                wf_panel_2_time=sig_wf_epoch_s,
                                mesh_time=mic_cwt_time_s1,
                                mesh_frequency=mic_cwt_frequency_hz1,
                                mesh_panel_1_trf=mic_cwt_bits1,
                                mesh_panel_1_colormap_scaling="range",
                                mesh_panel_0_tfr=mic_cwt_snr_entropy1,
                                wf_panel_2_units="Norm",
                                mesh_panel_1_cbar_units="bits",
                                mesh_panel_0_cbar_units="eSNR bits",
                                start_time_epoch=event_reference_time_epoch_s,
                                figure_title="Blueshift CWT for " + EVENT_NAME)

    pltq.plot_wf_mesh_mesh_vert(redvox_id=station_id_str,
                                wf_panel_2_sig=sig_wf,
                                wf_panel_2_time=sig_wf_epoch_s,
                                mesh_time=mic_cwt_time_s0,
                                mesh_frequency=mic_cwt_frequency_hz0,
                                mesh_panel_1_trf=mic_cwt_bits0,
                                mesh_panel_1_colormap_scaling="range",
                                mesh_panel_0_tfr=mic_cwt_snr_entropy0,
                                wf_panel_2_units="Norm",
                                mesh_panel_1_cbar_units="bits",
                                mesh_panel_0_cbar_units="eSNR bits",
                                start_time_epoch=event_reference_time_epoch_s,
                                figure_title="Classic CWT for " + EVENT_NAME)

    # Plot difference
    pltq.plot_wf_mesh_mesh_vert(redvox_id=station_id_str,
                                wf_panel_2_sig=sig_wf,
                                wf_panel_2_time=sig_wf_epoch_s,
                                mesh_time=mic_cwt_time_s,
                                mesh_frequency=mic_cwt_frequency_hz,
                                mesh_panel_1_trf=mic_cwt_bits-mic_cwt_bits1,
                                mesh_panel_1_colormap_scaling="range",
                                mesh_panel_0_tfr=mic_cwt_bits-mic_cwt_bits0,
                                mesh_panel_0_colormap_scaling="range",
                                wf_panel_2_units="Norm",
                                mesh_panel_1_cbar_units="R-B bits",
                                mesh_panel_0_cbar_units="R-C bits",
                                start_time_epoch=event_reference_time_epoch_s,
                                figure_title="Diff CWT for " + EVENT_NAME)

    # Compute short term Fourier transform (STFT) from segmented signal duration
    mic_stft, mic_stft_bits, mic_stft_time_s, mic_stft_frequency_hz = \
        spectra.stft_from_sig(sig_wf=sig_wf,
                              frequency_sample_rate_hz=sig_wf_sample_rate_hz,
                              band_order_Nth=order_number_input)
    mic_stft_snr, mic_stft_snr_bits, mic_stft_snr_entropy = entropy.snr_mean_max(mic_stft)

    # Plot STFT
    pltq.plot_wf_mesh_mesh_vert(redvox_id=station_id_str,
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
                                frequency_hz_ymin=cwt_frequency_low_hz,
                                frequency_hz_ymax=cwt_frequency_high_hz)

    if do_reassignment:
        # Compute reassigned spectrogram
        mic_stft_rsg, mic_stft_rsg_bits, \
        mic_stft2_time_s, mic_stft2_frequency_hz, \
        mic_stft_rsg_time_s, mic_stft_rsg_frequency_hz = \
            spectra.stft_reassign_from_sig(sig_wf=sig_wf,
                                           frequency_sample_rate_hz=sig_wf_sample_rate_hz,
                                           band_order_Nth=order_number_input)
        mic_stft_rsg_snr, mic_stft_rsg_snr_bits, mic_stft_rsg_snr_entropy = entropy.snr_mean_max(mic_stft_rsg)

        # print('stft_rsg', mic_stft_rsg)
        # print('stft shape', mic_stft.shape)
        # print('stft_rsg shape', mic_stft_rsg.shape)
        # print('f shape', mic_stft2_frequency_hz.shape)
        # print('t shape', mic_stft2_time_s.shape)

        # print("t_rsg:", mic_stft_rsg_time_s)
        # print("f_rsg:", mic_stft_rsg_frequency_hz)
        # Plot reassigned STFT, same as STFT (good to verify)
        pltq.plot_wf_mesh_mesh_vert(redvox_id=station_id_str,
                                    wf_panel_2_sig=sig_wf,
                                    wf_panel_2_time=sig_wf_epoch_s,
                                    mesh_time=mic_stft2_time_s,
                                    mesh_frequency=mic_stft2_frequency_hz,
                                    mesh_panel_1_trf=mic_stft_rsg_bits,
                                    mesh_panel_1_colormap_scaling="range",
                                    mesh_panel_0_tfr=mic_stft_rsg_snr_entropy,
                                    wf_panel_2_units="Norm",
                                    mesh_panel_1_cbar_units="bits",
                                    mesh_panel_0_cbar_units="eSNR bits",
                                    figure_title="Reassigned STFT for " + EVENT_NAME,
                                    frequency_hz_ymin=cwt_frequency_low_hz,
                                    frequency_hz_ymax=cwt_frequency_high_hz)

        pltpk.plot_wf_mesh_scatter_vert(redvox_id=station_id_str,
                                        wf_panel_2_sig=sig_wf,
                                        wf_panel_2_time=sig_wf_epoch_s,
                                        mesh_time=mic_stft2_time_s,
                                        mesh_frequency=mic_stft2_frequency_hz,
                                        scatter_time=mic_stft_rsg_time_s,
                                        scatter_frequency=mic_stft_rsg_frequency_hz,
                                        mesh_panel_1_trf=mic_stft_rsg_bits,
                                        mesh_panel_1_colormap_scaling="range",
                                        mesh_panel_1_color_range=6,
                                        mesh_panel_0_tfr=mic_stft_rsg_bits,
                                        mesh_panel_0_colormap_scaling="range",
                                        mesh_panel_0_color_range=12,
                                        wf_panel_2_units="Norm",
                                        mesh_panel_1_cbar_units="bits",
                                        mesh_panel_0_cbar_units="bits",
                                        figure_title="Reassigned STFT for " + EVENT_NAME,
                                        frequency_hz_ymin=cwt_frequency_low_hz,
                                        frequency_hz_ymax=cwt_frequency_high_hz)

    plt.show()
