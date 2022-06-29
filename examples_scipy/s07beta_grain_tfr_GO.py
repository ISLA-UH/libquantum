"""
libquantum example: s07_grain_tfr

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import atoms_styx, stft_styx
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq
from libquantum.styx import tfr_stx_fft
print(__doc__)


if __name__ == "__main__":
    """
    Compute the spectrogram over sliding windows.
    The Welch method is equivalent to averaging the spectrogram over the columns.
    """

    EVENT_NAME = 'grain test'
    station_id_str = 'synth'
    # alpha: Shape parameter of the Welch and STFT Tukey window, representing the fraction of the window inside the cosine tapered region.
    # If zero, the Tukey window is equivalent to a rectangular window.
    # If one, the Tukey window is equivalent to a Hann window.
    alpha = 0.25

    # Specifying the grain parameters requires some forethought
    frequency_center_hz = 100
    frequency_sample_rate_hz = 800
    order_number_input = 12

    # TODO: ADD Averaging frequency for fft_nd
    time_nd = 2**11
    time_fft_nd = 2**7

    # The CWX and STX will be evaluated from the number of points in FFT of the signal
    frequency_cwt_pos_hz = np.fft.rfftfreq(time_nd, d=1/frequency_sample_rate_hz)
    # Want to evaluate the CWX and STX at the NFFT frequencies of the sliding-window Welch/STFT spectra
    frequency_stft_pos_hz = np.fft.rfftfreq(time_fft_nd, d=1 / frequency_sample_rate_hz)

    # CWT
    cwt_fft_index = np.argmin(np.abs(frequency_cwt_pos_hz - frequency_center_hz))
    frequency_center_cwt_hz = frequency_cwt_pos_hz[cwt_fft_index]
    frequency_resolution_cwt_hz = frequency_sample_rate_hz / time_nd
    # STFT
    stft_fft_index = np.argmin(np.abs(frequency_stft_pos_hz - frequency_center_hz))
    frequency_center_stft_hz = frequency_stft_pos_hz[stft_fft_index]
    frequency_resolution_stft_hz = frequency_sample_rate_hz / time_fft_nd

    # Compare:
    print('These two should coincide for a fair comparison')
    print('Center CWT FFT frequency, Hz:', frequency_center_cwt_hz)
    print('Center STFT FFT frequency, Hz:', frequency_center_stft_hz)

    # exit()
    # TODO: Note oversampling on CWT leads to overestimation of energy!!
    frequency_cwt_fft_hz = frequency_stft_pos_hz[1:]
    # frequency_cwt_fft_hz = frequency_cwt_pos_hz[1:]

    mic_sig_complex, time_s, scale, omega, amp = \
        atoms_styx.wavelet_centered_4cwt(band_order_Nth=order_number_input,
                                         duration_points=time_nd,
                                         scale_frequency_center_hz=frequency_center_stft_hz,
                                         frequency_sample_rate_hz=frequency_sample_rate_hz,
                                         dictionary_type="norm")
    mic_sig_real = np.real(mic_sig_complex)
    mic_sig_imag = np.imag(mic_sig_complex)

    # Computed Variance; divides by the number of points
    mic_sig_real_var = np.var(mic_sig_real)
    mic_sig_imag_var = np.var(mic_sig_imag)

    # Theoretical variance TODO: construct function
    mic_sig_real_var_nominal = amp**2/len(time_s) * 0.5*np.sqrt(np.pi)*scale * \
                               (1 + np.exp(-(scale*omega)**2))
    mic_sig_imag_var_nominal = amp**2/len(time_s) * 0.5*np.sqrt(np.pi)*scale * \
                               (1 - np.exp(-(scale*omega)**2))

    # Mathematical integral ~ computed Variance * Number of Samples. The dictionary type = "norm" returns 1/2.
    mic_sig_real_integral = np.var(mic_sig_real)*len(mic_sig_real)
    mic_sig_imag_integral = np.var(mic_sig_imag)*len(mic_sig_real)

    print('\nAtom Variance')
    print('mic_sig_real_variance:', mic_sig_real_var)
    print('real_variance_nominal:', mic_sig_real_var_nominal)
    print('mic_sig_imag_variance:', mic_sig_imag_var)
    print('imag_variance_nominal:', mic_sig_imag_var_nominal)

    # Choose the real component as the test signal
    mic_sig = np.copy(mic_sig_real)
    mic_sig_var = mic_sig_real_var
    mic_sig_var_nominal = mic_sig_real_var_nominal
    print('\nChoose real part as signal:')
    print('var/nominal var:', mic_sig_var/mic_sig_var_nominal)

    # Compute the Welch PSD; averaged spectrum over sliding windows
    frequency_welch_hz, psd_welch_power = \
        stft_styx.welch_power_pow2(sig_wf=mic_sig,
                                   frequency_sample_rate_hz=frequency_sample_rate_hz,
                                   nfft_points=time_fft_nd)

    # Information overload methods
    welch_power, welch_power_per_band, welch_power_per_sample, welch_power_total, welch_power_scaled, \
    welch_information_bits, welch_information_bits_per_band, welch_information_bits_per_sample, \
    welch_information_bits_total, welch_information_scaled = stft_styx.power_and_information_shannon_welch(psd_welch_power)
    
    # Compute the spectrogram with the stft option
    frequency_stft_hz, time_stft_s, stft_complex = \
        stft_styx.stft_complex_pow2(sig_wf=mic_sig,
                                    frequency_sample_rate_hz=frequency_sample_rate_hz,
                                    nfft_points=time_fft_nd)

    # Information overload methods
    stft_power, stft_power_per_band, stft_power_per_sample, stft_power_total, stft_power_scaled, \
    stft_information_bits, stft_information_bits_per_band, stft_information_bits_per_sample, \
    stft_information_bits_total, stft_information_scaled = stft_styx.power_and_information_shannon_stft(stft_complex)

    # Compute complex wavelet transform (cwt) from signal duration using the Gabor atoms
    frequency_cwt_hz, time_cwt_s, cwt_complex = \
        atoms_styx.cwt_complex_any_scale(sig_wf=mic_sig,
                                         frequency_sample_rate_hz=frequency_sample_rate_hz,
                                         frequency_cwt_hz=frequency_cwt_fft_hz,
                                         band_order_Nth=order_number_input,
                                         dictionary_type="spect")

    # Information overload methods
    cwt_power, cwt_power_per_band, cwt_power_per_sample, cwt_power_total, cwt_power_scaled, \
    cwt_information_bits, cwt_information_bits_per_band, cwt_information_bits_per_sample, \
    cwt_information_bits_total, cwt_information_scaled = atoms_styx.power_and_information_shannon_cwt(cwt_complex)

    # Compute Stockwell transform TODO: Export time
    [stx_complex, _, frequency_stx_hz, frequency_stx_fft_hz, W] = \
        tfr_stx_fft(sig_wf=mic_sig,
                    time_sample_interval=1/frequency_sample_rate_hz,
                    frequency_min=frequency_resolution_stft_hz,
                    frequency_max=frequency_sample_rate_hz/2,
                    scale_order_input=order_number_input,
                    scale_ref_input=1 / frequency_center_stft_hz,
                    is_geometric=True,
                    is_inferno=False)

    stx_power = 2 * np.abs(stx_complex) ** 2

    # Scale power by variance
    welch_over_var = psd_welch_power / mic_sig_var
    stft_over_var = np.average(stft_power, axis=1) / mic_sig_var
    cwt_over_var = np.average(cwt_power, axis=1) / mic_sig_var
    stx_over_var = np.average(stx_power, axis=1) / mic_sig_var

    print('\nSum scaled spectral power')
    print('Sum Welch:', np.sum(welch_over_var))
    print('Sum STFT:', np.sum(stft_over_var))
    print('Sum CWT:', np.sum(cwt_over_var))
    print('Sum STX:', np.sum(stx_over_var))

    print('Sum cwt_power_scaled :', np.sum(cwt_power_scaled))
    print('Sum cwt_information_scaled :', np.sum(cwt_information_scaled))

    # exit()
    # # Express in bits; revisit
    # # TODO: What units shall we use? Evaluate CWT and Stockwell first
    # mic_stft_bits = utils.log2epsilon(np.sqrt(stft_power))
    # mic_cwt_bits = utils.log2epsilon(np.sqrt(cwt_power))
    # mic_stx_bits = utils.log2epsilon(np.sqrt(stx_power))

    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title('Synthetic CW, with taper')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax2.semilogx(frequency_welch_hz, welch_over_var, label='Welch')
    ax2.semilogx(frequency_stft_hz, stft_over_var, '.-', label="STFT")
    ax2.semilogx(frequency_cwt_hz, cwt_over_var, '-.', label='CWT')
    ax2.semilogx(frequency_stx_hz, stx_over_var, '--', label='STX')
    ax2.set_title('Welch and Spect FFT (RMS), f = ' + str(round(frequency_center_stft_hz * 100) / 100) + ' Hz')
    ax2.set_xlabel('Frequency, hz')
    ax2.set_ylabel('Power / var(sig)')
    ax2.grid(True)
    ax2.legend()

    plt.show()
    exit()

    # # Select plot frequencies
    # fmin = 2 * frequency_resolution_stft_hz
    # fmax = frequency_sample_rate_hz/2  # Nyquist
    #
    # pltq.plot_wf_mesh_vert(redvox_id='00',
    #                        wf_panel_a_sig=mic_sig,
    #                        wf_panel_a_time=time_s,
    #                        mesh_time=time_stft_s,
    #                        mesh_frequency=frequency_stft_hz,
    #                        mesh_panel_b_tfr=stft_information_scaled,
    #                        mesh_panel_b_colormap_scaling="range",
    #                        wf_panel_a_units="Norm",
    #                        mesh_panel_b_cbar_units="bits",
    #                        start_time_epoch=0,
    #                        figure_title="stft for " + EVENT_NAME,
    #                        frequency_hz_ymin=fmin,
    #                        frequency_hz_ymax=fmax)
    #
    # pltq.plot_wf_mesh_vert(redvox_id='00',
    #                        wf_panel_a_sig=mic_sig,
    #                        wf_panel_a_time=time_cwt_s,
    #                        mesh_time=time_cwt_s,
    #                        mesh_frequency=frequency_cwt_hz,
    #                        mesh_panel_b_tfr=cwt_information_bits,
    #                        mesh_panel_b_colormap_scaling="range",
    #                        wf_panel_a_units="Norm",
    #                        mesh_panel_b_cbar_units="bits",
    #                        start_time_epoch=0,
    #                        figure_title="cwt for " + EVENT_NAME,
    #                        frequency_hz_ymin=fmin,
    #                        frequency_hz_ymax=fmax)

    # pltq.plot_wf_mesh_vert(redvox_id='00',
    #                        wf_panel_a_sig=mic_sig,
    #                        wf_panel_a_time=time_cwt_s,
    #                        mesh_time=time_cwt_s,
    #                        mesh_frequency=frequency_stx_hz,
    #                        mesh_panel_b_tfr=mic_stx_bits,
    #                        mesh_panel_b_colormap_scaling="range",
    #                        wf_panel_a_units="Norm",
    #                        mesh_panel_b_cbar_units="bits",
    #                        start_time_epoch=0,
    #                        figure_title="STX for " + EVENT_NAME,
    #                        frequency_hz_ymin=fmin,
    #                        frequency_hz_ymax=fmax)


    plt.show()

