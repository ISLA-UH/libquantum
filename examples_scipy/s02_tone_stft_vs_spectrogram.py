"""
libquantum example: s02_tone_stft_vs_spectrogram.py
Compute and display spectrogram on simple tone with a taper window.
There is an independent Tukey taper (w/ alpha) on each Welch and Spectrogram subwindow.
Contract over the columns and compare to Welch power spectral density (PSD) to verify amplitudes.
Case study:
Sinusoid input with unit amplitude
Validate:
Welch power averaged over the signal duration is 1/2
RMS amplitude = 1/sqrt(2)

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import utils, benchmark_signals
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq
print(__doc__)

if __name__ == "__main__":
    """
    Compute the spectrogram over sliding windows. Added taper, noise, and AA to signal.
    Added STFT Tukey window alpha > 0.
    The Welch method is equivalent to averaging the spectrogram over the columns.
    """

    EVENT_NAME = 'tone test'
    # Construct a tone of fixed frequency with a constant sample rate
    # In this example, added noise, taper, and anti-aliasing filter.
    # In the first example (FFT), the nominal signal duration was 1s.
    # In this example the nominal signal duration is 16s, with averaging (fft) window duration of 1s.
    frequency_tone_hz = 60
    [mic_sig, time_s, time_fft_nd,
     frequency_sample_rate_hz, frequency_center_fft_hz, frequency_resolution_fft_hz] = \
        benchmark_signals.well_tempered_tone(frequency_center_hz=frequency_tone_hz,
                                             frequency_sample_rate_hz=800,
                                             time_duration_s=16,
                                             time_fft_s=1,
                                             use_fft_frequency=True,
                                             add_noise_taper_aa=True)



    # # Compare to synthetic tone with 2^n points and max FFT amplitude NOT at exact fft frequency
    # # It does not return unit amplitude (but it's close)
    # [mic_sig, time_s, time_fft_nd,
    #  frequency_sample_rate_hz, frequency_center_stft_hz, frequency_resolution_stft_hz] = \
    #     benchmark_signals.well_tempered_tone(frequency_center_hz=frequency_tone_hz,
    #                                          frequency_sample_rate_hz=800,
    #                                          time_duration_s=16,
    #                                          time_fft_s=1,
    #                                          use_fft_frequency=False,
    #                                          add_noise_taper_aa=True)

    # alpha: Shape parameter of the Welch and STFT Tukey window, representing the fraction of the window inside the cosine tapered region.
    # If zero, the Tukey window is equivalent to a rectangular window.
    # If one, the Tukey window is equivalent to a Hann window.
    alpha = 0.25  # 25% Tukey (Cosine) window

    # Computed and nominal values
    mic_sig_rms = np.std(mic_sig)
    mic_sig_rms_nominal = 1/np.sqrt(2)

    # Computed Variance; divides by the number of points
    mic_sig_var = np.var(mic_sig)
    mic_sig_var_nominal = 1/2.

    # Compute the Welch PSD; averaged spectrum over sliding windows
    frequency_welch_hz, psd_welch_power = signal.welch(x=mic_sig,
                                                       fs=frequency_sample_rate_hz,
                                                       window=('tukey', alpha),
                                                       nperseg=time_fft_nd,
                                                       noverlap=time_fft_nd // 2,
                                                       nfft=time_fft_nd,
                                                       detrend='constant',
                                                       return_onesided=True,
                                                       axis=-1,
                                                       scaling='spectrum',
                                                       average='mean')

    # Compute the spectrogram with the spectrum option
    frequency_spect_hz, time_spect_s, psd_spec_power = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=time_fft_nd,
                           noverlap=time_fft_nd // 2,
                           nfft=time_fft_nd,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='spectrum',
                           mode='psd')

    # Compute the spectrogram with the stft option
    frequency_stft_hz, time_stft_s, stft_complex = \
        signal.stft(x=mic_sig,
                    fs=frequency_sample_rate_hz,
                    window=('tukey', alpha),
                    nperseg=time_fft_nd,
                    noverlap=time_fft_nd // 2,
                    nfft=time_fft_nd,
                    detrend='constant',
                    return_onesided=True,
                    axis=-1,
                    boundary='zeros',
                    padded=True)

    stft_power = 2 * np.abs(stft_complex) ** 2

    fft_rms_welch = np.sqrt(np.abs(psd_welch_power)) / mic_sig_rms
    fft_rms_spect = np.sqrt(np.average(psd_spec_power, axis=1)) / mic_sig_rms
    fft_rms_stft = np.sqrt(np.average(stft_power, axis=1)) / mic_sig_rms

    welch_over_var = psd_welch_power / mic_sig_var
    spect_over_var = np.average(psd_spec_power, axis=1) / mic_sig_var
    stft_over_var = np.average(stft_power, axis=1) / mic_sig_var

    # Express in bits; revisit
    # TODO: What units shall we use? Evaluate Stockwell first
    mic_spect_bits = utils.log2epsilon(np.sqrt(psd_spec_power))
    mic_stft_bits = utils.log2epsilon(np.sqrt(stft_power))
    print('Max spect bits:', np.max(mic_spect_bits))
    print('Max stft bits:', np.max(mic_stft_bits))

    # Compute the inverse stft (istft)
    sig_time_istft, sig_wf_istft = signal.istft(Zxx=stft_complex,
                                                fs=frequency_sample_rate_hz,
                                                window=('tukey', alpha),
                                                nperseg=time_fft_nd,
                                                noverlap=time_fft_nd // 2,
                                                nfft=time_fft_nd,
                                                input_onesided=True,
                                                boundary=True,
                                                time_axis=-1,
                                                freq_axis=-2)

    print('\n*** SUMMARY: STFT Time-Frequency Representation (TFR) estimates for a constant-frequency tone  ***')
    print('The signal.stft and signal.spectrogram with scaling=spectrum and mode=psd are comparable.')
    print('The spectrogram returns power, whereas the stft returns invertible, complex Fourier coefficients.')
    print('The Welch spectrum is reproduced by averaging the stft over the time dimension.')
    print('** NOTE: EXACT RECONSTRUCTION NOT EXPECTED WITH TAPER AND OTHER DEVIATIONS FROM IDEAL. PLAY WITH alpha.'
          'ACCEPT AND QUANTIFY COMPROMISE **')

    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title('Synthetic CW, with taper')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax2.semilogx(frequency_welch_hz, welch_over_var, label='Welch')
    ax2.semilogx(frequency_spect_hz, spect_over_var, '-.', label='Spect')
    ax2.semilogx(frequency_stft_hz, stft_over_var, '.-', label='STFT')
    ax2.set_title('Welch, Spect, and STFT Power, f = ' + str(round(frequency_center_fft_hz*100)/100) + ' Hz')
    ax2.set_xlabel('Frequency, hz')
    ax2.set_ylabel('|power|/VAR(signal)')
    ax2.grid(True)
    ax2.legend()

    # Plot the inverse stft (full recovery)
    fig2, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(sig_time_istft, sig_wf_istft)
    ax1.set_title('Inverse CW from ISTFT')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax2.plot(sig_time_istft, (mic_sig - sig_wf_istft)**2)
    ax2.set_title('(original - inverse)**2')
    ax2.set_xlabel('Time, s')
    ax2.set_ylabel('Norm')

    # Select plot frequencies
    fmin = 2*frequency_resolution_fft_hz
    fmax = frequency_sample_rate_hz/2  # Nyquist
    pltq.plot_wf_mesh_vert(redvox_id='00',
                           wf_panel_a_sig=mic_sig,
                           wf_panel_a_time=time_s,
                           mesh_time=time_spect_s,
                           mesh_frequency=frequency_spect_hz,
                           mesh_panel_b_tfr=mic_spect_bits,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="spectrogram for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    pltq.plot_wf_mesh_vert(redvox_id='00',
                           wf_panel_a_sig=mic_sig,
                           wf_panel_a_time=time_s,
                           mesh_time=time_stft_s,
                           mesh_frequency=frequency_stft_hz,
                           mesh_panel_b_tfr=mic_stft_bits,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="stft for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    plt.show()
