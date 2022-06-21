"""
libquantum example: s05_stft_tone_quantum
Compute stft spectrogram with libquantum
TODO: Turn into functions with hard-coded defaults
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import utils, synthetics, spectra
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq

print(__doc__)
EVENT_NAME = 'tone test'

# alpha: Shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered region.
# If zero, the Tukey window is equivalent to a rectangular window.
# If one, the Tukey window is equivalent to a Hann window.
alpha = 1

if __name__ == "__main__":
    """
    Compute the spectrogram over sliding windows.
    The Welch method is equivalent to averaging the spectrogram over the columns.
    """

    # Construct a tone of fixed frequency with a constant sample rate
    frequency_sample_rate_hz = 800.
    frequency_center_hz = 100.
    time_duration_s = 16
    # Split the record into segments. Previous example showed 1s duration was adequate
    time_fft_s = time_duration_s/16  # 1 second nominal
    # The segments determine the spectral resolution
    frequency_resolution_hz = 1/time_fft_s

    # The FFT efficiency is based on powers of 2; it is always possible to pad with zeros.
    # Set the record duration, make a power of 2. Note that int rounds down
    time_duration_nd = 2**(int(np.log2(time_duration_s*frequency_sample_rate_hz)))
    # Set the fft duration, make a power of 2
    time_fft_nd = 2**(int(np.log2(time_fft_s*frequency_sample_rate_hz)))

    # The fft frequencies are set by the duration of the fft
    # In this example we only need the positive frequencies
    frequency_fft_pos_hz = np.fft.rfftfreq(time_fft_nd, d=1/frequency_sample_rate_hz)
    fft_index = np.argmin(np.abs(frequency_fft_pos_hz-frequency_center_hz))
    frequency_center_fft_hz = frequency_fft_pos_hz[fft_index]
    frequency_resolution_fft_hz = frequency_sample_rate_hz/time_fft_nd

    # Convert to dimensionless time and frequency, which is typically used in mathematical formulas.
    # Scale by the sample rate.
    # Dimensionless center frequency
    frequency_center = frequency_center_hz/frequency_sample_rate_hz
    frequency_center_fft = frequency_center_fft_hz/frequency_sample_rate_hz
    # Dimensionless time (samples)
    time_nd = np.arange(time_duration_nd)

    # Construct synthetic tone with 2^n points and max FFT amplitude at exact fft frequency
    mic_sig = np.cos(2*np.pi*frequency_center_fft*time_nd)

    # # Compare to synthetic tone with 2^n points and max FFT amplitude NOT at exact fft frequency
    # # It does NOT return unit amplitude (but it's close)
    # mic_sig = np.sin(2*np.pi*frequency_center*time_nd)
    # Add noise
    mic_sig += synthetics.white_noise_fbits(sig=mic_sig, std_bit_loss=12.)
    # Taper before AA
    mic_sig *= utils.taper_tukey(mic_sig, fraction_cosine=0.1)
    # Antialias (AA)
    synthetics.antialias_halfNyquist(mic_sig)

    # Computed and nominal values
    mic_sig_rms = np.std(mic_sig)
    mic_sig_rms_nominal = 1/np.sqrt(2)

    print('Nyquist frequency:', frequency_sample_rate_hz/2)
    print('Nominal signal frequency, hz:', frequency_center_hz)
    print('FFT signal frequency, hz:', frequency_center_fft_hz)
    print('Nominal spectral resolution, hz', frequency_resolution_hz)
    print('FFT spectral resolution, hz', frequency_resolution_fft_hz)
    print('Number of signal points:', time_duration_nd)
    print('log2(points):', np.log2(time_duration_nd))
    print('Number of FFT points:', time_fft_nd)
    print('log2(FFT points):', np.log2(time_fft_nd))
    print('Computed RMS(STD):', mic_sig_rms)
    print('Nominal RMS(STD):', mic_sig_rms_nominal)

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

    # Express in bits; revisit
    # TODO: What units shall we use? Evaluate Stockwell first
    mic_spect_bits = utils.log2epsilon(np.sqrt(psd_spec_power))
    mic_stft_bits = utils.log2epsilon(np.sqrt(stft_power))

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


    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_nd/frequency_sample_rate_hz, mic_sig)
    ax1.set_title('Synthetic CW, with taper')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax2.semilogx(frequency_fft_pos_hz, fft_rms_welch)
    ax2.semilogx(frequency_spect_hz, fft_rms_spect)
    ax2.semilogx(frequency_stft_hz, fft_rms_stft, '.-')
    ax2.set_title('Welch and Spect FFT (RMS), f = ' + str(round(frequency_center_fft_hz*100)/100) + ' Hz')
    ax2.set_xlabel('Frequency, hz')
    ax2.set_ylabel('FFT RMS * sqrt(2)')
    ax2.grid(True)

    # Plot the inverse stft (full recovery)
    plt.figure()
    plt.plot(sig_time_istft, sig_wf_istft)

    # Select plot frequencies
    fmin = 2*frequency_resolution_fft_hz
    fmax = frequency_sample_rate_hz/2  # Nyquist
    pltq.plot_wf_mesh_vert(redvox_id='00',
                           wf_panel_a_sig=mic_sig,
                           wf_panel_a_time=time_nd/frequency_sample_rate_hz,
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
                           wf_panel_a_time=time_nd/frequency_sample_rate_hz,
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

