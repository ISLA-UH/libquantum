"""
libquantum example: s00_stft_tone_intro.py
Compute Welch power spectral density (PSD) on simple tones to verify amplitudes

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import utils
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
    frequency_center_hz = 60.
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

    print('Nyquist frequency:', frequency_sample_rate_hz/2)
    print('Nominal signal frequency, hz:', frequency_center_hz)
    print('FFT signal frequency, hz:', frequency_center_fft_hz)
    print('Nominal spectral resolution, hz', frequency_resolution_hz)
    print('FFT spectral resolution, hz', frequency_resolution_fft_hz)
    print('Number of signal points:', time_duration_nd)
    print('log2(points):', np.log2(time_duration_nd))
    print('Number of FFT points:', time_fft_nd)
    print('log2(FFT points):', np.log2(time_fft_nd))

    # Compute the Welch PSD; averaged spectrum over sliding windows
    frequency_welch_hz, Pxx_spec = signal.welch(x=mic_sig,
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

    fft_rms_abs = np.sqrt(2)*np.sqrt(np.abs(Pxx_spec))

    # Compute the spectrogram with the spectrum option
    mic_stft_frequency_hz, mic_stft_time_s, mic_stft_psd_spec = \
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

    fft_rms_stft = np.sqrt(2)*np.sqrt(np.average(mic_stft_psd_spec, axis=1))

    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_nd/frequency_sample_rate_hz, mic_sig)
    ax1.set_title('Synthetic CW, no taper')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax2.semilogx(frequency_fft_pos_hz, fft_rms_abs)
    ax2.semilogx(mic_stft_frequency_hz, fft_rms_stft)
    ax2.set_title('Welch and Spect FFT (RMS), f = ' + str(round(frequency_center_fft_hz*100)/100) + ' Hz')
    ax2.set_xlabel('Frequency, hz')
    ax2.set_ylabel('FFT RMS * sqrt(2)')
    ax2.grid(True)

    mic_stft_bits = utils.log2epsilon(np.sqrt(np.abs(mic_stft_psd_spec)))
    # Select plot frequencies
    fmin = 2*frequency_resolution_fft_hz
    fmax = frequency_sample_rate_hz/2  # Nyquist
    pltq.plot_wf_mesh_vert(redvox_id='00',
                           wf_panel_a_sig=mic_sig,
                           wf_panel_a_time=time_nd/frequency_sample_rate_hz,
                           mesh_time=mic_stft_time_s,
                           mesh_frequency=mic_stft_frequency_hz,
                           mesh_panel_b_tfr=mic_stft_bits,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="stft for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    plt.show()

