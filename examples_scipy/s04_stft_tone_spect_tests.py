"""
libquantum example: s00_stft_tone_intro.py
Compute Welch power spectral density (PSD) on simple tones to verify amplitudes

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import utils, synthetics
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq

print(__doc__)
EVENT_NAME = 'tone test'

# alpha: Shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered region.
# If zero, the Tukey window is equivalent to a rectangular window.
# If one, the Tukey window is equivalent to a Hann window.
alpha = 0.20
# Changing alpha changes the 'alternate' scalings, so the weights depend on the window

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
    # Add noise
    mic_sig += synthetics.white_noise_fbits(sig=mic_sig, std_bit_loss=12.)
    # Taper before AA
    mic_sig *= utils.taper_tukey(mic_sig, fraction_cosine=0.1)
    # Antialias (AA)
    synthetics.antialias_halfNyquist(mic_sig)

    print('Nyquist frequency:', frequency_sample_rate_hz/2)
    print('Nominal signal frequency, hz:', frequency_center_hz)
    print('FFT signal frequency, hz:', frequency_center_fft_hz)
    print('Nominal spectral resolution, hz', frequency_resolution_hz)
    print('FFT spectral resolution, hz', frequency_resolution_fft_hz)
    print('Number of signal points:', time_duration_nd)
    print('log2(points):', np.log2(time_duration_nd))
    print('Number of FFT points:', time_fft_nd)
    print('log2(FFT points):', np.log2(time_fft_nd))

    nfft_center = time_fft_nd

    plt.figure()
    plt.plot(time_nd/frequency_sample_rate_hz, mic_sig)
    plt.title('Synthetic sinusoid with unit amplitude')

    # Welch
    welch_frequency_hz, Pxx = signal.welch(x=mic_sig,
                                           fs=frequency_sample_rate_hz,
                                           window=('tukey', alpha),
                                           nperseg=nfft_center,
                                           noverlap=nfft_center // 2,
                                           nfft=nfft_center,
                                           detrend='constant',
                                           return_onesided=True,
                                           axis=-1,
                                           scaling='density',
                                           average='mean')

    _, Pxx_spec = signal.welch(x=mic_sig,
                               fs=frequency_sample_rate_hz,
                               window=('tukey', alpha),
                               nperseg=nfft_center,
                               noverlap=nfft_center // 2,
                               nfft=nfft_center,
                               detrend='constant',
                               return_onesided=True,
                               axis=-1,
                               scaling='spectrum',
                               average='mean')

    # Spectrogram
    mic_stft_frequency_hz, mic_stft_time_s, mic_stft_psd = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='density',
                           mode='psd')

    _, _, mic_stft_complex = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='density',
                           mode='complex')

    _, _, mic_stft_magnitude = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='density',
                           mode='magnitude')

    _, _, mic_stft_psd_spec = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='spectrum',
                           mode='psd')

    _, _, mic_stft_complex_spec = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='spectrum',
                           mode='complex')

    _, _, mic_stft_magnitude_spec = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='spectrum',
                           mode='magnitude')

    plt.figure()
    # These appear to be the most predictable. The first two are power of the positive frequencies
    plt.plot(mic_stft_frequency_hz, np.sqrt(2*np.average(mic_stft_psd_spec, axis=1)), label='spec, psd')
    plt.plot(welch_frequency_hz, np.sqrt(2*Pxx_spec), '-.', label='spec, Pxx psd')
    plt.plot(mic_stft_frequency_hz, np.sqrt(2*np.average(mic_stft_magnitude_spec, axis=1)), label='spec, mag')
    # The next are the positive FFT coefficients, and need a factor of 2 in amplitude
    plt.plot(mic_stft_frequency_hz, np.average(2*np.abs(mic_stft_complex_spec), axis=1), label='spec, complex')
    plt.title('Scaled PSD amplitude returns near-unity at peak: preferred forms')
    plt.xlim(frequency_center_fft_hz-10, frequency_center_fft_hz+10)
    plt.xlabel('Frequency, hz')
    plt.ylabel('Scaled FFT RMS')
    plt.grid(True)
    plt.legend()

    # The next ones have other scaling factors which can be explored
    plt.figure()
    plt.plot(mic_stft_frequency_hz,
             np.sqrt(frequency_resolution_fft_hz)*np.sqrt(2*np.average(mic_stft_psd, axis=1)), label='density, psd')
    plt.plot(welch_frequency_hz,
             np.sqrt(frequency_resolution_fft_hz)*np.sqrt(2*Pxx), '-.', label='density, Pxx psd')
    plt.plot(mic_stft_frequency_hz,
             np.sqrt(frequency_resolution_fft_hz*2*np.average(mic_stft_magnitude, axis=1)), label='density, mag')
    # # The next are the positive FFT coefficients, and need a factor of 2 in amplitude
    plt.plot(mic_stft_frequency_hz,
             frequency_resolution_fft_hz*np.average(2*np.abs(mic_stft_complex), axis=1), label='density, complex')
    plt.title('Alternate PSD scalings depend on Tukey alpha')
    plt.xlim(frequency_center_fft_hz-10, frequency_center_fft_hz+10)
    plt.xlabel('Frequency, hz')
    plt.ylabel('Scaled FFT RMS')
    plt.grid(True)
    plt.legend()

    plt.show()
