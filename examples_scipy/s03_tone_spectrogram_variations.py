"""
libquantum example: s04_tone_spect_variations.py
Compute spectrogram with different scaling and mode options.
scaling{ ‘density’, ‘spectrum’ }
mode{‘psd’, ‘complex’, ‘magnitude’}
Contract over the columns and compare to Welch power spectral density (PSD) to verify amplitudes.
Tukey taper (w/ alpha) on each Welch and Spectrogram subwindow.
Case study:
Sinusoid input with unit amplitude
Validate:
Welch power averaged over the signal duration is 1/2
RMS amplitude = 1/sqrt(2)

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import benchmark_signals

print(__doc__)

# alpha: Shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered region.
# If zero, the Tukey window is equivalent to a rectangular window.
# If one, the Tukey window is equivalent to a Hann window.
alpha = 0.25
# Changing alpha changes the 'alternate' scalings, so the weights depend on the window

if __name__ == "__main__":
    # Construct a tone of fixed frequency with a constant sample rate as in previous
    # The nominal signal duration is 16s, with averaging (fft) window duration of 1s.
    frequency_tone_hz = 60
    [mic_sig, time_s, time_fft_nd,
     frequency_sample_rate_hz, frequency_center_fft_hz, frequency_resolution_fft_hz] = \
        benchmark_signals.well_tempered_tone(frequency_center_hz=frequency_tone_hz,
                                             add_noise_taper_aa=True)

    # Computed and nominal values
    mic_sig_rms = np.std(mic_sig)
    mic_sig_rms_nominal = 1/np.sqrt(2)

    nfft_center = time_fft_nd

    plt.figure()
    plt.plot(time_s, mic_sig)
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
    mic_spect_frequency_hz, mic_spect_time_s, mic_spect_psd = \
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

    _, _, mic_spect_complex = \
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

    _, _, mic_spect_magnitude = \
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

    _, _, mic_spect_psd_spec = \
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

    _, _, mic_spect_complex_spec = \
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

    _, _, mic_spect_magnitude_spec = \
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

    print('\n*** SUMMARY: Spectrogram and Welch comparisons for a constant-frequency tone  ***')
    print('Scipy STFT, when properly conditioned, is invertible and the preferred form.')
    print('Scipy signal.welch and signal.spectrogram provide compatible estimates.')
    print('Mode psd returns Welch power, whereas mode magnitude complex returns STFT coefficients.')
    print('The Welch power is reproduced by averaging the spectrogram over the time dimension.')
    print('This example uses the STD (e.g. RMS) instead of the variance. '
          'RMS signal amplitude is sometimes used instead of power.')

    plt.figure()
    # Scales with the signal variance
    plt.plot(mic_spect_frequency_hz,
             np.sqrt(np.average(mic_spect_psd_spec, axis=1))/mic_sig_rms, label='spec, psd')
    plt.plot(welch_frequency_hz,
             np.sqrt(Pxx_spec)/mic_sig_rms, '-.', label='spec, Welch')
    plt.plot(mic_spect_frequency_hz,
             np.sqrt(np.average(2*mic_spect_magnitude_spec**2, axis=1))/mic_sig_rms, '.-', label='spec, mag')
    # # The next are the positive spectral coefficients with no padding or boundary extension
    plt.plot(mic_spect_frequency_hz,
             np.sqrt(np.average(2*np.abs(mic_spect_complex_spec)**2, axis=1))/mic_sig_rms, '--', label='spec, complex')
    plt.title('Scaled PSD amplitude returns near-unity at peak: preferred forms')
    plt.xlim(frequency_center_fft_hz-10, frequency_center_fft_hz+10)
    plt.xlabel('Frequency, hz')
    plt.ylabel('FFT RMS/SIG RMS')
    plt.grid(True)
    plt.legend()

    # Power spectral density is scaled by spectral resolution in Hz.
    plt.figure()
    plt.plot(mic_spect_frequency_hz,
             np.sqrt(frequency_resolution_fft_hz*np.average(mic_spect_psd, axis=1))/mic_sig_rms, label='density, psd')
    plt.plot(welch_frequency_hz,
             np.sqrt(frequency_resolution_fft_hz*Pxx)/mic_sig_rms, '-.', label='density, Welch')
    plt.plot(mic_spect_frequency_hz,
             np.sqrt(frequency_resolution_fft_hz*np.average(2*mic_spect_magnitude**2, axis=1))/mic_sig_rms,
             '.-', label='density, mag')
    # # The next are the positive spectral coefficients with no padding or boundary extension
    plt.plot(mic_spect_frequency_hz,
             np.sqrt(frequency_resolution_fft_hz*np.average(2*np.abs(mic_spect_complex)**2, axis=1))/mic_sig_rms,
             '--', label='density, complex')
    plt.title('Spectral Density Scalings')
    plt.xlim(frequency_center_fft_hz-10, frequency_center_fft_hz+10)
    plt.xlabel('Frequency, hz')
    plt.ylabel('sqrt(df) * FFT RMS/SIG RMS')
    plt.grid(True)
    plt.legend()

    plt.show()
